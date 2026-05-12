"""
Web-based tool definitions and implementations for search agent.

This module provides web search and scraping tools using Serper API and Jina,
with URL-to-ID mapping for clean agent interaction.
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    cast,
)
import json
import os
import random
import time

import openai
import requests
import tiktoken
from pydantic import BaseModel, Field
import structlog

from harness.rerank import Reranker
from harness.tools import (
    GREP_CORPUS_SCHEMA,
    PRUNE_CHUNKS_SCHEMA,
    READ_DOCUMENT_SCHEMA,
    SEARCH_CORPUS_SCHEMA,
    GrepCorpusToolCallMetadata,
    PruneChunksTool,
    SearchCorpusToolCallMetadata,
    Tool,
    ToolCallMetadata,
    ToolSchema,
    ToolSet,
)

logger = structlog.get_logger("search_agent.web_tools")

# ============================================================================
# Constants
# ============================================================================

MAX_PAGE_TOKENS = 10000  # Maximum tokens before triggering long page handling
CHUNK_SIZE_TOKENS = 512  # Target chunk size for semantic chunking
TOP_K_CHUNKS = 10  # Number of chunks to return from semantic search
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_LIMIT = 250_000  # Max tokens per embedding batch

# ============================================================================
# Shared Session for Connection Pooling
# ============================================================================

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create a shared requests session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


# ============================================================================
# URL Mapper
# ============================================================================


def normalize_url(url: str) -> str:
    """Normalize a URL by stripping fragment identifiers.

    URLs may contain fragment identifiers like #section or #:~:text=... (Chrome text fragments).
    These should be stripped for comparison purposes since they refer to the same document.

    Args:
        url: The URL to normalize.

    Returns:
        The URL with any fragment identifier removed.
    """
    return url.split("#")[0]


class URLMapper:
    """Bidirectional mapping between URLs and short integer IDs.

    This class manages a mapping between URLs (which can be long and unwieldy)
    and short integer IDs that are easier for agents to work with. IDs are
    randomly generated in the range 0-100000 to avoid sequential patterns.

    URLs are normalized by stripping fragment identifiers (#...) before storage,
    so URLs like "https://example.com/page#:~:text=foo" and "https://example.com/page"
    will map to the same ID.
    """

    def __init__(self, seed: int = 0):
        self._url_to_id: Dict[str, str] = {}  # URL -> "42"
        self._id_to_url: Dict[str, str] = {}  # "42" -> URL
        self._rng = random.Random(seed)
        self._used_ids: Set[int] = set()

    def get_or_create_id(self, url: str) -> str:
        """Get existing ID for URL or create new random ID (0-100000).

        Args:
            url: The URL to map. Will be normalized (fragment stripped) before storage.

        Returns:
            A string ID (e.g., "42") that maps to the normalized URL.
        """
        # Normalize URL by stripping fragment identifiers
        normalized_url = normalize_url(url)

        if normalized_url in self._url_to_id:
            return self._url_to_id[normalized_url]

        # Generate a random ID that hasn't been used
        max_attempts = 1000
        for _ in range(max_attempts):
            new_id = self._rng.randint(0, 100000)
            if new_id not in self._used_ids:
                break
        else:
            # Fallback: use sequential ID if random fails
            new_id = len(self._used_ids)

        self._used_ids.add(new_id)
        id_str = str(new_id)
        self._url_to_id[normalized_url] = id_str
        self._id_to_url[id_str] = normalized_url
        return id_str

    def resolve_id(self, doc_id: str) -> Optional[str]:
        """Resolve an ID back to its URL.

        Args:
            doc_id: The string ID to resolve.

        Returns:
            The URL corresponding to the ID, or None if not found.
        """
        return self._id_to_url.get(doc_id)

    def get_mapping(self) -> Dict[str, str]:
        """Return ID-to-URL mapping for JSON serialization."""
        return self._id_to_url.copy()

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Return URL-to-ID mapping."""
        return self._url_to_id.copy()

    def __len__(self) -> int:
        """Return number of mapped URLs."""
        return len(self._url_to_id)


# ============================================================================
# Token Counting and Text Chunking Utilities
# ============================================================================

_tiktoken_encoder: Optional[tiktoken.Encoding] = None


def _get_tiktoken_encoder() -> tiktoken.Encoding:
    """Get or create tiktoken encoder (cl100k_base for OpenAI embeddings)."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken cl100k_base encoder."""
    enc = _get_tiktoken_encoder()
    return len(enc.encode(text))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS) -> List[str]:
    """Split text into approximately equal-sized token chunks.

    Args:
        text: The text to chunk.
        chunk_size: Target number of tokens per chunk.

    Returns:
        List of text chunks.
    """
    enc = _get_tiktoken_encoder()
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def get_embeddings(
    texts: List[str],
    openai_client: Optional[openai.OpenAI] = None,
) -> List[List[float]]:
    """Get embeddings for texts using OpenAI API with batching.

    Handles token limits by batching texts appropriately.
 
    Args:
        texts: List of texts to embed.
        openai_client: Optional OpenAI client. Creates new one if not provided.

    Returns:
        List of embedding vectors.
    """
    if openai_client is None:
        openai_client = openai.OpenAI()

    enc = _get_tiktoken_encoder()
    embeddings: List[List[float]] = []

    # Process in batches to avoid token limits
    current_batch: List[str] = []
    current_batch_tokens = 0

    for text in texts:
        text_tokens = len(enc.encode(text))

        # If this text would exceed batch limit, process current batch first
        if current_batch and current_batch_tokens + text_tokens > EMBEDDING_BATCH_LIMIT:
            resp = openai_client.embeddings.create(
                model=EMBEDDING_MODEL, input=current_batch, encoding_format="float"
            )
            embeddings.extend([e.embedding for e in resp.data])
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += text_tokens

    # Process remaining batch
    if current_batch:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=current_batch, encoding_format="float"
        )
        embeddings.extend([e.embedding for e in resp.data])

    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ============================================================================
# Long Page Handling
# ============================================================================


def search_long_page(
    page_text: str,
    query: str,
    top_k: int = TOP_K_CHUNKS,
    openai_client: Optional[openai.OpenAI] = None,
) -> List[Tuple[str, float]]:
    """Search a long page using semantic chunking and embedding similarity.

    Args:
        page_text: The full page text.
        query: The search query.
        top_k: Number of top chunks to return.
        openai_client: Optional OpenAI client.

    Returns:
        List of (chunk_text, similarity_score) tuples, sorted by relevance.
    """
    log = logger.bind(function="search_long_page")

    # Chunk the page
    chunks = chunk_text(page_text)
    log.info("page_chunked", num_chunks=len(chunks))

    if not chunks:
        return []

    # Get embeddings for chunks and query
    all_texts = chunks + [query]
    embeddings = get_embeddings(all_texts, openai_client)

    chunk_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Calculate similarities
    similarities = [
        (chunk, cosine_similarity(chunk_emb, query_embedding))
        for chunk, chunk_emb in zip(chunks, chunk_embeddings)
    ]

    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def ask_agent_for_page_query(
    url: str,
    page_preview: str,
    original_query: str,
    openai_client: Optional[openai.OpenAI] = None,
) -> Optional[str]:
    """Ask the agent (via LLM) for a search query to find relevant content in a long page.

    Args:
        url: The URL of the page.
        page_preview: A preview of the page content (first ~1000 chars).
        original_query: The original search query that led to this page.
        openai_client: Optional OpenAI client.

    Returns:
        A search query string, or None if the LLM couldn't provide one.
    """
    if openai_client is None:
        openai_client = openai.OpenAI()

    prompt = f"""You are helping an agent search a long webpage. The agent is looking for information related to:

Original query: {original_query}

The page at {url} is very long. Here's a preview of its content:

{page_preview[:1000]}...

Please provide a short, focused search query (3-7 words) that would help find the most relevant sections of this page. Just respond with the query, nothing else."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )
        query = response.choices[0].message.content
        if query:
            return query.strip().strip('"').strip("'")
        return None
    except Exception as e:
        logger.warning("ask_agent_for_query_failed", error=str(e))
        return None


def handle_long_page(
    page_text: str,
    url: str,
    original_query: Optional[str] = None,
    openai_client: Optional[openai.OpenAI] = None,
    max_tokens: int = MAX_PAGE_TOKENS,
) -> str:
    """Handle a long page by semantic chunking or truncation.

    If the page exceeds max_tokens:
    1. Try to get a search query (from original_query or by asking LLM)
    2. If query available, do semantic search over chunks
    3. Otherwise, truncate to max_tokens

    Args:
        page_text: The full page text.
        url: The page URL (for context in LLM query).
        original_query: Optional query that led to this page.
        openai_client: Optional OpenAI client.
        max_tokens: Maximum tokens before triggering long page handling.

    Returns:
        Processed page content (either semantically selected chunks or truncated).
    """
    log = logger.bind(function="handle_long_page", url=url)
    page_tokens = count_tokens(page_text)

    if page_tokens <= max_tokens:
        return page_text

    log.info("page_exceeds_limit", page_tokens=page_tokens, max_tokens=max_tokens)

    # Try to get a search query
    search_query = original_query
    if not search_query:
        search_query = ask_agent_for_page_query(
            url, page_text[:2000], original_query or "", openai_client
        )

    if search_query:
        log.info("using_semantic_search", query=search_query)
        results = search_long_page(page_text, search_query, openai_client=openai_client)

        if results:
            combined_chunks = []
            for chunk, score in results:
                combined_chunks.append(chunk)

            return "\n\n---\n\n".join(combined_chunks)

    # Fallback: truncate
    log.info("falling_back_to_truncation")
    enc = _get_tiktoken_encoder()
    tokens = enc.encode(page_text)[:max_tokens]
    return enc.decode(tokens) + "\n\n[Content truncated due to length]"


# ============================================================================
# Web Tool Implementations
# ============================================================================


class WebSearchCorpusTool(Tool):
    """A tool that searches the web using Serper API.

    Returns search results with URLs mapped to short IDs via URLMapper.
    """

    tool_schema: ToolSchema
    _reranker: Optional[Reranker] = None
    _token_counter: Optional[Callable[[str], int]] = None

    def __init__(
        self,
        reranker: Optional[Reranker] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        super().__init__(tool_schema=SEARCH_CORPUS_SCHEMA)
        self._reranker = reranker
        self._token_counter = token_counter

    def _search_serper(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Execute search via Serper API."""
        session = _get_session()
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query, "num": num_results})
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY", ""),
            "Content-Type": "application/json",
        }

        try:
            response = session.post(url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "organic" not in data:
                return []

            results = []
            for result in data["organic"]:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                    }
                )
            return results
        except requests.RequestException as e:
            logger.warning("serper_search_error", error=str(e))
            return []

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[SearchCorpusToolCallMetadata]]:
        log = logger.bind(tool=self.tool_schema.name)

        if not isinstance(params, dict) or "query" not in params:
            log.error("invalid_params", params_type=type(params).__name__)
            raise ValueError(f"Invalid params type: {type(params)}")

        query = params["query"]
        ignore_ids: List[str] = []
        url_mapper: Optional[URLMapper] = None

        if overrides is not None:
            ignore_ids = overrides.get("ignore_ids", [])
            url_mapper = overrides.get("url_mapper")

        log.info("web_search_corpus", query=query, ignore_ids_count=len(ignore_ids))

        # Perform search
        results = self._search_serper(query)

        if not results:
            return ("No results found", SearchCorpusToolCallMetadata(returned_chunk_ids=[]))

        # Map URLs to IDs and filter ignored
        urls = [r["link"] for r in results]
        documents = [r["title"] + "\n" + r["snippet"] for r in results]

        # Map URLs to IDs
        if url_mapper is not None:
            mapped_ids = [url_mapper.get_or_create_id(url) for url in urls]
        else:
            # Fallback: use URLs as IDs
            mapped_ids = urls

        # Filter out ignored IDs
        filtered_results = []
        for mapped_id, url, doc in zip(mapped_ids, urls, documents):
            if mapped_id not in ignore_ids:
                filtered_results.append((mapped_id, url, doc))

        if not filtered_results:
            return (
                "No new results found (all results already seen)",
                SearchCorpusToolCallMetadata(returned_chunk_ids=[]),
            )

        # Unpack filtered results
        mapped_ids = [r[0] for r in filtered_results]
        documents = [r[2] for r in filtered_results]

        # Get max_tokens override if provided
        max_tokens_override = (
            overrides.get("max_tokens") if overrides and "max_tokens" in overrides else None
        )

        # Rerank if reranker provided
        token_counts: List[Optional[int]] = [None] * len(mapped_ids)
        if self._reranker is not None:
            rerank_results = self._reranker(
                query, cast(List[str], documents), max_tokens=max_tokens_override
            )
            reranked_ids = [mapped_ids[r.original_index] for r in rerank_results]
            reranked_documents = [r.document for r in rerank_results]
            token_counts = [r.tokens for r in rerank_results]
            mapped_ids = reranked_ids
            documents = reranked_documents
            log.info("reranked_results", num_results=len(mapped_ids))

        # Format output (top 5)
        formatted = []
        for doc_id, doc, tokens in list(zip(mapped_ids, documents, token_counts))[:5]:
            token_str = f" ({tokens} tokens)" if tokens is not None else ""
            formatted.append(f"\n# DOCUMENT ID: {doc_id}{token_str}\n{doc}")

        return (
            "\n".join(formatted) if formatted else "No results found",
            SearchCorpusToolCallMetadata(returned_chunk_ids=mapped_ids[: len(formatted)]),
        )


class WebGrepCorpusTool(Tool):
    """A tool that performs pattern-based search using Serper API.

    Note: Serper doesn't support regex, so we use the pattern as a query.
    This provides similar behavior for finding relevant content.
    """

    tool_schema: ToolSchema
    _token_counter: Optional[Callable[[str], int]] = None

    def __init__(
        self,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        super().__init__(tool_schema=GREP_CORPUS_SCHEMA)
        self._token_counter = token_counter

    def _search_serper(self, pattern: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Execute search via Serper API using pattern as query."""
        session = _get_session()
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": pattern, "num": num_results})
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY", ""),
            "Content-Type": "application/json",
        }

        try:
            response = session.post(url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "organic" not in data:
                return []

            results = []
            for result in data["organic"]:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                    }
                )
            return results
        except requests.RequestException as e:
            logger.warning("serper_grep_error", error=str(e))
            return []

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[GrepCorpusToolCallMetadata]]:
        log = logger.bind(tool=self.tool_schema.name)

        if not isinstance(params, dict) or "pattern" not in params:
            log.error("invalid_params", params_type=type(params).__name__)
            raise ValueError(f"Invalid params type: {type(params)}")

        pattern = params["pattern"]
        url_mapper: Optional[URLMapper] = None

        if overrides is not None:
            url_mapper = overrides.get("url_mapper")

        log.info("web_grep_corpus", pattern=pattern)

        # Perform search
        results = self._search_serper(pattern)

        if not results:
            return ("No results found", GrepCorpusToolCallMetadata(returned_chunk_ids=[]))

        # Map URLs to IDs
        urls = [r["link"] for r in results]
        documents = [r["title"] + "\n" + r["snippet"] for r in results]

        if url_mapper is not None:
            mapped_ids = [url_mapper.get_or_create_id(url) for url in urls]
        else:
            mapped_ids = urls

        # Calculate token counts if available
        token_counts: List[Optional[int]] = (
            [self._token_counter(doc) for doc in documents]
            if self._token_counter is not None
            else [None] * len(documents)
        )

        # Format output
        formatted = []
        for doc_id, doc, tokens in zip(mapped_ids, documents, token_counts):
            token_str = f" ({tokens} tokens)" if tokens is not None else ""
            formatted.append(f"\n# DOCUMENT ID: {doc_id}{token_str}\n{doc}")

        return (
            "\n".join(formatted) if formatted else "No results found",
            GrepCorpusToolCallMetadata(returned_chunk_ids=mapped_ids),
        )


class WebReadDocumentTool(Tool):
    """A tool that reads web page content using Serper scraping and Jina fallback.

    Takes a mapped ID, resolves to URL via URLMapper, and fetches page content.
    Handles long pages with semantic chunking.
    """

    tool_schema: ToolSchema
    _reranker: Optional[Reranker] = None
    _token_counter: Optional[Callable[[str], int]] = None
    _max_tokens: Optional[int] = None
    _openai_client: Optional[openai.OpenAI] = None
    max_retries: int = 3

    def __init__(
        self,
        reranker: Optional[Reranker] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
        openai_client: Optional[openai.OpenAI] = None,
    ) -> None:
        if max_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_tokens is specified")
        super().__init__(tool_schema=READ_DOCUMENT_SCHEMA)
        self._reranker = reranker
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._openai_client = openai_client

    def _fetch_page_serper(self, url: str) -> Optional[str]:
        """Fetch page content using Serper scraping API."""
        session = _get_session()
        payload = {"url": url}
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY", ""),
            "Content-Type": "application/json",
        }

        try:
            response = session.post(
                "https://scrape.serper.dev", headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if "text" in data:
                return data["text"]
            return None
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("serper_scrape_error", url=url, error=str(e))
            return None

    def _fetch_page_jina(self, url: str) -> Optional[str]:
        """Backup page fetcher using Jina."""
        session = _get_session()
        full_url = f"https://r.jina.ai/{url}"
        headers = {
            "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}",
            "X-Retain-Images": "none",
            "X-Remove-Selector": "header, .class, #id, nav, footer, aside, .sidebar, .comments, .social",
        }

        try:
            response = session.get(full_url, headers=headers, timeout=30)
            response.raise_for_status()
            text = response.text
            marker = "Markdown Content:\n"
            if marker in text:
                return text.split(marker, 1)[1]
            return text
        except requests.RequestException as e:
            logger.warning("jina_fetch_error", url=url, error=str(e))
            return None

    def __call__(
        self, params: Dict[Any, Any], overrides: Optional[Dict[Any, Any]] = None
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        log = logger.bind(tool=self.tool_schema.name)

        if not isinstance(params, dict) or ("doc_id" not in params and "id" not in params):
            log.error("invalid_params", params_type=type(params).__name__)
            raise ValueError(f"Invalid params type: {type(params)}")

        doc_id = params.get("doc_id") or params.get("id", "")
        url_mapper: Optional[URLMapper] = None
        original_query: Optional[str] = None

        if overrides is not None:
            url_mapper = overrides.get("url_mapper")
            original_query = overrides.get("query")

        # Resolve ID to URL
        if url_mapper is not None:
            url = url_mapper.resolve_id(doc_id)
            if url is None:
                # Maybe it's already a URL
                if doc_id.startswith("http"):
                    url = doc_id
                else:
                    return (f"Error: Document ID '{doc_id}' not found", None)
        else:
            url = doc_id

        log.info("web_read_document", doc_id=doc_id, url=url)

        # Try to fetch page with retries
        page_text = None
        last_error = None

        for attempt in range(self.max_retries):
            # Try Serper first
            page_text = self._fetch_page_serper(url)
            if page_text:
                break

            # Try Jina backup
            page_text = self._fetch_page_jina(url)
            if page_text:
                break

            if attempt < self.max_retries - 1:
                wait_time = 2**attempt
                log.warning(
                    "fetch_retry",
                    url=url,
                    attempt=attempt + 1,
                    wait_time=wait_time,
                )
                time.sleep(wait_time)

        if not page_text:
            return (f"Error fetching page: Could not retrieve content from {url}", None)

        # Get max_tokens from overrides or instance default
        max_tokens = (
            overrides.get("max_tokens") if overrides and "max_tokens" in overrides else None
        ) or self._max_tokens

        # Handle long pages
        if self._token_counter is not None:
            page_tokens = self._token_counter(page_text)

            if max_tokens is not None and page_tokens > max_tokens:
                log.info(
                    "page_exceeds_limit",
                    page_tokens=page_tokens,
                    max_tokens=max_tokens,
                )

                # Use semantic chunking if we have a query
                if original_query and self._openai_client:
                    page_text = handle_long_page(
                        page_text,
                        url,
                        original_query,
                        self._openai_client,
                        max_tokens,
                    )
                else:
                    # Truncate to max_tokens
                    enc = _get_tiktoken_encoder()
                    tokens = enc.encode(page_text)[:max_tokens]
                    page_text = (
                        enc.decode(tokens) + "\n\n[Content truncated due to length]"
                    )

        # Add token count header
        if self._token_counter is not None:
            doc_tokens = self._token_counter(page_text)
            return (f"# Document ({doc_tokens} tokens)\n{page_text}", None)

        return (page_text, None)


# ============================================================================
# Web ToolSet Factory
# ============================================================================


class WebToolSet(ToolSet):
    """A ToolSet for web-based search tools."""

    @classmethod
    def create(
        cls,
        reranker: Optional[Reranker] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
        openai_client: Optional[openai.OpenAI] = None,
    ) -> "WebToolSet":
        """Create a WebToolSet with web-based tools.

        Args:
            reranker: Optional reranker for reordering search results.
            token_counter: Optional callable that counts tokens in a string.
            max_tokens: Maximum tokens for ReadDocumentTool output.
            openai_client: Optional OpenAI client for embeddings.

        Returns:
            A configured WebToolSet instance.
        """
        toolset = cls(name="web_toolset")

        toolset.add_tool(
            WebSearchCorpusTool(
                reranker=reranker,
                token_counter=token_counter,
            )
        )

        toolset.add_tool(
            WebGrepCorpusTool(
                token_counter=token_counter,
            )
        )

        toolset.add_tool(
            WebReadDocumentTool(
                reranker=reranker,
                token_counter=token_counter,
                max_tokens=max_tokens,
                openai_client=openai_client,
            )
        )

        toolset.add_tool(PruneChunksTool())

        return toolset
