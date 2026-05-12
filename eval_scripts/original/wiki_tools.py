"""
Wikipedia-based tool definitions and implementations for search agent.

This module provides Wikipedia-specific search and reading tools using Serper API
(with site:wikipedia.org filter) for search and the `wikipediaapi` library for
fetching page content. URL-to-ID mapping reuses URLMapper from web_tools.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)
import json
import os
import requests
import urllib.parse

import wikipediaapi
import structlog

from harness.rerank import Reranker
from harness.tools import (
    GREP_CORPUS_SCHEMA,
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
from web_tools import (
    URLMapper,
    normalize_url,
    _get_session,
    count_tokens,
    _get_tiktoken_encoder,
    handle_long_page,
    MAX_PAGE_TOKENS,
)

logger = structlog.get_logger("search_agent.wiki_tools")

# ============================================================================
# Helper: extract Wikipedia title from URL
# ============================================================================


def extract_wiki_title(url: str) -> Optional[str]:
    """Extract the article title from a Wikipedia URL.

    Args:
        url: A Wikipedia URL, e.g. "https://en.wikipedia.org/wiki/Title_Here"

    Returns:
        The decoded article title, or None for non-Wikipedia URLs.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if "wikipedia.org" not in parsed.netloc:
            return None
        path = parsed.path
        if "/wiki/" not in path:
            return None
        title = path.split("/wiki/", 1)[1]
        return urllib.parse.unquote(title).replace("_", " ")
    except Exception:
        return None


# ============================================================================
# Wikipedia Tool Implementations
# ============================================================================


class WikiSearchCorpusTool(Tool):
    """A tool that searches Wikipedia using Serper API with site:wikipedia.org filter.

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
        """Execute search via Serper API, restricted to Wikipedia."""
        session = _get_session()
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": f"site:wikipedia.org {query}", "num": num_results})
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
        except Exception as e:
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

        log.info("wiki_search_corpus", query=query, ignore_ids_count=len(ignore_ids))

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


class WikiGrepCorpusTool(Tool):
    """A tool that performs pattern-based search on Wikipedia using Serper API.

    Uses site:wikipedia.org filter to restrict results to Wikipedia.
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
        """Execute search via Serper API using pattern as query, restricted to Wikipedia."""
        session = _get_session()
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": f"site:wikipedia.org {pattern}", "num": num_results})
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
        except Exception as e:
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

        log.info("wiki_grep_corpus", pattern=pattern)

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


class WikiReadDocumentTool(Tool):
    """A tool that reads Wikipedia page content using the wikipediaapi library.

    Takes a mapped ID, resolves to URL via URLMapper, extracts the article title,
    and fetches content via wikipediaapi. Supports summary-only mode.
    """

    tool_schema: ToolSchema
    _reranker: Optional[Reranker] = None
    _token_counter: Optional[Callable[[str], int]] = None
    _max_tokens: Optional[int] = None
    _openai_client: Any = None
    _summary_only: bool = False
    _wiki_agent: Any = None
    max_retries: int = 3

    def __init__(
        self,
        reranker: Optional[Reranker] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
        openai_client: Any = None,
        summary_only: bool = False,
    ) -> None:
        if max_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_tokens is specified")
        super().__init__(tool_schema=READ_DOCUMENT_SCHEMA)
        self._reranker = reranker
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._openai_client = openai_client
        self._summary_only = summary_only
        self._wiki_agent = wikipediaapi.Wikipedia(
            user_agent="harness-1", language="en"
        )

    def _fetch_page_wikipedia(self, url: str) -> Optional[str]:
        """Fetch page content using the wikipediaapi library.

        Args:
            url: A Wikipedia URL to fetch.

        Returns:
            Formatted page content, or None if the page doesn't exist.
        """
        title = extract_wiki_title(url)
        if title is None:
            logger.warning("not_a_wikipedia_url", url=url)
            return None

        try:
            page = self._wiki_agent.page(title)
            if not page.exists():
                logger.warning("wikipedia_page_not_found", title=title)
                return None

            if self._summary_only:
                return f"# {page.title}\n\n{page.summary}"
            else:
                return f"# {page.title}\n\n## Summary\n{page.summary}\n\n## Full Text\n{page.text}"
        except KeyError as e:
            logger.warning("wikipedia_api_key_error", title=title, url=url, error=str(e))
            return None

    def _fetch_page_serper(self, url: str) -> Optional[str]:
        """Fallback: fetch page content using Serper scraping API."""
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
        """Fallback: fetch page content using Jina."""
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
                if doc_id.startswith("http"):
                    url = doc_id
                else:
                    return (f"Error: Document ID '{doc_id}' not found", None)
        else:
            url = doc_id

        log.info("wiki_read_document", doc_id=doc_id, url=url)

        # Try to fetch page: wikipediaapi -> serper scrape -> jina
        page_text = None

        for attempt in range(self.max_retries):
            page_text = self._fetch_page_wikipedia(url)
            if page_text:
                break

            # Fallback to Serper scrape
            page_text = self._fetch_page_serper(url)
            if page_text:
                log.info("fallback_serper_succeeded", url=url)
                break

            # Fallback to Jina
            page_text = self._fetch_page_jina(url)
            if page_text:
                log.info("fallback_jina_succeeded", url=url)
                break

            if attempt < self.max_retries - 1:
                import time

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

                if original_query and self._openai_client:
                    page_text = handle_long_page(
                        page_text,
                        url,
                        original_query,
                        self._openai_client,
                        max_tokens,
                    )
                else:
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
# Wikipedia ToolSet Factory
# ============================================================================


class WikiToolSet(ToolSet):
    """A ToolSet for Wikipedia-based search tools."""

    @classmethod
    def create(
        cls,
        reranker: Optional[Reranker] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
        openai_client: Any = None,
        summary_only: bool = False,
    ) -> "WikiToolSet":
        """Create a WikiToolSet with Wikipedia-based tools.

        Args:
            reranker: Optional reranker for reordering search results.
            token_counter: Optional callable that counts tokens in a string.
            max_tokens: Maximum tokens for ReadDocumentTool output.
            openai_client: Optional OpenAI client for embeddings.
            summary_only: If True, WikiReadDocumentTool returns only title + summary.

        Returns:
            A configured WikiToolSet instance.
        """
        toolset = cls(name="wiki_toolset")

        toolset.add_tool(
            WikiSearchCorpusTool(
                reranker=reranker,
                token_counter=token_counter,
            )
        )

        toolset.add_tool(
            WikiGrepCorpusTool(
                token_counter=token_counter,
            )
        )

        toolset.add_tool(
            WikiReadDocumentTool(
                reranker=reranker,
                token_counter=token_counter,
                max_tokens=max_tokens,
                openai_client=openai_client,
                summary_only=summary_only,
            )
        )

        toolset.add_tool(PruneChunksTool())

        return toolset
