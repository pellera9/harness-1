# Smoke Tests

Run from the repository root:

```bash
uv run python tests/smoke_imports.py
uv run python tests/smoke_cli.py
```

These tests check importability and command-line wiring. They do not run full
Tinker training, retrieval evaluation, or GPU model generation.
