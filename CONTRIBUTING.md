# Contributing

1. Fork and create a branch.
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. Make changes; add or update tests in `tests/`.
4. Run `ruff check .` and `pytest -q` before opening a pull request.

Prompt changes belong in `src/prompts.py`; please include before/after evaluation numbers in the PR description.
