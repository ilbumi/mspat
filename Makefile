.DEFAULT_GOAL := help
SHELL=bash
CODE_PATHS=packages/ scripts/
TESTS_PATH=tests/

.PHONY: test
test:
	uv run pytest -n 5 --cov .

.PHONY: format
format:
	uv run ssort ${CODE_PATHS}
	uv run isort ${CODE_PATHS}
	uv run ruff format ${CODE_PATHS} ${TESTS_PATH}
	uv run ruff check --fix ${CODE_PATHS} ${TESTS_PATH}

.PHONY: lint
lint:
	uv run ruff check ${CODE_PATHS} ${TESTS_PATH}
	uv run mypy ${CODE_PATHS}

.PHONY: sync
sync:
	uv sync

.PHONY: lock
lock:
	uv lock

.PHONY: docs
docs:
	sphinx-build docs docs/_build
