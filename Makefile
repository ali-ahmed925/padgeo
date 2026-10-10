.PHONY: check lint test format

# One command for everything the README promises: formatting, lint, type checks, tests.
check: lint test

lint:
	ruff format --check .
	ruff check .
	mypy

test:
	pytest

format:
	ruff format .
	ruff check --fix .
