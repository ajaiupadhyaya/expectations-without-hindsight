.PHONY: install test lint manifest verify-raw all
.DELETE_ON_ERROR:

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check ewh tests
	uv run ruff format --check ewh tests

manifest:
	uv run python -m ewh.provenance.manifest build

verify-raw:
	uv run python -m ewh.provenance.manifest verify

# Grows phase by phase: data -> estimation -> tables/figures -> paper.
all: verify-raw test
