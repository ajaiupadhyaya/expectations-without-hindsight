.PHONY: install test test-fast lint manifest verify-raw aklmt noise-mc spf-replication tokenization dryrun all
.DELETE_ON_ERROR:

install:
	uv sync

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

lint:
	uv run ruff check ewh tests
	uv run ruff format --check ewh tests

manifest:
	uv run python -m ewh.provenance.manifest build

verify-raw:
	uv run python -m ewh.provenance.manifest verify

aklmt:
	uv run python -m ewh.data.aklmt

noise-mc:
	uv run python -m ewh.sim.noise_mc

tokenization:
	uv run python -m ewh.prompts.tokenization

dryrun:
	uv run python -m ewh.dryrun
	cd paper && latexmk -pdf -interaction=nonstopmode -outdir=build dryrun.tex >/dev/null

spf-replication:
	uv run python -m ewh.data.spf_replication

# Grows phase by phase: data -> estimation -> tables/figures -> paper.
all: verify-raw test aklmt noise-mc spf-replication
