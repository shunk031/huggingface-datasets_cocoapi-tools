#
# Installation
#

.PHONY: setup
setup:
	pip install -U --no-cache-dir pip setuptools wheel poetry

.PHONY: install
install:
	poetry install

.PHONY: install-datasets
install-datasets:
	poetry install --extras datasets

.PHONY: install-cocoapi
install-cocoapi:
	poetry install --extras cocoapi

.PHONY: install-all
install-all:
	poetry install --extras all

#
# linter/formatter/typecheck
#

.PHONY: lint
lint: install
	poetry run ruff check --output-format=github .

.PHONY: format
format: install
	poetry run ruff format --check --diff .

.PHONY: typecheck
typecheck: install
	poetry run mypy --cache-dir=/dev/null .
