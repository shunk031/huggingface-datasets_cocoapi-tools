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
install-api:
	poetry install --extras datasets

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
