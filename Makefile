.PHONY: format lint typecheck ci

## format: auto-format all source code with Black
format:
	python -m black src/ tests/

## lint: lint and fix with Ruff (style, docstrings, annotations)
lint:
	python -m ruff check src/ tests/ --fix

## typecheck: run mypy in strict-ish mode
typecheck:
	python -m mypy -p ors

## ci: run all quality checks (format check, lint, typecheck)
##     suitable for CI pipelines – does not auto-fix, exits non-zero on failure
ci:
	python -m black --check src/ tests/
	python -m ruff check src/ tests/
	python -m mypy -p ors
