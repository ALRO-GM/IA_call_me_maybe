# call me maybe - common tasks
# Extra program options can be passed with ARGS, e.g.
#   make run ARGS="--input my_tests.json --visualize"

UV      := uv
PYTHON  := $(UV) run python
MODULE  := src
ARGS    ?=

MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
              --ignore-missing-imports --disallow-untyped-defs \
              --check-untyped-defs

.PHONY: all install run debug test clean lint lint-strict help

all: install

install:
	$(UV) sync

run:
	$(PYTHON) -m $(MODULE) $(ARGS)

debug:
	$(PYTHON) -m pdb -m $(MODULE) $(ARGS)

test:
	$(UV) run pytest -q

lint:
	$(UV) run flake8 .
	$(UV) run mypy . $(MYPY_FLAGS)

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache \
		-o -name .pytest_cache -o -name .ruff_cache \) \
		-not -path "./.venv/*" -prune -exec rm -rf {} +
	find . -type f -name "*.py[co]" -not -path "./.venv/*" -delete

help:
	@echo "install      install the dependencies with uv"
	@echo "run          run the program (ARGS=... for options)"
	@echo "debug        run the program under pdb"
	@echo "test         run the unit tests"
	@echo "lint         flake8 + mypy with the mandatory flags"
	@echo "lint-strict  flake8 + mypy --strict"
	@echo "clean        remove caches and temporary files"
