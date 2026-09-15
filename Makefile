.PHONY: install fetch index ask test eval lint serve clean

install:
	pip install -e ".[dev]"

fetch:          ## download ATO public pages into data/raw (not committed)
	python -m atoqa.cli fetch

index:          ## build the hybrid retrieval index from data/raw
	python -m atoqa.cli index

ask:            ## ask a question, e.g. make ask Q="what is the GST threshold?"
	python -m atoqa.cli ask "$(Q)"

test:
	pytest

lint:
	ruff check src evals tests

eval:           ## run the evaluation harness against regression thresholds
	python evals/run_eval.py

eval-fixtures:  ## run evals against the synthetic fixture corpus (no network)
	python evals/run_eval.py --corpus tests/fixtures/corpus

serve:
	uvicorn atoqa.api:app --reload --port 8000

clean:
	rm -rf data/index data/traces evals/reports .pytest_cache .ruff_cache
