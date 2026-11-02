.PHONY: install fetch index ask test eval lint serve clean db-up db-init

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

db-up:          ## start a local PostgreSQL (docker-compose.yml)
	docker compose up -d db

db-init:        ## create the storage tables; needs ATOQA_DATABASE_URL
	python -m atoqa.cli db-init

clean:
	rm -rf data/index data/traces evals/reports .pytest_cache .ruff_cache
