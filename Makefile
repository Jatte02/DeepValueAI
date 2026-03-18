.PHONY: run test lint fix install dataset train pipeline clean help

help:  ## Show available commands
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

run:  ## Launch the Streamlit dashboard
	cd app && streamlit run streamlit_app.py

test:  ## Run all unit tests
	python -m pytest tests/ -v

lint:  ## Check code quality with Ruff
	python -m ruff check .

fix:  ## Auto-fix lint issues and format code
	python -m ruff check --fix .
	python -m ruff format .

install:  ## Install all dependencies (including dev)
	pip install -e ".[dev]"

dataset:  ## Generate the training dataset (ETL)
	python ml_pipeline/generate_dataset.py

train:  ## Train and compare ML models
	python ml_pipeline/train_model.py

pipeline: dataset train  ## Full pipeline: generate data then train

clean:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache