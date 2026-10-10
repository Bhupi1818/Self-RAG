.PHONY: install data ingest api ui testset eval test lint

install:
	pip install -r requirements.txt -r requirements-dev.txt

data:
	python scripts/download_data.py --arxiv-curated --wikipedia

ingest:
	python -m src.ingest --reset

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py

testset:
	python eval/generate_testset.py --n 40

eval:
	python eval/run_eval.py

test:
	pytest -q

lint:
	ruff check .
