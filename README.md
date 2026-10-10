# Self-RAG: A Self-Correcting RAG Pipeline

A local-first Retrieval-Augmented Generation (RAG) project that
retrieves evidence, filters passages for relevance, generates an answer,
and checks whether the answer is supported by the retrieved context. If
evidence is insufficient, the pipeline can rewrite the query and retry
instead of presenting an unsupported answer as grounded.

**Repository:**
[Bhupi1818/Self-RAG](https://github.com/Bhupi1818/Self-RAG)

> This project implements a practical self-reflective RAG workflow. It
> is inspired by the broader idea of self-critique in RAG systems; it is
> not a reproduction of the original trained SELF-RAG model.

## Why this project?

A basic RAG system retrieves passages and passes them to a language
model. It can fail when passages are irrelevant, the question is phrased
differently from the source, or the answer contains claims the context
does not support.

This project adds explicit checks:

-   **Retrieve** candidate passages from the indexed knowledge base.
-   **Grade** passage relevance to the question.
-   **Generate** an answer using selected context.
-   **Verify** whether the answer is supported and answers the question.
-   **Rewrite and retry** when retrieval or verification is
    insufficient.
-   **Refuse when necessary** rather than claiming unsupported
    information is certain.

## Pipeline overview

``` mermaid
flowchart TD
    A[User question] --> B[Route question]
    B --> C[Retrieve candidate passages]
    C --> D[Grade passage relevance]
    D --> E{Enough relevant evidence?}
    E -- No --> F[Rewrite search query]
    F --> C
    E -- Yes --> G[Generate answer from context]
    G --> H[Verify support and answer quality]
    H --> I{Supported and answers question?}
    I -- Yes --> J[Return grounded answer]
    I -- No, retry available --> F
    I -- No, retries exhausted --> K[Return refusal / insufficient evidence]
```

Retries are bounded and configurable. Verification can reduce risk, but
it does not guarantee that every answer is correct.

## Features

-   Local retrieval with a Chroma vector database.
-   Sentence-transformer embeddings
    (`sentence-transformers/all-MiniLM-L6-v2` by default).
-   Configurable LLM provider, including Ollama and Anthropic support in
    the project code.
-   Passage relevance grading before generation.
-   Answer-support verification after generation.
-   Query rewriting and bounded retries.
-   FastAPI endpoint and Streamlit interface.
-   Automated tests and evaluation scripts.

## Tech stack

  Component          Technology
  ------------------ ----------------------------------
  Language           Python
  API                FastAPI, Uvicorn
  UI                 Streamlit
  Vector store       Chroma
  Embeddings         Sentence Transformers
  Local LLM option   Ollama (for example, `llama3.2`)
  Tests              pytest

## Quick start

### Prerequisites

-   Python 3.11
-   Git
-   An LLM provider configured for the project
-   For local inference, [Ollama](https://ollama.com/) and a downloaded
    model

### 1. Clone the repository

``` bash
git clone https://github.com/Bhupi1818/Self-RAG.git
cd Self-RAG
```

### 2. Create a virtual environment and install dependencies

On macOS/Linux:

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure environment variables

``` bash
cp .env.example .env
```

Edit `.env` to match your setup. For a local Ollama configuration, the
relevant values are typically:

``` dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=chroma_db
COLLECTION_NAME=self_rag_docs
RAW_DIR=data/raw
CHUNK_SIZE=250
CHUNK_OVERLAP=40
TOP_K=8
MAX_RETRIES=1
MIN_RELEVANT=1
SUPPORT_THRESHOLD=0.8
```

Keep any additional variables in `.env.example`. That file is the source
of truth for supported configuration. If using Ollama, start it and
download the model:

``` bash
ollama pull llama3.2
ollama list
```

For a hosted provider, configure its credentials locally in `.env`.
Never commit API keys or passwords.

### 4. Prepare the knowledge base

Place the documents you want to query in the configured raw-data
directory (`data/raw/`), then run:

``` bash
python -m src.ingest --reset
```

Ingestion chunks documents, creates embeddings, and builds the local
Chroma index. The index is stored under `chroma_db/` by default and is
intentionally excluded from Git.

See [`docs/DATA.md`](docs/DATA.md) and [`docs/SETUP.md`](docs/SETUP.md)
for project-specific details.

### 5. Run the API

``` bash
make api
```

The development server is configured for `http://127.0.0.1:8000`.
Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

### 6. Run the Streamlit interface

In a second terminal, activate the virtual environment and run:

``` bash
streamlit run app/streamlit_app.py
```

## Example API request

With the API running:

``` bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why are dot products scaled by the square root of d_k in the Transformer?",
    "mode": "self_rag"
  }'
```

Inspect the live response schema in Swagger UI (`/docs`) or
[`docs/API.md`](docs/API.md), since fields can evolve. Answers depend on
the material indexed in your knowledge base. A refusal or
insufficient-evidence response is expected when the documents do not
support an answer.

## Tests

Run:

``` bash
python -m pytest -q
```

The current development baseline has **14 passing tests**. Re-run the
suite after changing pipeline behavior, prompts, configuration, or
schemas.

## Evaluation

Evaluation utilities and assets are included:

-   [`eval/testset.json`](eval/testset.json) --- test questions / cases.
-   [`eval/benchmark.json`](eval/benchmark.json) --- benchmark input.
-   [`eval/benchmark_results.json`](eval/benchmark_results.json) ---
    stored results.
-   [`eval/run_eval.py`](eval/run_eval.py) --- evaluation runner.
-   [`eval/run_benchmark.py`](eval/run_benchmark.py) --- benchmark
    runner.
-   [`eval/generate_testset.py`](eval/generate_testset.py) --- test-set
    generation utility.
-   [`docs/EVALUATION.md`](docs/EVALUATION.md) --- methodology and
    usage.

Stored results apply to the run and configuration that produced them;
they are not a performance guarantee for other models, corpora, or
hardware.

## Repository layout

``` text
.
├── api/                 # FastAPI application
├── app/                 # Streamlit interface
├── data/                # Raw and processed data placeholders
├── docs/                # Setup, API, architecture, and evaluation docs
├── eval/                # Evaluation scripts and benchmark assets
├── scripts/             # Data preparation utilities
├── src/
│   ├── config.py        # Environment configuration
│   ├── ingest.py        # Document ingestion and indexing
│   ├── retriever.py     # Retrieval
│   ├── grader.py        # Passage relevance grading
│   ├── generator.py     # Answer generation
│   ├── verifier.py      # Answer-support verification
│   ├── rewriter.py      # Query rewriting
│   ├── router.py        # Question routing
│   └── pipeline.py      # Pipeline orchestration
├── tests/               # Automated tests
├── .env.example         # Configuration template
├── Makefile             # Common development commands
└── requirements.txt     # Runtime dependencies
```

## Configuration reference

  Setting                         Purpose
  ------------------------------- --------------------------------------
  `LLM_PROVIDER`                  Language-model provider
  `LLM_MODEL`                     Model name
  `OLLAMA_HOST`                   Ollama server URL
  `EMBEDDING_MODEL`               Embedding model
  `CHROMA_DIR`                    Local vector database directory
  `COLLECTION_NAME`               Chroma collection
  `RAW_DIR`                       Raw document directory
  `CHUNK_SIZE`, `CHUNK_OVERLAP`   Document chunking
  `TOP_K`                         Number of candidate passages
  `MAX_RETRIES`                   Maximum retry count
  `MIN_RELEVANT`                  Minimum relevant passages to proceed
  `SUPPORT_THRESHOLD`             Answer-support threshold

See [`src/config.py`](src/config.py) and `.env.example` for the current
supported settings.

## Security

-   Never commit `.env` or `.envx`.
-   Keep real credentials out of `.env.example`.
-   Do not commit private source documents or API keys.
-   Keep local indexes and virtual environments excluded from Git.
-   Review `git status` before committing.

## Known limitations

-   Answer quality depends on the coverage and quality of the indexed
    documents.
-   Latency and answer quality vary by model and hardware.
-   Relevance grading and verification are model judgments and can be
    wrong.
-   Passing verification is not a formal proof of factual correctness.
-   Evaluation results depend on the corpus and configuration.

## Documentation

-   [Requirements and scope](docs/PRD.md)
-   [Architecture](docs/ARCHITECTURE.md)
-   [Pipeline](docs/PIPELINE.md)
-   [Setup](docs/SETUP.md)
-   [API](docs/API.md)
-   [Data preparation](docs/DATA.md)
-   [Evaluation](docs/EVALUATION.md)
-   [Design decisions](docs/DECISIONS.md)

## Contributing

Issues and contributions are welcome. Run the test suite before
submitting changes and do not include credentials, local indexes, or
private data. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

See [`LICENSE`](LICENSE) for the license that applies to this
repository.
