# RAGLab

A transparent Arabic/French/English document-grounded QA lab. The supported
pipeline now uses **only two models**:

- **Embeddings:** `nvidia/nemotron-3-embed-1b`, native 2048 dimensions.
- **Answers:** `qwen/qwen3.8-max:free` through xKiro, with live free-price checks.

Retrieval uses the original query, local ChromaDB/cosine, and no separate chat
translation model. There is no model/provider fallback. Retired provider choices
and stale translation-enabled configuration fail before model calls.

```bash
cd raglab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # configure only NVIDIA_API_KEY and XKIRO_API_KEY
python main.py inspect --data-dir ../docs
python main.py ingest --reset --data-dir ../docs
python main.py query "What is Murabaha?" --query-lang en
python main.py answer "What is Murabaha?" --query-lang en
```

**Not production-ready for a banking service.** The measured Qwen profile passed
13/14 development rubric checks, 18/18 held-out answer checks and three synthetic
source-injection fixtures. That small, correlated test set is not a security,
legal-correctness or availability guarantee. Suitable for a supervised pilot with
approved nonconfidential documents, not unsupervised customer banking advice.

See [the CLI guide](raglab/README.md), [readiness assessment](raglab/READINESS.md),
and [measured model comparison](raglab/FREE_MODELS_REPORT.md). Historical reports
and fixtures remain as evidence; their earlier providers are not active choices.
The sample bank in `data/` is fictional; `docs/` contains the four real evaluation
documents. No UI, account access, cloud vector database, or orchestration framework.
