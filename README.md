# RAGLab

This repository contains `raglab/` — a small local Python laboratory for
testing a multilingual (Arabic, French, English) RAG retrieval pipeline:
document cleaning, chunking, embedding via the Google Gemini API (a free
Google AI Studio key works), storage in local ChromaDB, and retrieval
evaluation. No LLM generation, no UI.

Start here:

```bash
cd raglab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your embedding API key
python main.py inspect      # no API calls
python main.py ingest --reset
python main.py query "..."  # / evaluate
```

See [`raglab/README.md`](raglab/README.md) for full documentation.

> ⚠️ The sample documents in `raglab/data/` describe a **fictional** bank and
> products. All fees, rates, thresholds and contact details are invented.
