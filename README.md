# RAGLab

A transparent Python laboratory for multilingual banking RAG (Arabic, French,
English): document extraction → chunking → embeddings → local Chroma retrieval
→ optional **cited, document-grounded answers**. No UI, cloud vector database,
or orchestration framework.

The current NVIDIA experiment uses these **exact** models:

- Embeddings: `nvidia/nemotron-3-embed-1b` (native 2048-dimensional hosted API).
- Translation comparison: `moonshotai/kimi-k3`,
  `deepseek-ai/deepseek-v4-pro-0813`, `nvidia/riva-translate-4b-instruct-v2`.
- Optional answers: the same Kimi or DeepSeek model, with evidence validation
  and refusal when the documents do not support an answer.

```bash
cd raglab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-benchmark.txt
cp .env.example .env                 # configure NVIDIA_API_KEY locally
python main.py inspect --data-dir ../docs
python main.py ingest --reset --data-dir ../docs
python main.py query "What is Murabaha?" --query-lang en
python main.py answer "What is Murabaha?" --query-lang en
python main.py benchmark --stage all # explicit live API use; saves results/nvidia/
```

See [the guide](raglab/README.md) and
[benchmark methodology](raglab/benchmarks/README.md). The previous session's
measurements remain in [CI_REPORT.md](raglab/CI_REPORT.md); those are historical,
not NVIDIA results. `chat_history.html` is the supplied previous-session archive.

**Not a production-certified banking service.** Retrieval scores and citation
membership do not prove factual/legal correctness. The sample bank in `data/`
is fictional; `docs/` is a separate real-document evaluation corpus. Do not send
personal or confidential banking data to a trial endpoint without approval.
