"""Uncached live probes for the four EXACT requested NIM model IDs.

Run in Actions when the key is only stored as a repository secret. Never
export secrets from Actions; only final sample output and diagnostics leave it.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from nvidia_api import (EMBED_MODEL, RIVA_MODEL, TRANSLATION_MODELS,
                        NvidiaClient, safe_error)


def annotation(value):
    # Also accessible through the Checks API if blob/log downloads are blocked.
    text = json.dumps(value, ensure_ascii=False).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
    print("::notice title=NVIDIA exact-model result::" + text, flush=True)


def probe(output):
    client = NvidiaClient(attempts=2)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "models": {}}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        ids = client.models()
        report["catalog_count"] = len(ids)
        report["catalog_requested_models"] = {m: m in ids for m in (EMBED_MODEL, *TRANSLATION_MODELS)}
    except Exception as exc:
        report["catalog_error"] = safe_error(exc)
    save()
    for model in (EMBED_MODEL, *TRANSLATION_MODELS):
        print(f"[probe] testing exact ID {model}", flush=True)
        try:
            if model == EMBED_MODEL:
                data = client.request("embeddings", {
                    "model": model, "input": ["savings account", "compte épargne", "حساب التوفير"],
                    "input_type": "query", "encoding_format": "float", "truncate": "NONE"})
                rows = sorted(data["data"], key=lambda item: item["index"])
                if len(rows) != 3 or any(len(r["embedding"]) != 2048 for r in rows):
                    raise ValueError("Expected three native 2048-dimensional embeddings")
                from math import sqrt, isfinite
                vectors = [r["embedding"] for r in rows]
                if any(not all(isfinite(v) for v in row) or not any(row) for row in vectors):
                    raise ValueError("Invalid embedding vector")
                def cosine(a, b):
                    return sum(x*y for x,y in zip(a,b)) / sqrt(sum(x*x for x in a)*sum(x*x for x in b))
                result = {"dimensions": 2048, "served_model": data.get("model"),
                          "similarities": {"en-fr": cosine(vectors[0], vectors[1]),
                                           "en-ar": cosine(vectors[0], vectors[2]),
                                           "fr-ar": cosine(vectors[1], vectors[2])},
                          "usage": data.get("usage")}
            else:
                source = "According to BCT circular 2019-08, what are investment deposits?"
                messages = [{"role": "system", "content": "en-ar" if model == RIVA_MODEL else
                             "Translate the user's text into Arabic. Preserve references and numbers. Output only the translation; do not answer the question."},
                            {"role": "user", "content": source}]
                result = client.chat(model, messages, max_tokens=4096 if model != RIVA_MODEL else 512)
            report["models"][model] = {"status": "ok", **result}
            print(f"[probe] OK {model}: {json.dumps(result, ensure_ascii=False)}", flush=True)
            annotation({"model": model, "status": "ok", **result})
        except Exception as exc:
            report["models"][model] = {"status": "failed", "error": safe_error(exc),
                                       "http_status": getattr(exc, "status_code", None)}
            print(f"[probe] FAILED {model}: {safe_error(exc)}", flush=True)
            annotation({"model": model, **report["models"][model]})
        save()
    report["status"] = "completed" if all(r["status"] == "ok" for r in report["models"].values()) else "incomplete"
    report["http_requests"] = client.calls
    save()
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report = json.loads((Path(__file__).parent / "results/nvidia/probe.json").read_text())
        for model, result in report["models"].items():
            annotation({"model": model, **result})
    else:
        sys.exit(probe(Path(__file__).parent / "results/nvidia/probe.json"))
