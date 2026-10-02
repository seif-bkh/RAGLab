#!/usr/bin/env python3
"""Deterministic DEMO service — Experiment 7 (owner directive 2026-10-01
«أنجزها كلها»).

A fully working RAGLab service with ZERO API keys, for the owner to try the
1.3.0 behavior interactively:

- embeddings: an injected deterministic stub (same 8-dim one-hot the offline
  tests use) — retrieval runs in rrf mode so BM25 carries the ranking;
- the answer 'model' is a deterministic quote-bot: it answers with a verbatim
  slice of the top source, THROUGH THE REAL CITATION GATE (quote membership,
  number sourcing, allowed/in-force checks all execute for real);
- the REAL docs/ corpus (restructure arm, 339 chunks);
- Phase-5/6 gates ON so the full behavior is visible: sufficiency fields,
  the commitment (refusal before the model + partial downgrade), the audit
  trail (/audit), /numbers, unit ids on citations.

Honest limits (by design): the answers are verbatim quotes, not generated
prose — the deployed live arm (real embeddings + a real chat model) is what
CI measures. This demo exists to let the owner FEEL the deterministic layers.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DOCS = HERE.parent / "docs"

# --- 1) deterministic embedding stub (injected before the service builds it)
st_mod = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.prompts = {"query": ""}

    @staticmethod
    def get_embedding_dimension():
        return 8

    def encode(self, texts, batch_size=None, normalize_embeddings=None,
               show_progress_bar=None, convert_to_numpy=None, **kwargs):
        rows = []
        for t in texts:
            row = [0.0] * 8
            row[hash(t) % 8] = 1.0
            rows.append(row)
        return rows


st_mod.SentenceTransformer = _StubSentenceTransformer
sys.modules["sentence_transformers"] = st_mod


# --- 2) the deterministic answer 'model' (through the REAL gate)
class DemoQuoteClient:
    """Answers with verbatim source slices — the citation gate validates
    them for real (membership + numbers + allowed + in-force)."""
    base_url = "https://demo.local/none"
    api_key = "none"

    @staticmethod
    def _slice(text, n=170):
        # start at the first non-header line for readability, but the quote
        # stays a VERBATIM substring of the source (gate membership holds)
        offset = 0
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith((">", "#", "**")):
                break
            offset += len(line) + 1
        return " ".join(text[offset:].split())[:n]

    def chat(self, model, messages, *, max_tokens=4096):
        payload = json.loads(messages[1]["content"])
        sources = payload.get("sources") or []
        claims = []
        for s in sources[:2]:
            quote = self._slice(s["text"])
            if len(quote) >= 12:
                claims.append({"text": quote,
                               "evidence": [{"source_id": s["source_id"],
                                             "quote": quote}]})
        out = {"answerable": bool(claims), "claims": claims}
        return {"text": json.dumps(out, ensure_ascii=False),
                "served_model": "demo-deterministic-quotebot", "usage": {},
                "seconds": 0.0}


def build_corpus_index(local):
    """Pre-build the collection so the service starts ready to answer."""
    import chunker
    import loader
    from embedder import build_embedder
    from store import get_collection, store_chunks

    docs = loader.load_all([DOCS])
    chunks = chunker.chunk_all(docs, local)
    embedder = build_embedder(local)
    collection = get_collection(local, reset=True)
    store_chunks(collection, list(zip(chunks, embedder.embed_texts(
        [c.text for c in chunks]))), local)
    return collection.count()


def main() -> int:
    import uvicorn
    import profiles
    import service
    from answer import AnswerGenerator
    from profiles import build_lab_config

    state = profiles.default_state()
    state["embedding"] = {"provider": "huggingface",
                          "model": "Qwen/Qwen3-Embedding-0.6B"}
    state["answer"] = {"provider": "nvidia", "model": "demo-deterministic"}
    state["chunking"] = {"mode": "restructure", "size": 220, "overlap": 40}
    state["retrieval"] = {"mode": "rrf", "lang_filter": None,
                          "top_k": 20, "neighbor_radius": 0}
    state["data_dirs"] = [str(DOCS)]

    demo_dir = HERE / "results" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "CHROMA_DIR": demo_dir / "chroma",
        "EMBEDDING_CACHE_PATH": demo_dir / "emb.json",
        "NVIDIA_EMBEDDING_CACHE_PATH": demo_dir / "emb.json",
        "ANSWER_CACHE_PATH": demo_dir / "answers.json",
        "RESULTS_DIR": demo_dir,
        "SUFFICIENCY_FIELDS_ENABLED": True,     # show the evidence state
        "ANSWER_SUFFICIENCY_COMMITMENT": True,  # refuse/partial for real
    }
    local = build_lab_config(state)
    for key, value in overrides.items():
        setattr(local, key, value)

    print("[demo] building the deterministic index over docs/ ...")
    n = build_corpus_index(local)
    print(f"[demo] index ready: {n} chunks")

    generator = AnswerGenerator(local, client=DemoQuoteClient(),
                                approved_models=("demo-deterministic",))
    app = service.create_app(state, generator=generator,
                             allow_profile_switch=False,
                             config_overrides=overrides)

    # a minimal owner-facing page (relative URLs only — works behind the
    # live preview proxy). The demo page REPLACES the service's plain "/"
    # listing so the owner lands on it directly.
    app.router.routes = [r for r in app.router.routes
                         if not (getattr(r, "path", None) == "/"
                                 and "GET" in getattr(r, "methods", set()))]

    @app.get("/", include_in_schema=False)
    def _index():
        from fastapi.responses import HTMLResponse
        return HTMLResponse("""<!doctype html><html lang="ar" dir="rtl"><head>
<meta charset="utf-8"><title>RAGLab — خدمة تجريبية حتمية</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;background:#fafaf7;color:#1a1a1a}
h1{font-size:1.3rem}small{color:#666}input[type=text]{width:100%;padding:10px;font-size:1.05rem;border:1px solid #bbb;border-radius:8px}
button{margin-top:8px;padding:10px 22px;font-size:1rem;border:0;border-radius:8px;background:#0b5d3b;color:#fff;cursor:pointer}
.out{margin-top:16px;border:1px solid #ddd;border-radius:10px;padding:14px;white-space:pre-wrap;line-height:1.9}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.85rem;margin-inline-end:6px}
.ok{background:#dcf5e5}.no{background:#fbe3e3}.wait{background:#eee}
.q{margin:6px 0;padding:6px 10px;border-radius:8px;background:#fff;border:1px solid #eee;cursor:pointer;display:inline-block}
blockquote{border-inline-start:3px solid #0b5d3b;margin:6px 0;padding-inline-start:10px;color:#333;font-size:.95rem}
code{background:#f0f0ec;padding:1px 6px;border-radius:6px}</style></head><body>
<h1>خدمة RAGLab التجريبية <small>(v1.3.0 — حتمية بالكامل، بلا مفاتيح)</small></h1>
<p><small>نموذج العرض «روبوت اقتباس»: يجيب <b>باقتباس حرفي من المصدر</b> عبر بوابة الاستشهاد الحقيقية (عضوية الاقتباس + توثيق الأرقام + مسموح/نافذ). حقول الكفاية والالتزام مفعّلة: أسئلة خارج المدونة تُرفض قبل أي نموذج. الذراع المنشورة الحقيقية (تضمين حي + نموذج حي) هي التي يقيسها CI.</small></p>
<input id="q" type="text" placeholder="اكتب سؤالك… مثال: ما هي عملية المرابحة على معنى القانون عدد 48 لسنة 2016؟">
<button onclick="ask()">اسأل</button>
<div id="samples"></div>
<div id="out" class="out" hidden></div>
<script>
const SAMPLES=["ما هي عملية المرابحة على معنى القانون عدد 48 لسنة 2016؟","ما التعريف القانوني للبنك في تونس؟","ما شروط فتح حساب مصرفي إسلامي؟","السلام عليكم","كيف احجز تذكرة طائرة من تونس الى دبي؟"];
const samplesEl=document.getElementById('samples');
SAMPLES.forEach(s=>{const b=document.createElement('span');b.className='q';b.textContent=s;b.onclick=()=>{document.getElementById('q').value=s;ask()};samplesEl.appendChild(b);});
async function ask(){
 const q=document.getElementById('q').value.trim();if(!q)return;
 const out=document.getElementById('out');out.hidden=false;out.textContent='…جارٍ';
 try{
  const r=await fetch('answer',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,k:6})});
  const b=await r.json();
  let h='';
  const cls=b.status==='answered'?'ok':(b.status==='refused'?'no':'wait');
  h+=`<span class="badge ${cls}">${b.status}${b.reason?'/'+b.reason:''}</span>`;
  if(b.evidence_status)h+=`<span class="badge ${b.evidence_status==='كافٍ'?'ok':'no'}">كفاية الدليل: ${b.evidence_status}</span>`;
  if(b.model)h+=`<small> — ${b.model}</small>`;
  (b.claims||[]).forEach(c=>{h+=`<div>◆ ${c.text}</div>`;(c.evidence||[]).forEach(e=>{h+=`<blockquote>${e.quote}<br><small>— ${e.source_id} ${e.unit_id||''} ${e.in_force===false?'(غير نافذ)':''}</small></blockquote>`;});});
  if(b.refusal_reason)h+=`<div>⛔ ${b.refusal_reason}</div>`;
  if(b.referral)h+=`<div>↪ الحالة مناسبة للإحالة إلى مختص.</div>`;
  if(b.greeting_reply)h+=`<div>${b.greeting_reply}</div>`;
  h+='<hr><small>المصادر: '+(b.sources||[]).map(s=>s.source_id).join('، ')+'</small>';
  out.innerHTML=h;
 }catch(e){out.textContent='تعذر الاتصال: '+e;}
}
</script></body></html>""")

    print("[demo] serving on 0.0.0.0:8000 — open / in a browser, or POST "
          "/answer, GET /audit, GET /numbers, GET /docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
