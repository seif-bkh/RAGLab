"""profiles.py — the runtime half of the console/service pair.

Everything a RAGLab runtime needs to actually RUN a selected profile, with no
console and no HTTP in it: the provider/model registries (what "registered"
means: a provider the codebase can drive, its env keys, its model IDs), the
pinned-pair constants from pipeline_policy, the lab-config copy builder
(build_lab_config — a SimpleNamespace copy of config, never a mutation of the
module, with active_embedding_model() overridden so store.py labels chunks
honestly), the chat clients for every answer provider (xKiro pinned + the
experimental gateway path, NVIDIA build endpoint, Google free tier via
llm_smoke, Kira AI), profile-scoped Chroma collection naming, corpus/ingest
plumbing, and the local greeting short-circuit.

Two frontends import this module and nothing of each other:

  * app.py     — the interactive console (menus, key prompts, app_state.json)
  * service.py — the standalone HTTP microservice (FastAPI, env-configured)

Policy: registries and the pinned pair are shared, so the console and the
service can never disagree about what is selectable. The supported benchmark
pipeline stays pinned in pipeline_policy.py; non-pinned selections are lab /
service surfaces with no benchmark attribution.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import config as cfg
from nvidia_api import ANSWER_MODELS, NvidiaAPIError, NvidiaClient, safe_error
from pipeline_policy import (ANSWER_MODEL as POLICY_ANSWER_MODEL,
                             ANSWER_PROVIDER as POLICY_ANSWER_PROVIDER,
                             EMBEDDING_MODEL as POLICY_EMBEDDING_MODEL,
                             EMBEDDING_PROVIDER as POLICY_EMBEDDING_PROVIDER)
from provider_catalog import PROVIDERS as GATEWAY_CATALOG
from chat import CHAT_MODEL
import chat as chat_mod

PROJECT_DIR = Path(__file__).resolve().parent


def greeting_reply(question, *, language=None):
    """(language, reply_text) for a pure greeting, or None for a real question.

    Shared by the console (prints it) and the service (returns it as JSON):
    a greeting asserts nothing about the corpus, so neither frontend sends it
    through retrieval or the citation gate.
    """
    if not is_greeting(question):
        return None
    lang = language if language in GREETING_REPLIES else greeting_language(question)
    return lang, GREETING_REPLIES[lang]


NVIDIA_CHAT_BASE_URL = "https://integrate.api.nvidia.com/v1"

GOOGLE_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

KIRA_BASE_URL = "https://kiraai.vn/api/v1"

# The pinned pair every selection is compared against (single source of truth:
# pipeline_policy, the same constants main.py answer enforces).
SUPPORTED_EMBEDDING = {"provider": POLICY_EMBEDDING_PROVIDER, "model": POLICY_EMBEDDING_MODEL}

SUPPORTED_ANSWER = {"provider": POLICY_ANSWER_PROVIDER, "model": POLICY_ANSWER_MODEL}

EMBEDDING_PROVIDERS = {
    "nvidia": {
        "label": "NVIDIA build endpoint (stdlib HTTPS, no SDK)",
        "key_envs": ("NVIDIA_API_KEY",),
        "key_hint": "build.nvidia.com keys start with nvapi-",
        "models": [{"id": POLICY_EMBEDDING_MODEL,
                    "note": "2048 dims — the supported pipeline embedding"}],
        "sdk": None,
        "custom": True,
    },
    "gemini": {
        "label": "Google AI Studio (Gemini)",
        "key_envs": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "key_hint": "create one at https://aistudio.google.com/apikey",
        "models": [
            {"id": "gemini-embedding-2", "note": "multilingual; 3072 dims truncated to 768"},
            {"id": "gemini-embedding-001", "note": "older text-only family"},
        ],
        "sdk": ("google.genai", "google-genai"),
        "custom": True,
    },
    "jina": {
        "label": "Jina embeddings (stdlib HTTPS, no SDK)",
        "key_envs": ("JINA_API_KEY",),
        "key_hint": "jina.ai dashboard -> API keys",
        "models": [{"id": "jina-embeddings-v5-omni-small",
                    "note": "multilingual, L2-normalized by the API"}],
        "sdk": None,
        "custom": True,
    },
    "huggingface": {
        "label": "HuggingFace local — offline, no key, runs on your machine",
        "key_envs": (),
        "models": [
            {"id": "Qwen/Qwen3-Embedding-0.6B", "note": "1024 dims, ~640MB first download"},
            {"id": "BAAI/bge-m3", "note": "1024 dims, battle-tested alternative"},
        ],
        "sdk": ("sentence_transformers", "sentence-transformers"),
        "custom": True,
    },
    "openai": {
        "label": "OpenAI embeddings (needs the openai SDK)",
        "key_envs": ("OPENAI_API_KEY",),
        "models": [
            {"id": "text-embedding-3-large", "note": "3072 dims"},
            {"id": "text-embedding-3-small", "note": "1536 dims, cheaper"},
        ],
        "sdk": ("openai", "openai"),
        "custom": True,
    },
    "cohere": {
        "label": "Cohere embeddings (needs the cohere SDK)",
        "key_envs": ("COHERE_API_KEY",),
        "models": [{"id": "embed-multilingual-v3", "note": "multilingual"}],
        "sdk": ("cohere", "cohere"),
        "custom": True,
    },
    "voyage": {
        "label": "Voyage embeddings (needs the voyageai SDK)",
        "key_envs": ("VOYAGE_API_KEY",),
        "models": [
            {"id": "voyage-3-large", "note": "general-purpose, high quality"},
            {"id": "voyage-3", "note": "general-purpose"},
        ],
        "sdk": ("voyageai", "voyageai"),
        "custom": True,
    },
}

ANSWER_PROVIDERS = {
    "xkiro": {
        "label": "xKiro gateway — the supported answer path (live zero-price check)",
        "key_envs": ("XKIRO_API_KEY",),
        "key_hint": "https://docs.xkiro.com/ — dashboard -> API keys",
        "models": [{"id": POLICY_ANSWER_MODEL,
                    "note": "pinned SKU; live free-price check; the policy forbids substitutes "
                            "on the benchmark path"}],
        # The pinned SKU is the only benchmark-attributable answerer, but this is
        # the LAB console: other IDs on the same gateway may be typed and are
        # routed as clearly-labelled experimental calls (no price gate, no
        # benchmark attribution) — the pin itself stays intact for main.py.
        "custom": True,
    },
    "nvidia": {
        "label": "NVIDIA build endpoint — free chat models",
        "key_envs": ("NVIDIA_API_KEY",),
        "key_hint": "build.nvidia.com keys start with nvapi-",
        "models": ([{"id": CHAT_MODEL, "note": "the chat.py conversational profile"}]
                   + [{"id": model, "note": "registered nvidia_api.ANSWER_MODELS entry"}
                      for model in ANSWER_MODELS]),
        "custom": True,
    },
    "google": {
        "label": "Google AI Studio — cheapest free-tier Gemini (the llm_smoke path)",
        "key_envs": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "key_hint": "create one at https://aistudio.google.com/apikey",
        "models": [{"id": "auto",
                    "note": "cheapest free gemini-* model, discovered from your key"}],
        "custom": True,
    },
    "kira": {
        "label": "Kira AI — OpenAI-compatible gateway (kiraai.vn)",
        "key_envs": ("KIRA_API_KEY",),
        "key_hint": "the key from your Kira AI dashboard (https://kiraai.vn)",
        "models": [{"id": "glm-5.3-free",
                    "note": "free GLM chat model from Kira's own usage example"}],
        "custom": True,
    },
}

# Every key this repo knows about, for the key-manager screen. Descriptions
# say what the key unlocks, because a name like XKIRO_API_KEY does not.
KEY_ENV_INFO = {
    "NVIDIA_API_KEY": "NVIDIA build endpoint — nemotron embeddings + the free chat models (nvapi-…)",
    "XKIRO_API_KEY": "xKiro gateway — the supported answer path (qwen/qwen3.8-max:free)",
    "GEMINI_API_KEY": "Google AI Studio — Gemini embeddings (GOOGLE_API_KEY is also accepted)",
    "GOOGLE_API_KEY": "Google AI Studio — Gemini chat fallback (llm_smoke phase B reads this one)",
    "JINA_API_KEY": "Jina embeddings",
    "KIRA_API_KEY": "Kira AI gateway — OpenAI-compatible chat models (kiraai.vn)",
    "OPENAI_API_KEY": "OpenAI embeddings",
    "COHERE_API_KEY": "Cohere embeddings",
    "VOYAGE_API_KEY": "Voyage embeddings",
    "EXPERIENTIAL_API_KEY": "Experiential Labs judge — hard-harness grading only, not used by this console (xpl_…)",
}

def first_set_env(names):
    """(env_name, value) of the first configured key among `names`; value '' if none."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return name, value
    return (names[0] if names else ""), ""

def masked(value: str) -> str:
    """The repo's standing rule: never echo more than the first 8 characters."""
    text = str(value or "")
    return text[:8] + "…" if len(text) > 8 else ("set" if text else "")

def missing_key_slots(state: dict) -> list:
    """Slots ('embedding'/'answer') whose selected provider has no key."""
    missing = []
    for slot, registry in (("embedding", EMBEDDING_PROVIDERS), ("answer", ANSWER_PROVIDERS)):
        info = registry.get(state[slot]["provider"], {})
        envs = info.get("key_envs", ())
        if envs and not first_set_env(envs)[1]:
            missing.append(slot)
    return missing

def default_state() -> dict:
    return {
        "version": 1,
        "embedding": dict(SUPPORTED_EMBEDDING),
        "answer": dict(SUPPORTED_ANSWER),
        "chunking": {"mode": cfg.CHUNKING_MODE,
                     "size": cfg.CHUNK_SIZE_TOKENS,
                     "overlap": cfg.CHUNK_OVERLAP_TOKENS},
        "retrieval": {"top_k": cfg.ANSWER_TOP_K, "mode": "vector",
                      "lang_filter": None, "neighbor_radius": 0},
        "data_dirs": None,                  # None = chat.py's default corpus
        "custom_models": {},                # provider -> [model IDs you typed]
    }

def pipeline_marker(state: dict) -> str:
    if state["embedding"] == SUPPORTED_EMBEDDING and state["answer"] == SUPPORTED_ANSWER:
        return "supported pipeline — the same model pair main.py answer uses"
    return ("experimental lab profile — same chunker/retrieval/citation checks, "
            "but no benchmark number belongs to it")

def slot_display(entry: dict) -> str:
    """"provider/model" without the redundant vendor repeat (nvidia/nvidia/…)."""
    provider, model = entry["provider"], entry["model"]
    prefix = provider + "/"
    return f"{provider}/{model[len(prefix):]}" if model.startswith(prefix) else f"{provider}/{model}"

def model_slug(model: str) -> str:
    name = model.split("/", 1)[-1]                    # drop the vendor prefix
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "model"

def collection_name(state: dict) -> str:
    """One collection per embedding space AND chunking mode.

    Embedding spaces are provider/model-specific and chunk texts are
    segmentation-specific, so the app gives every combination its own
    collection instead of resetting a shared one: switching back and forth
    never destroys an index, and store.py's fingerprint guards stay honest.
    """
    emb = state["embedding"]
    return f"raglab_app_{emb['provider']}_{model_slug(emb['model'])}_{state['chunking']['mode']}"

def build_lab_config(state: dict) -> SimpleNamespace:
    local = SimpleNamespace(**{key: value for key, value in vars(cfg).items()
                               if not key.startswith("_") and (key.isupper() or callable(value))})
    emb = state["embedding"]
    provider, model = emb["provider"], emb["model"]

    # config.active_embedding_model() is a closure over the MODULE's globals:
    # copied as-is it would report config's model, not the selected one, and
    # store.py would tag every app chunk with the wrong embedding model. The
    # override below is what keeps the lab copy honest.
    local.active_embedding_model = lambda model=model: model

    local.EMBEDDING_PROVIDER = provider
    # RAGLAB_CACHE_DIR (set by the Docker image / compose) relocates the
    # resumable caches to a mounted volume, so a recreated container does not
    # re-embed the corpus. Default: next to the code, as before.
    cache_dir = Path(os.getenv("RAGLAB_CACHE_DIR", str(PROJECT_DIR)))
    cache = cache_dir / f"embeddings_cache_app_{provider}.json"
    local.EMBEDDING_CACHE_PATH = cache
    if provider == "nvidia":
        local.NVIDIA_EMBEDDING_MODEL = model
        local.NVIDIA_EMBEDDING_CACHE_PATH = cache
        local.NVIDIA_EMBEDDING_DIM = (2048 if model == POLICY_EMBEDDING_MODEL
                                      else int(getattr(local, "NVIDIA_EMBEDDING_DIM", 0) or 0))
    elif provider == "gemini":
        local.EMBEDDING_MODEL = model          # the generic knob the Gemini embedder reads
    elif provider == "jina":
        local.JINA_EMBEDDING_MODEL = model
        local.JINA_EMBEDDING_CACHE_PATH = cache
    elif provider == "huggingface":
        local.HF_EMBEDDING_MODEL = model
    else:                                     # openai / cohere / voyage share the generic knob
        local.EMBEDDING_MODEL = model
    local.CHROMA_COLLECTION_NAME = collection_name(state)

    chunking = state["chunking"]
    local.CHUNKING_MODE = chunking["mode"]
    local.CHUNK_SIZE_TOKENS = int(chunking["size"])
    local.CHUNK_OVERLAP_TOKENS = int(chunking["overlap"])

    retrieval = state["retrieval"]
    local.ANSWER_TOP_K = max(1, int(retrieval["top_k"]))
    local.ANSWER_NEIGHBOR_RADIUS = int(retrieval["neighbor_radius"])
    # Room for every retrieved excerpt (chat.py's rule): a token ceiling that
    # silently drops the 5th hit makes "k=5" quietly mean 4.
    local.ANSWER_CONTEXT_TOKENS = max(3000, local.ANSWER_TOP_K
                                      * (local.CHUNK_SIZE_TOKENS + local.CHUNK_OVERLAP_TOKENS))

    ans = state["answer"]
    local.ANSWER_PROVIDER = ans["provider"]
    local.ANSWER_MODEL = ans["model"]
    local.ANSWER_PROMPT_VERSION = "grounded-v1"
    local.ANSWER_CACHE_PATH = PROJECT_DIR / f"answers_cache_app_{ans['provider']}.json"

    local.QUERY_VARIANT_STRATEGY = "original"     # query translation is retired in this pipeline
    local.QUERY_TRANSLATION_ENABLED = False
    local.NVIDIA_CHAT_STREAM = False
    local.RESULTS_DIR = cfg.RESULTS_DIR / "app"
    return local

class GoogleChatClient:
    """llm_smoke's Google free-tier path behind the client interface
    AnswerGenerator expects: .chat(model, messages, max_tokens=...) -> {text}.

    Reusing llm_smoke's tested request shape (systemInstruction + JSON mime
    type + cheapest-first model ranking) instead of writing a second Google
    client means the app and the CI smoke test make literally the same calls.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = GOOGLE_API_ROOT
        self.calls = 0

    def resolve_model(self) -> str:
        from llm_smoke import google_candidates
        candidates = google_candidates(self.api_key)
        if not candidates:
            raise RuntimeError("no gemini chat model is visible to this key "
                               "(the /models listing returned nothing usable)")
        return candidates[0]

    def chat(self, model, messages, *, max_tokens=4096):
        import time
        from llm_smoke import google_call
        self.calls += 1
        started = time.monotonic()
        text, error = google_call(model, self.api_key, messages, max_tokens)
        if error:
            raise RuntimeError(error)
        return {"text": text, "requested_model": model, "served_model": model,
                "usage": {}, "seconds": round(time.monotonic() - started, 3)}

class GatewayChatClient(NvidiaClient):
    """OpenAI-compatible gateway chat over the lab's tested stdlib HTTPS
    transport (retries, pacing, key redaction, SSE) — no SDK dependency.

    Used for Kira and for EXPERIMENTAL xKiro SKUs. The pinned xKiro SKU never
    goes through this: build_answer_generator keeps its live free-price
    verification, and 'no substitution' stays true on the benchmark path.
    """

    def __init__(self, *, base_url: str, api_key: str, provider_label: str):
        super().__init__(base_url=base_url, api_key=api_key,
                         timeout=120, attempts=2, min_interval=3.0,
                         max_retry_delay=60)
        self.provider_label = provider_label

    def request(self, path, payload=None):
        try:
            return super().request(path, payload)
        except NvidiaAPIError as exc:
            raise NvidiaAPIError(str(exc).replace("NVIDIA", self.provider_label),
                                 exc.status_code, exc.retry_after) from None

def build_generator(local: SimpleNamespace):
    """AnswerGenerator wired to the selected answer provider, or an error saying why not."""
    from answer import AnswerGenerator, build_answer_generator
    provider, model = local.ANSWER_PROVIDER, local.ANSWER_MODEL
    if provider == "xkiro" and model == SUPPORTED_ANSWER["model"]:
        # The supported path: live zero-price verification, no substitution.
        return build_answer_generator(local, call_budget=1000)
    if provider == "xkiro":
        # A custom SKU on the same gateway: an experimental lab call. The price
        # gate is not weakened for the pinned SKU — it is simply not claimed for
        # this one, and no benchmark number is attributed to it.
        key = os.environ.get("XKIRO_API_KEY", "").strip()
        if not key:
            raise ValueError("XKIRO_API_KEY is not set — add it from the API-keys menu, "
                             "then try again.")
        print(f"[app] xKiro/{model} is an EXPERIMENTAL SKU: the live free-price check and "
              f"benchmark attribution belong to the pinned {SUPPORTED_ANSWER['model']} only.")
        return AnswerGenerator(local, client=GatewayChatClient(
            base_url=GATEWAY_CATALOG["xkiro"]["base_url"], api_key=key,
            provider_label="XKIRO"), approved_models=(model,))
    if provider == "kira":
        key = os.environ.get("KIRA_API_KEY", "").strip()
        if not key:
            raise ValueError("KIRA_API_KEY is not set — add it from the API-keys menu, "
                             "then try again.")
        return AnswerGenerator(local, client=GatewayChatClient(
            base_url=KIRA_BASE_URL, api_key=key, provider_label="KIRA"),
            approved_models=(model,))
    if provider == "nvidia":
        from nvidia_api import NvidiaClient
        client = NvidiaClient(base_url=NVIDIA_CHAT_BASE_URL,
                              timeout=getattr(local, "NVIDIA_API_TIMEOUT", 120),
                              attempts=getattr(local, "NVIDIA_API_ATTEMPTS", 3),
                              min_interval=getattr(local, "NVIDIA_MIN_INTERVAL", 1.6),
                              max_retry_delay=getattr(local, "NVIDIA_MAX_RETRY_DELAY", 30))
        if not client.api_key:
            raise ValueError("NVIDIA_API_KEY is not set — add it from the API-keys menu "
                             "or raglab/.env, then try again.")
        return AnswerGenerator(local, client=client, approved_models=(model,))
    if provider == "google":
        key = first_set_env(("GOOGLE_API_KEY", "GEMINI_API_KEY"))[1]
        if not key:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set — "
                             "add it from the API-keys menu, then try again.")
        client = GoogleChatClient(key)
        if model == "auto":
            model = client.resolve_model()       # dated model IDs self-resolve per session
            local.ANSWER_MODEL = model
            print(f"[app] google: cheapest free model visible to this key is {model}")
        return AnswerGenerator(local, client=client, approved_models=(model,))
    raise ValueError(f"unknown answer provider {provider!r}")

def data_dirs(state: dict):
    """The corpus directories — chat.py's defaults (../docs + raglab/data) unless overridden."""
    return chat_mod.data_dirs(state.get("data_dirs"))

def ingest(local, embedder, state, *, reset: bool):
    """Chunk the corpus, embed it, store it in the app's own collection."""
    from chunker import chunk_all
    from loader import load_all
    from store import get_collection, store_chunks
    dirs = data_dirs(state)
    docs = load_all(dirs)
    if not docs:
        raise RuntimeError("no documents found in: " + ", ".join(str(d) for d in dirs))
    chunks = chunk_all(docs, local)
    if not chunks:
        raise RuntimeError("chunking produced nothing; nothing to ingest")
    print(f"[app] embedding {len(chunks)} chunk(s) from {len(docs)} document(s) with "
          f"{embedder.provider_name}/{embedder.model} (batch {embedder.batch_size})...")
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    collection = get_collection(local, reset=reset)
    stored = store_chunks(collection, list(zip(chunks, embeddings)), local)
    print(f"[app] index ready: {stored} record(s) in {local.CHROMA_COLLECTION_NAME} | "
          f"api calls={embedder.api_calls} cache hits={embedder.cache_hits}")
    return collection

def collection_count(state: dict):
    """Chunk count for the current profile's collection, or None if unknowable."""
    try:
        from store import _client
        client = _client(build_lab_config(state))
        try:
            return client.get_collection(collection_name(state)).count()
        except Exception:                                    # noqa: BLE001 — "missing" is fine
            return 0
    except Exception:                                        # noqa: BLE001 — e.g. chromadb absent
        return None

def stored_index_info(state: dict):
    """(count, stored_chunk_fp) of the profile's collection; fp None when absent.

    The chunk fingerprint embeds the tokenizer identity ('…:tok<name>:…'), so
    comparing it with this machine's tokenizer_identity() answers the classic
    cross-device puzzle: two machines that ingested 'the same' corpus with
    different tokenizer availability produce different chunk boundaries, and a
    sentence whole inside one chunk on machine A is split across two on B.
    """
    try:
        from store import _client
        client = _client(build_lab_config(state))
        collection = client.get_collection(collection_name(state))
        count = collection.count()
        if not count:
            return 0, None
        metas = collection.get(include=["metadatas"], limit=1).get("metadatas") or [{}]
        return count, (metas[0] or {}).get("chunk_fp")
    except Exception:                                        # noqa: BLE001 — absent/chromadb missing
        return 0, None

def consistent_model(provider: str, entry: dict, state: dict, previous: str) -> str:
    """The model to preselect after a provider switch: the previous model only
    if the NEW provider offers it, else its first registered model.

    Never carry another provider's model across a switch — a gemini slot that
    keeps 'nvidia/nemotron-3-embed-1b' (because model selection was backed out
    of) is a profile that cannot be built or honestly labelled.
    """
    offered = ({model["id"] for model in entry["models"]}
               | set(state.get("custom_models", {}).get(provider, [])))
    return previous if previous in offered else entry["models"][0]["id"]

# ---------------------------------------------------------------------------
# Greetings / smalltalk — answered locally, zero calls
# ---------------------------------------------------------------------------
# A pure greeting is not a document question: the grounded contract would only
# force the model to abstain (correctly) after paying for retrieval + a
# completion. Like answer.local_private_refusal, this is decided locally before
# any provider call — and because it asserts NOTHING about the corpus, it never
# enters the citation gate. Anchored to the WHOLE input, so "bonjour, what is
# murabaha?" still goes through the full grounded path.
GREETING_RE = re.compile(
    r"^(?:bonjour|bonsoir|salut|coucou|merci(?:\s+beaucoup)?|de\s+rien"
    r"|hello|hi|hey|good\s+(?:morning|afternoon|evening)|thank\s+you"
    r"|thanks(?:\s+(?:a\s+lot|very\s+much))?"
    r"|السلام\s+عليكم|سلام\s+عليكم|مرحبا|مرحبتين|أهلا|اهلا|صباح\s+الخير"
    r"|مساء\s+الخير|شكرا|شكراً|شكرا\s+جزيلا)[\s!.,;?؟…]*$",
    re.IGNORECASE)

GREETING_REPLIES = {
    "fr": "Bonjour ! Je suis l'assistant documentaire du laboratoire : posez une question sur le "
          "corpus (ex. « Qu'est-ce que la Murabaha ? », « ما هي المرابحة؟ »). Une salutation "
          "n'interroge aucun document, donc aucun appel au modèle n'a été fait.",
    "en": "Hello! I'm the lab's document-grounded assistant: ask something about the corpus "
          "(e.g. 'What is Murabaha?', « Qu'est-ce que la Murabaha ? », « ما هي المرابحة؟ »). "
          "A greeting queries no document, so no model call was made.",
    "ar": "مرحبا! انا مساعد هذا المختبر للإجابة من المستندات: اطرح سؤالا عن المدونة (مثال: "
          "«ما هي المرابحة؟» أو What is Murabaha). التحية لا تستند الى اي مستند، لذلك لم يتم "
          "اي استدعاء للنموذج.",
}

def is_greeting(text: str) -> bool:
    return bool(GREETING_RE.match((text or "").strip()))

def greeting_language(text: str) -> str:
    """fr/ar/en for a pure greeting. The generic detector returns 'en' for a
    single French word (no stopwords to bite on), so French/Arabic tokens are
    recognized explicitly — the reply should not answer 'bonjour' in English."""
    lowered = (text or "").strip().lower()
    if any(word in lowered for word in ("bonjour", "bonsoir", "salut", "coucou",
                                        "merci", "de rien")):
        return "fr"
    if re.search(r"[\u0600-\u06FF]", lowered):
        return "ar"
    return "en"
