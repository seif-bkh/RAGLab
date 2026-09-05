"""translate.py — query translation for cross-lingual retrieval (EXPERIMENT).

The v2 diagnostics showed the remaining failures are LANGUAGE ROUTING: an
Arabic question clusters on Arabic chunks while the fact also exists in the
French document. Chunking cannot fix that, so this module translates each
query into every corpus language and retrieval runs per variant (see
store.best_variant_merge, "best-score fusion"): each chunk keeps the score
from the variant in which it matches best, so a French chunk can win even for
an Arabic question.

Implementation notes:
- Backend switchable via QUERY_TRANSLATION_PROVIDER:
    "gemini" (default): SAME Google AI Studio key (GEMINI_API_KEY /
      GOOGLE_API_KEY) and google-genai SDK as the embedder — no extra dep.
    "nvidia": free NVIDIA NIM LLM endpoints (build.nvidia.com) via raw HTTPS
      (no SDK): moonshotai/kimi-k3 or deepseek-ai/deepseek-v4-pro,
      key from NVIDIA_API_KEY. Translation only — answer.py stays a stub.
- Translation model configurable: QUERY_TRANSLATION_MODEL (gemini) /
  NVIDIA_TRANSLATION_MODEL (nvidia), default gemini-3.5-flash-lite /
  moonshotai/kimi-k3. Whole batches are translated in a single
  numbered-lines request (2 calls per question set per target at most).
- Results cached in translations_cache.json (sha256(model + target + text)),
  so CI warm-ups and local re-runs cost nothing. The cache is keyed by
  model, so Gemini and Kimi/DeepSeek entries coexist.
- If translation is unavailable or fails, callers degrade to the original
  query only and record it — translation is an enhancement, never a blocker.
- RETRIEVAL-SIDE ONLY: nothing here generates answers; answer.py stays a stub.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from embedder import require_provider_sdk

LANG_NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}

_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*?)\s*$")


def detect_language(text: str) -> str:
    """Cheap script/stopword guess: Arabic script -> 'ar', French markers ->
    'fr', otherwise 'en'. The question set (questions.json) declares its own
    language, so this is only used by the `query` CLI without --query-lang."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    lowered = text.casefold()
    french_hints = re.findall(
        r"\b(?:le|la|les|du|de|des|un|une|est|pour|quel|quelle|quels|quelles|"
        r"comment|combien|qui|quoi|montant|frais|compte|livret|taux)\b",
        lowered,
    )
    if french_hints:
        return "fr"
    return "en"


class TranslationCache:
    """Tiny JSON cache, same spirit as the embedding cache.

    Shape: {"model": ..., "entries": {sha256(model+target+text): {target,
    preview, translation}}}. A broken cache must never block work.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict = {}
        self.model = ""
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = data.get("entries", {})
            self.model = data.get("model", "")
            print(f"[translate] cache loaded: {len(self.entries)} entries "
                  f"from {self.path.name}")
        except Exception as exc:  # noqa: BLE001 — a broken cache never blocks work
            print(f"[translate] WARNING: could not read cache {self.path}: {exc}")
            self.entries = {}

    @staticmethod
    def key(model: str, target: str, text: str) -> str:
        return hashlib.sha256(f"{model}\n{target}\n{text}".encode()).hexdigest()

    def get(self, model: str, target: str, text: str):
        return self.entries.get(self.key(model, target, text))

    def put(self, model: str, target: str, text: str, translation: str):
        self.entries[self.key(model, target, text)] = {
            "target": target,
            "preview": text[-90:],
            "translation": translation,
        }

    def save(self):
        payload = {
            "model": self.model,
            "entries": self.entries,
            "note": "key = sha256(model + '\\n' + target + '\\n' + text); "
                    "preview shows the source text tail",
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )


class QueryTranslator:
    """Translate batches of questions with the configured Gemini model.

    Available only when a key is present (same key resolution as the
    embedder). Every method degrades gracefully: a failed batch returns None
    and records itself, so retrieval can proceed with the original query.
    """

    def __init__(self, cfg):
        self.provider = str(getattr(
            cfg, "QUERY_TRANSLATION_PROVIDER", "gemini")).strip().lower()
        if self.provider == "nvidia":
            self.model = str(getattr(
                cfg, "NVIDIA_TRANSLATION_MODEL", "moonshotai/kimi-k3"))
            fallback_cfg = getattr(cfg, "NVIDIA_TRANSLATION_FALLBACK_MODELS",
                                   "deepseek-ai/deepseek-v4-pro")
            self.base_url = str(getattr(
                cfg, "NVIDIA_TRANSLATION_BASE_URL",
                "https://integrate.api.nvidia.com/v1/chat/completions"))
        else:
            self.provider = "gemini"  # normalize anything else to the default
            self.model = str(getattr(
                cfg, "QUERY_TRANSLATION_MODEL", "gemini-3.5-flash-lite"))
            fallback_cfg = getattr(cfg, "QUERY_TRANSLATION_FALLBACK_MODELS", "")
            self.base_url = ""
        # If the primary model is unavailable for this key/project, fall back
        # to the configured alternates; the switch is loud and recorded.
        self.fallback_models = [
            m.strip() for m in str(fallback_cfg).split(",") if m.strip()]
        self.active_model = self.model
        self.cache_path = Path(getattr(cfg, "QUERY_TRANSLATION_CACHE_PATH", "translations_cache.json"))
        self.cache = TranslationCache(self.cache_path)
        self.cache.model = self.model

        self.api_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.dropped: list[str] = []  # "ar -> fr (first words...)"
        self.last_error = ""          # last API error, for transparent failure
        self._client = None
        self._types = None
        self.available = self._make_client()

    # -- client ------------------------------------------------------------

    def _make_client(self) -> bool:
        """Provider key + client handle; actionable message on failure."""
        import os

        if self.provider == "nvidia":
            api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
            if not api_key:
                print("[translate] WARNING: no NVIDIA_API_KEY — query "
                      "translation DISABLED, using original queries only "
                      "(set NVIDIA_API_KEY in .env for "
                      "QUERY_TRANSLATION_PROVIDER=nvidia)")
                return False
            # Raw HTTPS handle (no SDK): dict + urllib in _call_api.
            self._client = {"api_key": api_key, "base_url": self.base_url}
            print(f"[translate] ready (nvidia) | model={self.model} | "
                  f"cache={self.cache_path.name} | "
                  f"fallbacks={self.fallback_models or 'none'}")
            return True

        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        if not api_key:
            print("[translate] WARNING: no Gemini key — query translation "
                  "DISABLED, using original queries only "
                  "(set GEMINI_API_KEY or GOOGLE_API_KEY in .env)")
            return False
        try:
            genai = require_provider_sdk("google.genai", "google-genai")
            types = require_provider_sdk("google.genai.types", "google-genai")
            self._types = types
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=120_000),  # milliseconds
            )
            print(f"[translate] ready (gemini) | model={self.model} | "
                  f"cache={self.cache_path.name}")
            return True
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — surface, but keep lab usable
            print(f"[translate] WARNING: could not build client: {exc} — "
                  "using original queries only")
            return False

    # -- translation ---------------------------------------------------------

    def translate_many(self, texts: list[str], target: str) -> list[str] | None:
        """Translate a batch in ONE API call (numbered lines).

        Returns translations in input order, or None if any part fails
        (callers fall back to the original query and record the drop).
        """
        if target not in LANG_NAMES:
            print(f"[translate] unknown target language {target!r}")
            self.failures += 1
            return None
        if not texts:
            return []

        # Cache first: translate only the missing indices, splice back.
        results: list[str | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            cached = self.cache.get(self.model, target, text)
            if cached:
                results[i] = cached.get("translation")
                self.cache_hits += 1
            else:
                missing.append(i)
        if not missing:
            return [t for t in results if t is not None]  # all cached
        if not self.available:
            return None

        missing_texts = [texts[i] for i in missing]
        translated = self._call_api(missing_texts, target)
        if translated is None:
            self.dropped.append(f"batch {target} ({len(missing_texts)} line(s))")
            return None
        for idx, line in zip(missing, translated):
            results[idx] = line
            self.cache.put(self.model, target, texts[idx], line)
        self.cache.save()
        return [r for r in results if r is not None]

    def translate_one(self, text: str, target: str) -> str | None:
        out = self.translate_many([text], target)
        return out[0] if out else None

    def _call_api(self, texts: list[str], target: str) -> list[str] | None:
        """One numbered-lines request; one retry; falls back across models.

        The primary model is tried first; on API errors (e.g. a model that is
        not available for this key/project) the configured fallbacks are
        tried in order. The switch is printed and recorded (active_model), so
        a fallback is never silent. Parse failures retry once on the same
        model before moving on.
        """
        lang_name = LANG_NAMES[target]
        # Newlines would break the numbering; questions never need them.
        clean = [re.sub(r"\s+", " ", t).strip() for t in texts]
        prompt = (
            f"Translate each numbered line below into {lang_name}. "
            "Keep product and brand names (e.g. 'Atlas', 'Livret Croissance') "
            "as-is. Output ONLY the translations, one per line, in the SAME "
            "order, each prefixed with its number and a dot. No explanations.\n\n"
            + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(clean))
        )

        models = [self.active_model] + [
            m for m in self.fallback_models if m != self.active_model]
        for model in models:
            if model != self.active_model:
                print(f"[translate] WARNING: model {self.active_model!r} "
                      f"failed; retrying with fallback {model!r}")
                self.active_model = model
                print(f"[translate] active model switched to {model}")
            for attempt in range(2):
                try:
                    self.api_calls += 1
                    text = self._chat(model, prompt)
                    parsed = self._parse(text or "", len(clean))
                    if parsed is not None:
                        return parsed
                    self.last_error = f"malformed response (model={model})"
                    print(f"[translate] WARNING: malformed response for target "
                          f"{target} (attempt {attempt + 1}/2), retrying")
                except Exception as exc:  # noqa: BLE001 — API/network: warn + retry
                    self.last_error = f"{type(exc).__name__}: {exc} (model={model})"
                    print(f"[translate] WARNING: {self.last_error} "
                          f"(attempt {attempt + 1}/2)")
                if attempt == 0:
                    time.sleep(1.5)
        self.failures += 1
        return None

    def _chat(self, model: str, prompt: str) -> str:
        """One chat completion request; provider-branched (gemini / nvidia).

        Returns the assistant text (never raises for HTTP errors — the
        caller catches and retries/falls back; status codes are recorded in
        self.last_error for transparency).
        """
        if self.provider == "nvidia":
            import urllib.error
            import urllib.request

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2000,
                # NIM reasoning models support disabling thinking via the
                # chat template; we only want the translation, not a chain.
                "chat_template_kwargs": {"thinking": False},
            }
            req = urllib.request.Request(
                self._client["base_url"],
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._client['api_key']}",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:200]
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"NVIDIA API HTTP {exc.code}: {detail or exc.reason}"
                ) from exc
            except (urllib.error.URLError, TimeoutError,
                    ConnectionError) as exc:
                raise RuntimeError(f"NVIDIA API connection error: {exc}") from exc
            data = json.loads(body)
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"NVIDIA API unexpected response: "
                    f"{json.dumps(data)[:200]}") from exc
            # Strip reasoning fences if a model returns them anyway.
            content = re.sub(r"```(?:thinking|reasoning)?\s*", "", content)
            return content.strip()
        # gemini (default)
        response = self._client.models.generate_content(model=model,
                                                        contents=[prompt])
        return response.text or ""

    @staticmethod
    def _parse(output: str, expected: int) -> list[str] | None:
        """Parse numbered lines; strict — wrong count means retry/fallback."""
        lines: dict[int, str] = {}
        for raw in output.splitlines():
            m = _NUMBERED_LINE_RE.match(raw)
            if not m:
                continue
            lines[int(m.group(1))] = m.group(2).strip().strip('"“”')
        if len(lines) != expected or sorted(lines) != list(range(1, expected + 1)):
            return None
        return [lines[i] for i in range(1, expected + 1)]

    # -- variants -------------------------------------------------------------

    def build_variants(self, question_text: str, query_lang: str | None,
                       corpus_langs: list[str]) -> list[dict]:
        """Original query + one translation per corpus language (if any).

        Each dict: {label, lang, translated, text}. On failure the target is
        dropped (recorded in self.dropped) — never a broken variant.
        """
        variants: list[dict] = [{
            "label": f"{query_lang or detect_language(question_text)}(original)",
            "lang": query_lang,
            "translated": False,
            "text": question_text,
        }]
        for target in sorted(set(corpus_langs)):
            if target == query_lang:
                continue
            translated = self.translate_one(question_text, target)
            if translated is None:
                self.dropped.append(
                    f"{query_lang or '?'} -> {target} ({question_text[:50]!r})")
                continue
            variants.append({
                "label": f"{target}(translated)",
                "lang": target,
                "translated": True,
                "text": translated,
            })
        return variants

    # -- reporting ------------------------------------------------------------

    def summary(self) -> str:
        return (f"model={self.model} active={self.active_model} "
                f"available={self.available} api_calls={self.api_calls} "
                f"cache_hits={self.cache_hits} failures={self.failures} "
                f"dropped={len(self.dropped)}"
                + (f" last_error={self.last_error!r}" if self.last_error else ""))
