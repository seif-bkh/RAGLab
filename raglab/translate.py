"""translate.py — query translation for cross-lingual retrieval (EXPERIMENT).

The v2 diagnostics showed the remaining failures are LANGUAGE ROUTING: an
Arabic question clusters on Arabic chunks while the fact also exists in the
French document. Chunking cannot fix that, so this module translates each
query into every corpus language and retrieval runs per variant (see
store.best_variant_merge, "best-score fusion"): each chunk keeps the score
from the variant in which it matches best, so a French chunk can win even for
an Arabic question.

Implementation notes:
- Uses the SAME Google AI Studio key (GEMINI_API_KEY / GOOGLE_API_KEY) and the
  SAME google-genai SDK as the embedder — no new dependency, no new secret.
- Translation model configurable: QUERY_TRANSLATION_MODEL, default
  `gemini-2.5-flash` (free tier, 1500 RPD / 15 RPM — we need at most 2 calls
  per run because whole batches are translated in a single numbered-lines
  request).
- Results cached in translations_cache.json (sha256(model + target + text)),
  so CI warm-ups and local re-runs cost nothing.
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
        self.model = str(getattr(cfg, "QUERY_TRANSLATION_MODEL", "gemini-2.5-flash"))
        self.cache_path = Path(getattr(cfg, "QUERY_TRANSLATION_CACHE_PATH", "translations_cache.json"))
        self.cache = TranslationCache(self.cache_path)
        self.cache.model = self.model

        self.api_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.dropped: list[str] = []  # "ar -> fr (first words...)"
        self._client = None
        self._types = None
        self.available = self._make_client()

    # -- client ------------------------------------------------------------

    def _make_client(self) -> bool:
        """Same key + SDK as the embedder; actionable message on failure."""
        import os

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
            print(f"[translate] ready | model={self.model} | cache={self.cache_path.name}")
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
        """One numbered-lines request; one retry if parsing fails."""
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
        for attempt in range(2):
            try:
                self.api_calls += 1
                response = self._client.models.generate_content(
                    model=self.model, contents=[prompt]
                )
                parsed = self._parse(response.text or "", len(clean))
                if parsed is not None:
                    return parsed
                print(f"[translate] WARNING: malformed response for target "
                      f"{target} (attempt {attempt + 1}), retrying")
            except Exception as exc:  # noqa: BLE001 — API/network: warn and retry once
                print(f"[translate] WARNING: {type(exc).__name__} translating "
                      f"to {target} (attempt {attempt + 1}/2): {exc}")
            if attempt == 0:
                time.sleep(1.5)
        self.failures += 1
        return None

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
        return (f"model={self.model} available={self.available} "
                f"api_calls={self.api_calls} cache_hits={self.cache_hits} "
                f"failures={self.failures} dropped={len(self.dropped)}")
