#!/usr/bin/env python3
"""diag_bridge.py — one-shot remote diagnostic for the cross-script bridge
(2026-10-02, the owner's 'still refused after rebuild' case).

WHY: a duplicated corpus (the same document pushed several times through
POST /documents) multiplies every term's document frequency. The bridge's
df guard (df <= CROSS_DF_MAX = 30) then honestly refuses to anchor
LEGITIMATE terms: on the owner's server df(المرابحة) ≈ 17 × 5 copies = 85
> 30, so 'what is murabaha?' is refused even though the corpus defines it.

WHAT IT DOES (read-only, three HTTP calls per 100 chunks):
  1. paginates GET /chunks  -> the EXACT df of every bridge term over the
     live index, plus the per-document rollup (duplicates are visible);
  2. runs POST /search for 'what is murabaha?' (and takaful as the
     differential: low-df term) at the deployment's k -> does the window
     even reach the evidence?
  3. prints a verdict: duplicates / df-blocked / window-too-narrow /
     should-answer, with the exact remediation for each.

Usage (from raglab/, against a running service):
    python diag_bridge.py [--base-url http://localhost:8000] [--token ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

from local_front import Api, DEFAULT_BASE_URL

# the declared bridge table's Arabic side (kept in sync manually — the
# diagnostic only needs the Arabic terms, and importing sufficiency here
# would tie the client to the lab's version)
BRIDGE_ARABIC = ["مرابحة", "مضاربة", "مشاركة", "صكوك", "اجارة",
                 # «سلف» dropped 2026-10-06: tokenized df=0 on this corpus
                 # (the text only has «سلفة», a different token), so counting
                 # it reported a df that could never anchor.
                 "تكافل", "ربا", "استصناع",
                 # 2026-10-05 field-vocabulary bridges (sufficiency
                 # .FIELD_BRIDGE_TERMS / FIELD_BRIDGE_PHRASES, gated by
                 # SUFFICIENCY_FIELD_BRIDGES_ENABLED). Their df matters for
                 # the TM03/TM04 arm, so the diagnostic has to measure them
                 # on the deployment's own index — a duplicated corpus
                 # multiplies every df and the scaled cap may not keep up.
                 "اعتماد", "مستندي", "صرف", "عمليات"]
DF_MAX = 30          # sufficiency.CROSS_DF_MAX (declared floor)
CALIBRATION_N = 339   # the calibrated corpus (fallback token estimator);
                      # tiktoken envs chunk the same corpus finer (~2.5x)


def df_cap(total_chunks: int) -> int:
    """The scaled anchor cap for this index size (mirrors
    sufficiency._df_cap): max(30, ceil(30 * N / 339))."""
    if total_chunks <= CALIBRATION_N:
        return DF_MAX
    return max(DF_MAX, -(-DF_MAX * total_chunks // CALIBRATION_N))


def verdict(df: Counter, k: int, hits_with_term: int, rollup: list[dict],
            total_chunks: int) -> list[str]:
    """Pure verdict logic — unit-tested."""
    notes: list[str] = []
    cap = df_cap(total_chunks)
    baked = {str(d.get("source", "")) for d in rollup
             if not str(d.get("source", "")).startswith("pushed-")}
    def _base(name):
        return str(name).removeprefix("pushed-")

    duplicates = [d for d in rollup
                  if str(d.get("source", "")).startswith("pushed-")
                  and _base(d.get("source")) in baked]
    extra_pushed = [d for d in rollup
                    if str(d.get("source", "")).startswith("pushed-")
                    and _base(d.get("source")) not in baked]
    if duplicates:
        names = ", ".join(sorted(str(d["source"])[:44] for d in duplicates))
        pushed = sum(int(d.get("chunks", 0)) for d in duplicates)
        notes.append(
            f"DUPLICATES: {len(duplicates)} pushed document(s) RE-PUSHING "
            f"baked-in documents (+{pushed} chunks; total {total_chunks}; "
            f"the baked-in corpus is {CALIBRATION_N} chunks under the "
            f"fallback estimator, ~2.5x more with tiktoken)\n"
            f"      → {names}\n"
            f"      fix: local_front menu 14 → remove each of these. The "
            f"pushed copies are often DEGRADED re-extractions (a pushed "
            f"visual-order law copy chunks worse than the baked-in one).\n"
            f"      (DELETE /documents purges its chunks immediately; no "
            f"reingest needed)")
    if extra_pushed:
        names = ", ".join(sorted(str(d["source"])[:44] for d in extra_pushed))
        notes.append(f"NEW PUSHED (kept — not duplicates of baked-in docs): "
                     f"{names}")
    for term in ("مرابحة", "تكافل"):
        df_t = df.get(term, 0)
        flag = "OK " if df_t <= cap else "BLOCKED"
        notes.append(f"df({term}) = {df_t} [{flag}] (scaled bridge guard: "
                     f"df <= {cap} for {total_chunks} chunks)")
    if df.get("مرابحة", 0) > cap:
        notes.append(
            "VERDICT: df-blocked — المرابحة behaves like boilerplate in "
            "THIS index (present in a huge share of chunks). This is not "
            "normal duplication; inspect the index contents.")
    elif hits_with_term == 0:
        notes.append(
            "VERDICT: window/retrieval — the top-k results do not contain "
            "any المرابحة chunk, so there is nothing to anchor. Raise "
            "RAGLAB_TOP_K (you are likely on 5; every live measurement "
            "used 20), set it in raglab/.env, then: docker compose up -d")
    else:
        notes.append(
            "VERDICT: should-answer — df is under the guard and the window "
            "reaches المرابحة evidence. If the question is still refused, "
            "the container is not running the bridge build; verify with:\n"
            "      docker compose exec raglab python -c "
            "\"import sufficiency; print(sufficiency.CROSS_BRIDGE_ENABLED)\"")
    return notes


def count_bridge_df(texts: list[str]) -> tuple[Counter, str]:
    """Document frequency of every bridge term over the given chunk texts.

    Counting MUST match what sufficiency._anchors() compares against, or the
    verdict lies. _anchors() works on TOKENIZED, normalized terms
    (sufficiency._hit_terms), where «الاعتماد» and «اعتماد» are the same term
    and «الصرف» does not contain the token «صرف». A plain substring count
    inflates short Arabic terms badly — measured on the 339-chunk corpus:
    صرف 95 substring vs 14 tokenized (6.8x), ربا 48 vs 13 (3.7x), تكافل 6 vs
    3 (2.0x). That produced false DF-BLOCKED verdicts on a 1713-chunk index
    (صرف reported 335 while the term still anchored).

    Returns (df, method) where method names the counting actually used, so the
    output never implies more precision than it has.
    """
    try:
        import sufficiency as _suf
    except Exception:                       # client-only install: degrade loudly
        _suf = None
    df: Counter = Counter()
    if _suf is None:
        for text in texts:
            for term in BRIDGE_ARABIC:
                if term in text:
                    df[term] += 1
        return df, "substring (sufficiency unavailable — df OVERSTATED)"
    wanted = {term: _suf._hit_terms(term) for term in BRIDGE_ARABIC}
    for text in texts:
        terms = _suf._hit_terms(text)
        for term, needle in wanted.items():
            if needle & terms:
                df[term] += 1
    return df, "tokenized (same normalization as sufficiency._anchors)"


def run(api: Api) -> int:
    # 1) df + rollup from the live index
    df: Counter = Counter()
    rollup: list[dict] = []
    total = offset = 0
    pages: list[str] = []
    while True:
        status, page = api.get("/chunks", params={"limit": 100, "offset": offset})
        if status != 200:
            print(f"[diag] GET /chunks answered HTTP {status}: {page}")
            return 1
        total = page.get("total", 0)
        if offset == 0:
            rollup = page.get("documents") or []
        items = page.get("items") or []
        pages.extend(item.get("text") or "" for item in items)
        offset += len(items)
        if offset >= total or not items:
            break
    df, method = count_bridge_df(pages)
    # 2) does the deployment's window reach the evidence?
    status, search = api.post("/search", payload={
        "question": "what is murabaha?", "k": 5})
    hits_with_term = sum(1 for h in (search.get("hits") or [])
                         if "مرابح" in (h.get("text") or "")) \
        if status == 200 else -1
    k_used = len(search.get("hits") or []) if status == 200 else 0

    print(f"[diag] index: {total} chunks | search top-{k_used}: "
          f"{max(hits_with_term, 0)} hit(s) contain المرابحة")
    # Per-term df against THIS index's scaled cap, so the 2026-10-05 field
    # bridges can be judged on the deployment's own corpus (a duplicated
    # index multiplies every df; the cap only scales with the chunk count).
    cap = df_cap(total)
    print(f"[diag] bridge-term df over the live index (scaled cap = {cap}; "
          f"counting: {method}):")
    for term in BRIDGE_ARABIC:
        n = df.get(term, 0)
        state = ("anchors" if 0 < n <= cap
                 else "DF-BLOCKED" if n > cap else "absent")
        print(f"  {term:<10} df={n:<6} {state}")
    for note in verdict(df, k_used, hits_with_term, rollup, total):
        print(f"  {note}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="diag_bridge", description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    api = Api(args.base_url, token=args.token)
    return run(api)


if __name__ == "__main__":
    sys.exit(main())
