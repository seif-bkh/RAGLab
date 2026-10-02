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
                 "تكافل", "ربا", "سلف", "استصناع"]
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


def run(api: Api) -> int:
    # 1) df + rollup from the live index
    df: Counter = Counter()
    rollup: list[dict] = []
    total = offset = 0
    while True:
        status, page = api.get("/chunks", params={"limit": 100, "offset": offset})
        if status != 200:
            print(f"[diag] GET /chunks answered HTTP {status}: {page}")
            return 1
        total = page.get("total", 0)
        if offset == 0:
            rollup = page.get("documents") or []
        for item in page.get("items") or []:
            text = item.get("text") or ""
            for term in BRIDGE_ARABIC:
                if term in text:
                    df[term] += 1
        offset += len(page.get("items") or [])
        if offset >= total or not page.get("items"):
            break
    # 2) does the deployment's window reach the evidence?
    status, search = api.post("/search", payload={
        "question": "what is murabaha?", "k": 5})
    hits_with_term = sum(1 for h in (search.get("hits") or [])
                         if "مرابح" in (h.get("text") or "")) \
        if status == 200 else -1
    k_used = len(search.get("hits") or []) if status == 200 else 0

    print(f"[diag] index: {total} chunks | search top-{k_used}: "
          f"{max(hits_with_term, 0)} hit(s) contain المرابحة")
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
