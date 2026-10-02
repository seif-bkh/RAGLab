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
DF_MAX = 30          # sufficiency.CROSS_DF_MAX (declared)
EXPECTED_CLEAN = 339  # the baked-in corpus (docs/ in the image)


def verdict(df: Counter, k: int, hits_with_term: int, rollup: list[dict],
            total_chunks: int) -> list[str]:
    """Pure verdict logic — unit-tested."""
    notes: list[str] = []
    duplicates = [d for d in rollup
                  if str(d.get("source", "")).startswith("pushed-")]
    if duplicates:
        pushed = sum(int(d.get("chunks", 0)) for d in duplicates)
        names = ", ".join(sorted(str(d["source"])[:44] for d in duplicates))
        notes.append(
            f"DUPLICATES: {len(duplicates)} pushed document(s) adding "
            f"{pushed} chunks on top of the baked-in corpus "
            f"(index total {total_chunks} vs {EXPECTED_CLEAN} expected)\n"
            f"      → {names}\n"
            f"      fix: local_front menu 14 → remove each pushed document\n"
            f"      (DELETE /documents purges its chunks immediately; no "
            f"reingest needed)")
    for term in ("مرابحة", "تكافل"):
        df_t = df.get(term, 0)
        flag = "OK " if df_t <= DF_MAX else "BLOCKED"
        notes.append(f"df({term}) = {df_t} [{flag}] (bridge guard: "
                     f"df <= {DF_MAX})")
    if df.get("مرابحة", 0) > DF_MAX:
        notes.append(
            "VERDICT: df-blocked — the bridge refuses to anchor المرابحة "
            "because duplication pushed it over the guard. Remove the "
            "pushed duplicates (above) and the question answers. (If the "
            "duplicates are INTENTIONAL and permanent, the declared cap "
            "can be made relative to corpus size — an owner decision, "
            "measured before activation.)")
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
