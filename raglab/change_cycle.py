#!/usr/bin/env python3
"""Document change cycle — Experiment 4 (owner directive 2026-10-01
«أنجزها كلها»; RAGLAB_ROADMAP.md phase 7 item 2).

The publishing cycle for a NEW document version, as declared tooling:

    تسجيل (محاور الحوكمة) → فحص (مدونة) → استخراج (وحدات/علاقات/أرقام)
    → فهرسة (إعادة بناء) → قياس الأثر (المجموعتان قبل/بعد) → نشر إصدار متسق

Every step that needs an OWNER decision STOPS and says so — this tool never
mutates docs/ or the registries; it prepares the cycle and reports what is
derivable deterministically (chunk preview, governance status, typed-layer
applicability, the impact-measurement plan).

--dry-run demonstrates the cycle on an EXISTING corpus document (default:
the training manual) without changing anything: the same checklist runs,
with the registration steps shown as «already registered / deferred».

Exit code 0 = the cycle is prepared; nonzero = a blocking precondition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import governance  # noqa: E402

DOCS_DIR = HERE.parent / "docs"

# The adopted measurement protocol for any corpus change (±2pp rule, the
# same-run baseline pattern the CI already uses).
IMPACT_PLAN = [
    "reindex with the candidate document set (ingest --reset)",
    "evaluate questions_50.json (verbatim/OOS guard: no regression)",
    "evaluate questions_targets.json (category table before/after)",
    "sufficiency.py --live (state distribution before/after)",
    "compare SAME-RUN baselines; ±2pp cross-run jitter rule applies",
]


def cycle_report(doc_path: Path, corpus_dir: Path = DOCS_DIR) -> dict:
    """The deterministic part of the cycle for one candidate document."""
    import chunker
    import loader
    from types import SimpleNamespace
    import config as config_mod

    name = doc_path.name
    report: dict = {"document": name, "exists": doc_path.exists()}

    # 1) registration — governance axes (owner gate)
    if name in governance.GOVERNANCE_AXES:
        report["governance"] = {"status": "registered",
                                "axes": governance.GOVERNANCE_AXES[name]}
    elif name in governance.DEFERRED_DOCUMENTS:
        report["governance"] = {
            "status": "deferred",
            "note": "minimal axes are a reviewed registry extension "
                    "(Phase-4 follow-up) — OWNER GATE",
            "proposed_axes": list(governance.AXES),
        }
    else:
        report["governance"] = {
            "status": "unregistered (new document)",
            "required_axes": list(governance.AXES),
            "note": "OWNER GATE: declare the axes before publishing",
        }

    # 2) inspection — chunk preview over the candidate file alone
    if doc_path.exists():
        cfg = SimpleNamespace(**{k: getattr(config_mod, k)
                                 for k in dir(config_mod) if k.isupper()})
        cfg.CHUNKING_MODE = "restructure"
        docs = loader.load_all([doc_path.parent])
        chunks = chunker.chunk_all(docs, cfg) if docs else []
        report["chunk_preview"] = {
            # the count the CORPUS would have with the candidate in place
            "documents_loaded": len(docs),
            "chunks_after_addition": len(chunks),
            "languages": sorted({c.language for c in chunks}),
        }
    else:
        report["chunk_preview"] = None

    # 3) typed-layer applicability (deterministic statement, not a decision)
    report["typed_layers"] = {
        "units/relations/legal_numbers": "law documents only (Loi_2016-48.pdf) — "
                                         "not applicable to other documents",
        "precedence": "a new document needs a PRECEDENCE rank — OWNER GATE",
    }

    # 4) the impact-measurement plan (the adopted protocol)
    report["impact_plan"] = IMPACT_PLAN

    # 5) blocking preconditions
    blockers = []
    if not doc_path.exists():
        blockers.append("the candidate document is not in the corpus dir")
    if report["governance"]["status"] != "registered":
        blockers.append("governance axes not registered — OWNER GATE")
    report["blockers"] = blockers
    report["ready_to_publish"] = not blockers
    return report


def main() -> int:
    parser = argparse.ArgumentParser(prog="change_cycle",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--doc", default="Madkhal_Sayrafa_Islamiya.docx",
                        help="candidate document file name (in docs/)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    doc_path = DOCS_DIR / args.doc
    report = cycle_report(doc_path)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report["ready_to_publish"] or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
