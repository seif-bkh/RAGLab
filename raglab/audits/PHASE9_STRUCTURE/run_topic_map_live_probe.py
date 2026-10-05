#!/usr/bin/env python3
"""Paired low-N live comparison of the pre- and post-Phase-9 topic maps.

Each of six independent multilingual questions is sent once through the exact
pre-enrichment implementation from BASELINE_COMMIT and once through the
active implementation. Both use the same pinned xKiro model. The legacy
source is loaded from git history, not reimplemented, so the baseline retains
its historical prompt, parser, and substring topic validation.

The probe calls only interrogate.interrogate (the descriptive analysis call).
It bypasses POST /answer's sufficiency gate and performs no retrieval,
embedding, or answer generation. Live inference is guarded to GitHub Actions;
--validate-only is safe to run locally.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

RAGLAB = Path(__file__).resolve().parents[2]
REPO = RAGLAB.parent
if str(RAGLAB) not in sys.path:
    sys.path.insert(0, str(RAGLAB))

PLAN_PATH = Path(__file__).resolve().with_name("topic_map_live_probes.json")
DEFAULT_OUTPUT = RAGLAB / "results" / "phase9_topic_map" / "live_probe_results.json"
BASELINE_COMMIT = "965634f966fbcce3c6987d693d38d3d0e8780f06"
BASELINE_MAP_PATH = "raglab/topic_map.py"
BASELINE_INTERROGATE_PATH = "raglab/interrogate.py"
ARMS = ("before", "after")


def load_plan() -> dict:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "phase9-topic-map-live-probes-1.0":
        raise ValueError("unsupported topic-map live probe schema")
    if plan.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("probe plan baseline commit differs from the pinned historical implementation")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("probe plan must contain a non-empty cases list")
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("probe case IDs must be present and unique")
    for case in cases:
        if (not isinstance(case.get("question"), str) or not case["question"].strip()
                or case.get("language") not in {"ar", "fr", "en"}
                or not isinstance(case.get("accepted_topic_ids"), list)):
            raise ValueError(f"invalid probe case: {case.get('id')}")
        expected = case.get("expected_classification")
        if expected is not None and expected not in {
                "regulatory", "procedural", "definitional", "numeric", "non_banking"}:
            raise ValueError(f"invalid expected classification for {case['id']}")
        if any(not isinstance(topic_id, str) or not topic_id
               for topic_id in case["accepted_topic_ids"]):
            raise ValueError(f"invalid accepted topic IDs for {case['id']}")
    return plan


def resolve_corpus():
    import profiles
    state = profiles.default_state()
    return state, profiles.data_dirs(state)


def make_active_catalog(data_dirs):
    import topic_map
    all_entries = topic_map.build_topic_map(data_dirs)
    entries = topic_map.prompt_entries(data_dirs)
    if not entries:
        raise ValueError("the active corpus produced an empty topic map")
    return all_entries, entries, topic_map.render_topic_map(entries)


def _git_source(commit_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{commit_path}"],
        cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def load_legacy_modules():
    """Load the exact historical map and interrogator without shadowing live imports."""
    legacy_map_source = _git_source(BASELINE_MAP_PATH)
    legacy_interrogate_source = _git_source(BASELINE_INTERROGATE_PATH)

    legacy_map = ModuleType("_phase9_pre_enrichment_topic_map")
    legacy_map.__file__ = str(RAGLAB / "topic_map.py")
    legacy_map.__package__ = ""
    exec(compile(legacy_map_source, f"{BASELINE_COMMIT}:{BASELINE_MAP_PATH}", "exec"),
         legacy_map.__dict__)

    legacy_interrogate = ModuleType("_phase9_pre_enrichment_interrogate")
    legacy_interrogate.__file__ = str(RAGLAB / "interrogate.py")
    legacy_interrogate.__package__ = ""
    previous = sys.modules.get("topic_map")
    sys.modules["topic_map"] = legacy_map
    try:
        exec(compile(legacy_interrogate_source,
                     f"{BASELINE_COMMIT}:{BASELINE_INTERROGATE_PATH}", "exec"),
             legacy_interrogate.__dict__)
    finally:
        if previous is None:
            sys.modules.pop("topic_map", None)
        else:
            sys.modules["topic_map"] = previous
    if legacy_interrogate.__dict__.get("topic_map") is not legacy_map:
        raise RuntimeError("historical interrogator did not bind to its historical topic map")
    if previous is not None and sys.modules.get("topic_map") is not previous:
        raise RuntimeError("loading the historical modules changed the active topic map")
    return legacy_map, legacy_interrogate


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _legacy_normalized(value: str) -> str:
    # Matches the pre-enrichment interrogator's whitespace normalization exactly.
    return re.sub(r"\s+", " ", str(value or "").strip())


def _legacy_line(entry: dict) -> str:
    unit = f" [{entry['unit_id']}]" if entry.get("unit_id") else ""
    return f"({entry['document']}{unit}) {entry['heading']}"


def validate_plan_against_catalog(plan: dict, active_entries: list[dict],
                                  legacy_entries: list[dict]) -> None:
    active_by_id = {entry.get("topic_id"): entry for entry in active_entries}
    legacy_keys = {(entry.get("document"), entry.get("heading"), entry.get("unit_id"))
                   for entry in legacy_entries}
    for case in plan["cases"]:
        missing = set(case["accepted_topic_ids"]) - set(active_by_id)
        if missing:
            raise ValueError(f"{case['id']} expects topic IDs not visible in the active map: "
                             + ", ".join(sorted(missing)))
        # Check that every target has a comparable source identity in the old map.
        for topic_id in case["accepted_topic_ids"]:
            entry = active_by_id[topic_id]
            if entry.get("unit_id"):
                legacy_key = (entry["document"], entry["heading"], entry["unit_id"])
                matches = legacy_key in legacy_keys
            else:
                matches = any(doc == entry["document"] and heading == entry["heading"]
                              for doc, heading, _unit_id in legacy_keys)
            if not matches:
                raise ValueError(f"{case['id']} target {topic_id} has no comparable legacy topic")


def legacy_selected_ids(selected_topics: list[str], legacy_entries: list[dict],
                        active_entries: list[dict]) -> dict:
    """Map exact historical labels back to stable source identities, conservatively."""
    current_by_identity: dict[tuple[str, str], list[str]] = {}
    for entry in active_entries:
        current_by_identity.setdefault((entry["document"], entry["heading"]), []).append(
            entry["topic_id"])

    ids, unmapped, ambiguous = [], [], []
    for candidate in selected_topics:
        key = _normalized(candidate)
        matches = []
        for entry in legacy_entries:
            aliases = {_normalized(entry["heading"]), _normalized(_legacy_line(entry)),
                       _normalized("- " + _legacy_line(entry))}
            if entry.get("unit_id"):
                aliases.add(_normalized(entry["unit_id"]))
            if key and key in aliases:
                matches.append(entry)
        identities = {(entry["document"], entry["heading"], entry.get("unit_id"))
                      for entry in matches}
        if not identities:
            unmapped.append(candidate)
            continue
        if len(identities) > 1:
            ambiguous.append(candidate)
            continue
        entry = matches[0]
        if entry.get("unit_id"):
            resolved = [entry["unit_id"]]
        else:
            resolved = current_by_identity.get((entry["document"], entry["heading"]), [])
        if not resolved:
            unmapped.append(candidate)
            continue
        if len(set(resolved)) != 1:
            ambiguous.append(candidate)
            continue
        for topic_id in resolved:
            if topic_id not in ids:
                ids.append(topic_id)
    return {"topic_ids": ids, "unmapped_topics": unmapped,
            "ambiguous_topics": ambiguous}


def _bounded_text(value, limit: int = 1200):
    if not isinstance(value, str):
        return None
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return text[:limit]


def _json_object(raw: str):
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class CapturingClient:
    """Capture one logical response while delegating to the pinned client."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.response = None
        self.error = None
        self.calls = 0

    def chat(self, *args, **kwargs):
        from nvidia_api import safe_error
        self.calls += 1
        try:
            self.response = self.delegate.chat(*args, **kwargs)
            return self.response
        except Exception as exc:  # preserve the production parser's fail-closed behavior
            self.error = safe_error(exc)
            raise


def _usage_summary(response) -> dict:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return {}
    return {key: usage[key] for key in
            ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(usage.get(key), (int, float))}


def price_verification(client, model: str) -> dict:
    """Record the live free-price evidence already enforced by the pinned client."""
    from free_gateway import free_eligibility

    pricing = getattr(client, "pricing", None)
    if not isinstance(pricing, dict) or not isinstance(pricing.get("catalog"), dict):
        raise RuntimeError("pinned xKiro client did not retain its live pricing catalog")
    eligible, reason, row = free_eligibility("xkiro", model, pricing["catalog"])
    if not eligible:
        raise RuntimeError(f"pinned xKiro model failed free-price verification: {reason}")
    prices = row.get("pricing") or {}
    return {
        "verified": True,
        "reason": reason,
        "checked_at": pricing.get("checked_at"),
        "catalog_url": pricing.get("url"),
        "access_tier": row.get("access_tier"),
        "currency": prices.get("currency"),
        "unit": prices.get("unit"),
        "token_prices": {key: prices[key] for key in
                          ("input", "output", "cache_read", "cache_write")
                          if key in prices},
    }


def run_case(case: dict, *, arm: str, generator, data_dirs,
             active_entries: list[dict], legacy_entries: list[dict],
             legacy_text: str, legacy_interrogate, call_tracker: dict | None = None) -> dict:
    import interrogate as active_interrogator

    if arm == "after":
        module = active_interrogator
        entries = active_entries
        messages = module.interrogation_messages(case["question"], data_dirs, entries)
        raw_field = "nearest_topic_ids"
    elif arm == "before":
        module = legacy_interrogate
        entries = legacy_entries
        messages = module.interrogation_messages(case["question"], data_dirs)
        raw_field = "nearest_topics"
    else:
        raise ValueError(f"unknown comparison arm {arm!r}")

    capture = CapturingClient(generator.client)
    if call_tracker is not None:
        call_tracker["capture"] = capture
    original_client = generator.client
    generator.client = capture
    started = time.perf_counter()
    try:
        analysis = module.interrogate(generator, case["question"], data_dirs)
    finally:
        generator.client = original_client
    elapsed = round(time.perf_counter() - started, 3)

    response = capture.response
    raw_text = response.get("text") if isinstance(response, dict) else None
    payload = _json_object(raw_text)
    raw_candidates = payload.get(raw_field) if payload else None
    field_ok = isinstance(raw_candidates, list)
    candidates = raw_candidates if field_ok else []
    string_candidates = [value.strip() for value in candidates
                         if isinstance(value, str) and value.strip()]
    blank_string_count = sum(isinstance(value, str) and not value.strip()
                             for value in candidates)
    non_string_count = sum(not isinstance(value, str) for value in candidates)

    if arm == "after":
        prompt_ids = {entry["prompt_id"] for entry in entries}
        invalid_count = (sum(value not in prompt_ids for value in string_candidates)
                         + blank_string_count)
        protocol_ok = (
            field_ok
            and len(candidates) <= module.TOPICS_MAX
            and non_string_count == 0
            and blank_string_count == 0
            and invalid_count == 0
            and (bool(candidates) or not case["accepted_topic_ids"])
            and len(set(string_candidates)) == len(string_candidates)
        )
        resolved_ids = list(analysis.get("topic_ids", [])) if analysis else []
        selected_topics = list(analysis.get("topics", [])) if analysis else []
        legacy_membership_ok = None
        unmapped_topics, ambiguous_topics = [], []
    else:
        prompt_text = _legacy_normalized(legacy_text)
        invalid_count = (sum(_legacy_normalized(value) not in prompt_text
                             for value in string_candidates) + blank_string_count)
        legacy_membership_ok = (
            field_ok and len(candidates) <= module.TOPICS_MAX
            and non_string_count == 0 and blank_string_count == 0
            and invalid_count == 0)
        protocol_ok = None
        selected_topics = list(analysis.get("topics", [])) if analysis else []
        mapping = legacy_selected_ids(selected_topics, entries, active_entries)
        resolved_ids = mapping["topic_ids"]
        unmapped_topics = mapping["unmapped_topics"]
        ambiguous_topics = mapping["ambiguous_topics"]

    expected_ids = case["accepted_topic_ids"]
    expected_hit = (bool(set(resolved_ids) & set(expected_ids)) if expected_ids else None)
    expected_classification = case.get("expected_classification")
    actual_classification = (analysis.get("classification") if analysis else
                             payload.get("classification") if payload else None)
    classification_match = (
        actual_classification == expected_classification
        if expected_classification else None)

    if capture.error:
        status = "api_error"
    elif not isinstance(raw_text, str) or not raw_text.strip():
        status = "empty_response"
    elif payload is None:
        status = "malformed_json"
    elif analysis is None:
        status = "parser_rejected"
    else:
        status = "parsed"

    entries_by_id = {entry["topic_id"]: entry for entry in active_entries}
    selected_sources = []
    for topic_id in resolved_ids:
        source = entries_by_id.get(topic_id, {}).get("document")
        if source and source not in selected_sources:
            selected_sources.append(source)

    prompt_candidates = string_candidates[:module.TOPICS_MAX]
    return {
        "arm": arm,
        "status": status,
        "elapsed_seconds": elapsed,
        "logical_chat_calls": capture.calls,
        "prompt_chars": len(messages[1].get("content", "")),
        "response_field": raw_field,
        "response_field_present": field_ok,
        "returned_prompt_ids": prompt_candidates if arm == "after" else [],
        "returned_topic_labels": prompt_candidates if arm == "before" else [],
        "invalid_candidate_count": invalid_count + non_string_count,
        "prompt_id_protocol_ok": protocol_ok,
        "legacy_substring_membership_ok": legacy_membership_ok,
        "selected_topic_ids": resolved_ids,
        "selected_topics": selected_topics,
        "selected_sources": selected_sources,
        "unmapped_legacy_topics": unmapped_topics,
        "ambiguous_legacy_topics": ambiguous_topics,
        "expected_topic_hit": expected_hit,
        "expected_classification": expected_classification,
        "actual_classification": actual_classification,
        "classification_match": classification_match,
        "paraphrase": analysis.get("paraphrase") if analysis else None,
        "confidence": analysis.get("confidence") if analysis else None,
        "usage": _usage_summary(response),
        "raw_response_preview": _bounded_text(raw_text),
        "error": capture.error,
    }


def summarize(report: dict) -> str:
    cases = report.get("cases", [])
    parts = [
        f"model={report.get('model', 'unknown')}",
        f"live_zero_price_verified={str((report.get('free_price_verification') or {}).get('verified') is True).lower()}",
        f"paired_cases={len(cases)}",
        f"logical_calls={report.get('logical_chat_calls', 0)}",
    ]
    for arm, label in (("before", "before"), ("after", "after")):
        rows = [case.get(arm, {}) for case in cases]
        parsed = sum(row.get("status") == "parsed" for row in rows)
        target_rows = [row for row in rows if row.get("expected_topic_hit") is not None]
        target_hits = sum(row.get("expected_topic_hit") is True for row in target_rows)
        class_rows = [row for row in rows if row.get("classification_match") is not None]
        class_hits = sum(row.get("classification_match") is True for row in class_rows)
        errors = sum(row.get("status") in {"api_error", "probe_error"} for row in rows)
        if arm == "before":
            valid = sum(row.get("legacy_substring_membership_ok") is True for row in rows)
            metric = f"legacy_substring_membership={valid}/{len(rows)}"
        else:
            valid = sum(row.get("prompt_id_protocol_ok") is True for row in rows)
            metric = f"exact_prompt_ID_protocol={valid}/{len(rows)}"
        parts.append(
            f"{label}[parsed={parsed}/{len(rows)},{metric},topic_hits={target_hits}/{len(target_rows)},"
            f"non_banking={class_hits}/{len(class_rows)},errors={errors}]")
    before = report.get("before_map", {})
    after = report.get("after_map", {})
    parts.append(f"map_entries={before.get('entry_count')}->{after.get('entry_count')}")
    parts.append(f"map_chars={before.get('rendered_chars')}->{after.get('rendered_chars')}")
    return "; ".join(parts)


def annotation(report: dict) -> str:
    details = []
    for case in report.get("cases", []):
        arms = []
        for arm in ARMS:
            row = case.get(arm, {})
            hit = row.get("expected_topic_hit")
            hit_label = "na" if hit is None else ("hit" if hit else "miss")
            cls = row.get("actual_classification") or "?"
            chosen = (", ".join(row.get("returned_prompt_ids", [])) if arm == "after"
                      else ", ".join(row.get("returned_topic_labels", []))) or "none"
            text_preview = row.get("paraphrase") or row.get("raw_response_preview") or ""
            paraphrase = re.sub(r"[\r\n|;]+", " ", text_preview)[:48]
            arms.append(f"{arm}={row.get('status')}/{hit_label}/{cls}/{chosen}/{paraphrase or 'no-preview'}")
        details.append(f"{case.get('id')}[{' ; '.join(arms)}]")
    return summarize(report) + ("; cases=" + " || ".join(details) if details else "")


def render_markdown(report: dict) -> str:
    before_map = report.get("before_map", {})
    after_map = report.get("after_map", {})
    lines = [
        "# Phase 9 — paired live topic-map selection comparison",
        "",
        "**Scope:** each independent case is sent through both historical and active `interrogate.interrogate` implementations with the same pinned xKiro model. This directly exercises topic analysis only: it bypasses `POST /answer`'s sufficiency gate and performs no retrieval, embedding, or answer generation.",
        "**Interpretation:** six paired cases are a small diagnostic, not a quality benchmark, significance test, or proof of improved end-user answers. The before/after difference includes the historical prompt/parser and the map format together; it does not isolate a causal effect of map content alone.",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Provider/model: `{report.get('provider')}` / `{report.get('model')}`",
        f"- Live zero-price verification: `{report.get('free_price_verification') or report.get('free_price_verification_error') or report.get('setup_error') or 'not run'}`",
        f"- Historical arm: `{before_map.get('baseline_commit')}`; {before_map.get('entry_count')} entries; {before_map.get('rendered_chars')} rendered map characters",
        f"- Active arm: {after_map.get('entry_count')} entries; {after_map.get('rendered_chars')} rendered map characters",
        f"- Summary: `{summarize(report)}`",
        "",
        "| Case | Lang | Before status | Before mapped IDs | Before target | Before class | After status | After prompt IDs | After target | After class |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in report.get("cases", []):
        before = case.get("before", {})
        after = case.get("after", {})
        expected_before = before.get("expected_topic_hit")
        expected_after = after.get("expected_topic_hit")
        old_hit = "—" if expected_before is None else ("yes" if expected_before else "no")
        new_hit = "—" if expected_after is None else ("yes" if expected_after else "no")
        old_ids = ", ".join(before.get("selected_topic_ids", [])) or "—"
        new_ids = ", ".join(after.get("returned_prompt_ids", [])) or "—"
        lines.append(
            f"| {case.get('id')} | {case.get('language')} | {before.get('status')} | `{old_ids}` | {old_hit} | {before.get('actual_classification') or '—'} | "
            f"{after.get('status')} | `{new_ids}` | {new_hit} | {after.get('actual_classification') or '—'} |")
    lines.extend(["", "## Paired probe details", ""])
    for case in report.get("cases", []):
        lines.extend([
            f"### {case.get('id')} ({case.get('language')})",
            "",
            f"- Question: {case.get('question')}",
            f"- Expected stable topic IDs: `{', '.join(case.get('accepted_topic_ids', [])) or 'none (out of corpus)'}`",
        ])
        for arm, label in (("before", "Before — baseline"), ("after", "After — active map")):
            row = case.get(arm, {})
            raw_preview = (_bounded_text(row.get("raw_response_preview"), 500) or "—")
            raw_preview = (raw_preview.replace("|", "\\|")
                           .replace("`", "\\`").replace("\r", " ").replace("\n", " "))
            lines.extend([
                f"- {label}: status `{row.get('status')}`; returned labels `{'; '.join(row.get('returned_topic_labels', [])) or '—'}`; returned prompt IDs `{', '.join(row.get('returned_prompt_ids', [])) or '—'}`; resolved stable IDs `{', '.join(row.get('selected_topic_ids', [])) or '—'}`; target hit `{row.get('expected_topic_hit')}`; classification `{row.get('actual_classification')}`.",
                f"  - Paraphrase: {row.get('paraphrase') or '—'}",
                f"  - Raw response preview: `{raw_preview}`",
                f"  - Legacy unmapped / ambiguous labels: `{'; '.join(row.get('unmapped_legacy_topics', [])) or '—'}` / `{'; '.join(row.get('ambiguous_legacy_topics', [])) or '—'}`",
                f"  - Error: {row.get('error') or 'none'}",
            ])
        lines.append("")
    return "\n".join(lines) + "\n"


def run(validate_only: bool = False, output: Path = DEFAULT_OUTPUT) -> int:
    if not validate_only and os.environ.get("GITHUB_ACTIONS") != "true":
        print("live model calls are restricted to GitHub Actions; use --validate-only locally",
              file=sys.stderr)
        return 2

    from artifacts import write_json
    import profiles

    plan = load_plan()
    state, data_dirs = resolve_corpus()
    all_active_entries, active_entries, active_text = make_active_catalog(data_dirs)
    legacy_topic_map, legacy_interrogator = load_legacy_modules()
    legacy_entries = legacy_topic_map.build_topic_map(data_dirs)
    legacy_text = legacy_topic_map.for_prompt(data_dirs)
    if not legacy_entries:
        raise ValueError("historical topic map produced no entries")
    # The old map is built from chunker heading metadata, so its entry count
    # legitimately varies between cl100k_base and the documented local
    # estimator fallback. Record the environment; never pin the count.
    import chunker
    legacy_tokenizer = chunker.tokenizer_identity()
    validate_plan_against_catalog(plan, active_entries, legacy_entries)

    if validate_only:
        print(f"paired probe plan OK: {len(plan['cases'])} cases x 2 arms; "
              f"baseline {BASELINE_COMMIT[:7]}={len(legacy_entries)} entries / {len(legacy_text)} chars "
              f"(tokenizer={legacy_tokenizer}); "
              f"active={len(active_entries)} entries / {len(active_text)} chars; "
              f"all expected stable IDs map to both versions; no model call")
        return 0

    if state["answer"] != profiles.SUPPORTED_ANSWER:
        raise RuntimeError("default profile is not the pinned xKiro answer model")

    base = {
        "schema_version": "phase9-topic-map-paired-live-results-1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": plan["purpose"],
        "scope": "paired historical-vs-active interrogation component; bypasses POST /answer sufficiency gate; no retrieval, embeddings, or answer generation",
        "provider": state["answer"]["provider"],
        "model": state["answer"]["model"],
        "corpus_dirs": [Path(path).name for path in data_dirs],
        "free_price_verification": None,
        "before_map": {
            "baseline_commit": BASELINE_COMMIT,
            "entry_count": len(legacy_entries),
            "rendered_chars": len(legacy_text),
            "tokenizer_identity": legacy_tokenizer,
            "prompt_version": "exact historical topic_map.py + interrogate.py",
        },
        "after_map": {
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      capture_output=True, text=True, check=True).stdout.strip(),
            "entry_count": len(active_entries),
            "catalog_entry_count": len(all_active_entries),
            "rendered_chars": len(active_text),
            "prompt_version": "active topic_map.py + interrogate.py",
        },
        "expected_logical_chat_calls": len(plan["cases"]) * len(ARMS),
        "logical_chat_calls": 0,
        "cases": [],
    }

    try:
        local = profiles.build_lab_config(state)
        generator = profiles.build_generator(local)
        base["free_price_verification"] = price_verification(
            generator.client, state["answer"]["model"])
    except Exception as exc:
        from nvidia_api import safe_error
        base["setup_error"] = safe_error(exc)
        base["summary"] = summarize(base)
        base["markdown"] = render_markdown(base)
        write_json(output, base)
        output.with_suffix(".md").write_text(base["markdown"], encoding="utf-8")
        print(base["summary"])
        print("ANNO| phase9-topic-map | " + annotation(base))
        return 1

    for index, case in enumerate(plan["cases"]):
        order = ("before", "after") if index % 2 == 0 else ("after", "before")
        paired = {
            "id": case["id"],
            "language": case["language"],
            "question": case["question"],
            "accepted_topic_ids": case["accepted_topic_ids"],
            "expected_classification": case.get("expected_classification"),
        }
        for arm in order:
            call_tracker = {}
            try:
                row = run_case(
                    case, arm=arm, generator=generator, data_dirs=data_dirs,
                    active_entries=active_entries, legacy_entries=legacy_entries,
                    legacy_text=legacy_text, legacy_interrogate=legacy_interrogator,
                    call_tracker=call_tracker)
            except Exception as exc:  # retain the paired row if one arm crashes
                from nvidia_api import safe_error
                capture = call_tracker.get("capture")
                row = {
                    "arm": arm, "status": "probe_error",
                    "logical_chat_calls": capture.calls if capture else 0,
                    "selected_topic_ids": [], "selected_topics": [],
                    "expected_topic_hit": False if case["accepted_topic_ids"] else None,
                    "actual_classification": None, "classification_match": None,
                    "returned_prompt_ids": [], "returned_topic_labels": [],
                    "unmapped_legacy_topics": [], "ambiguous_legacy_topics": [],
                    "raw_response_preview": _bounded_text(
                        capture.response.get("text") if capture and isinstance(capture.response, dict) else None),
                    "error": safe_error(exc),
                }
            paired[arm] = row
            base["logical_chat_calls"] += row.get("logical_chat_calls", 0)
        before_hit = paired["before"].get("expected_topic_hit")
        after_hit = paired["after"].get("expected_topic_hit")
        paired["comparison"] = {
            "topic_hit_change": (None if before_hit is None or after_hit is None
                                 else int(after_hit) - int(before_hit)),
            "classification_match_change": (
                None if paired["expected_classification"] is None
                else int(paired["after"].get("classification_match") is True)
                - int(paired["before"].get("classification_match") is True)),
        }
        base["cases"].append(paired)

    try:
        base["free_price_verification"] = price_verification(
            generator.client, state["answer"]["model"])
    except Exception as exc:
        from nvidia_api import safe_error
        base["free_price_verification_error"] = safe_error(exc)
    base["summary"] = summarize(base)
    base["markdown"] = render_markdown(base)
    write_json(output, base)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(base["markdown"], encoding="utf-8")
    print(base["summary"])
    print(f"results: {output.relative_to(RAGLAB)}")
    print(f"summary: {markdown_path.relative_to(RAGLAB)}")
    print("ANNO| phase9-topic-map | " + annotation(base))
    probe_errors = sum(case[arm].get("status") in {"api_error", "probe_error"}
                       for case in base["cases"] for arm in ARMS)
    return 1 if (probe_errors
                 or base["logical_chat_calls"] != base["expected_logical_chat_calls"]
                 or (base.get("free_price_verification") or {}).get("verified") is not True) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="reconstruct and validate both map versions without constructing a provider or calling a model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return run(validate_only=args.validate_only, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
