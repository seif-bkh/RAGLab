#!/usr/bin/env python3
"""Run the six Phase-9 probe questions through the real POST /answer path.

The live runner uses the supported NVIDIA-embedding + xKiro/Qwen profile,
restructure chunks, vector retrieval, and the service's default answer policy
(including sufficiency commitment and Phase-8 interrogation). It writes a
case-level JSON/Markdown report and bounded GitHub-check annotations so the
results remain inspectable when Actions artifacts cannot be downloaded.

Live calls are restricted to GitHub Actions. --validate-only reads the probe
plan and active topic catalog locally without constructing a provider, embedding,
or calling a model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RAGLAB = Path(__file__).resolve().parents[2]
REPO = RAGLAB.parent
PHASE9 = Path(__file__).resolve().parent
if str(RAGLAB) not in sys.path:
    sys.path.insert(0, str(RAGLAB))

PLAN_PATH = PHASE9 / "topic_map_live_probes.json"
DEFAULT_OUTPUT = RAGLAB / "results" / "phase9_answer_probe" / "live_answer_results.json"
EXPECTED_PROFILE = {
    "embedding": {"provider": "nvidia", "model": "nvidia/nemotron-3-embed-1b"},
    "answer": {"provider": "xkiro", "model": "qwen/qwen3.8-max:free"},
}
MAX_LOGICAL_CHAT_CALLS = 18  # six cases x (Phase-8 interrogation + one answer), with headroom


def load_plan() -> dict:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "phase9-topic-map-live-probes-1.0":
        raise ValueError("unsupported Phase-9 probe schema")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("the answer probe requires the same six Phase-9 cases")
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or len(set(ids)) != len(ids):
        raise ValueError("probe case IDs must be present and unique")
    for case in cases:
        if (not isinstance(case.get("question"), str) or not case["question"].strip()
                or case.get("language") not in {"ar", "fr", "en"}
                or not isinstance(case.get("accepted_topic_ids"), list)):
            raise ValueError(f"invalid probe case: {case.get('id')}")
    return plan


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def active_catalog(data_dirs) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    import topic_map

    entries = topic_map.prompt_entries(data_dirs)
    if not entries:
        raise ValueError("active corpus produced an empty topic map")
    by_id = {entry["topic_id"]: entry for entry in entries}
    by_display: dict[str, list[dict]] = {}
    for entry in entries:
        by_display.setdefault(_normalize(entry.get("display")), []).append(entry)
    return entries, by_id, by_display


def validate_active_targets(plan: dict, by_id: dict[str, dict]) -> None:
    for case in plan["cases"]:
        missing = set(case["accepted_topic_ids"]) - set(by_id)
        if missing:
            raise ValueError(f"{case['id']} target IDs absent from active map: "
                             + ", ".join(sorted(missing)))


def _safe_usage(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()
            if isinstance(item, (int, float)) and not isinstance(item, bool)}


def _call_kind(messages) -> str:
    try:
        system = str(messages[0].get("content", "")).lower()
    except (IndexError, AttributeError, TypeError):
        return "other"
    if "requirements analyst" in system:
        return "interrogation"
    if "document-grounded banking assistant" in system:
        return "answer_generation"
    return "other"


class CountingChatClient:
    """Count real logical model calls without retaining prompts or key material."""

    def __init__(self, delegate):
        self._delegate = delegate
        self.events: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    @property
    def base_url(self):
        return getattr(self._delegate, "base_url", "")

    def chat(self, model, messages, *, max_tokens=4096):
        event = {"kind": _call_kind(messages), "model": model}
        started = time.monotonic()
        try:
            response = self._delegate.chat(model, messages, max_tokens=max_tokens)
            if isinstance(response, dict):
                event["served_model"] = response.get("served_model")
                event["usage"] = _safe_usage(response.get("usage"))
            return response
        except Exception as exc:
            from nvidia_api import safe_error
            event["error"] = safe_error(exc)
            raise
        finally:
            event["seconds"] = round(time.monotonic() - started, 3)
            self.events.append(event)


def _expected_documents(case: dict, by_id: dict[str, dict]) -> list[str]:
    docs = {str(by_id[topic_id].get("document") or by_id[topic_id].get("source"))
            for topic_id in case["accepted_topic_ids"]}
    return sorted(doc for doc in docs if doc and doc != "None")


def _case_result(case: dict, payload: dict, *, http_status: int,
                 error: str | None, by_id: dict[str, dict],
                 by_display: dict[str, list[dict]], calls: list[dict]) -> dict:
    sources_by_id = {source.get("source_id"): source
                     for source in payload.get("sources", [])
                     if isinstance(source, dict) and source.get("source_id")}
    claims = []
    cited_documents = set()
    for claim in payload.get("claims", []) or []:
        if not isinstance(claim, dict):
            continue
        evidence = []
        for item in claim.get("evidence", []) or []:
            if not isinstance(item, dict):
                continue
            source = sources_by_id.get(item.get("source_id"), {})
            document = source.get("document")
            if document:
                cited_documents.add(document)
            evidence.append({
                "source_id": item.get("source_id"),
                "document": document,
                "heading": source.get("heading"),
                "unit_id": source.get("unit_id"),
                "quote": item.get("quote"),
            })
        claims.append({"text": claim.get("text"), "evidence": evidence})

    interrogation = payload.get("interrogation")
    interrogation = interrogation if isinstance(interrogation, dict) else None
    selected_ids: list[str] = []
    ambiguous_topics: list[str] = []
    if interrogation:
        for topic in interrogation.get("topics", []) or []:
            matches = by_display.get(_normalize(topic), [])
            if len(matches) == 1:
                selected_ids.append(matches[0]["topic_id"])
            elif len(matches) > 1:
                ambiguous_topics.append(topic)
    expected_ids = case["accepted_topic_ids"]
    expected_docs = _expected_documents(case, by_id)
    target_doc_hit = (bool(cited_documents & set(expected_docs))
                      if expected_docs else None)
    topic_hit = (bool(set(selected_ids) & set(expected_ids))
                 if interrogation and expected_ids else None)
    expected_class = case.get("expected_classification")
    actual_class = interrogation.get("classification") if interrogation else None
    status = payload.get("status")
    is_oos_control = expected_class == "non_banking" and not expected_ids

    return {
        "id": case["id"],
        "language_expected": case["language"],
        "question": case["question"],
        "http_status": http_status,
        "http_error": error,
        "status": status,
        "reason": payload.get("reason"),
        "answer": payload.get("answer"),
        "language_actual": payload.get("language"),
        "model": payload.get("model"),
        "validation_ok": payload.get("validation_ok"),
        "inference_performed": payload.get("inference_performed"),
        "evidence_status": payload.get("evidence_status"),
        "requirements_covered": payload.get("requirements_covered", []),
        "requirements_missing": payload.get("requirements_missing", []),
        "partial": payload.get("partial", False),
        "answered_requirements": payload.get("answered_requirements", []),
        "unanswered_requirements": payload.get("unanswered_requirements", []),
        "refusal_reason": payload.get("refusal_reason"),
        "understood_as": payload.get("understood_as"),
        "interrogation": interrogation,
        "interrogation_topic_ids": selected_ids,
        "interrogation_topic_hit": topic_hit,
        "ambiguous_interrogation_topics": ambiguous_topics,
        "retrieved": payload.get("retrieved", 0),
        "dropped_for_budget": payload.get("dropped_for_budget", 0),
        "seconds": payload.get("seconds"),
        "raw_preview": payload.get("raw_preview"),
        "sources": [{key: source.get(key) for key in
                     ("source_id", "document", "heading", "unit_id", "chunk_id")}
                    for source in payload.get("sources", []) if isinstance(source, dict)],
        "claims": claims,
        "expected_topic_ids": expected_ids,
        "expected_documents": expected_docs,
        "cited_documents": sorted(cited_documents),
        "citation_target_document_hit": target_doc_hit,
        "expected_classification": expected_class,
        "actual_classification": actual_class,
        "classification_match": (actual_class == expected_class
                                  if expected_class and actual_class else None),
        "expected_out_of_scope_refusal": is_oos_control,
        "out_of_scope_refusal_match": (status == "refused" if is_oos_control else None),
        "chat_calls": calls,
        "chat_call_counts": dict(Counter(call.get("kind", "other") for call in calls)),
    }


def summarize(report: dict) -> dict:
    cases = report.get("cases", [])
    status_counts = Counter(row.get("status") or "http_error" for row in cases)
    in_scope = [row for row in cases if row.get("expected_documents")]
    oos = [row for row in cases if row.get("expected_out_of_scope_refusal")]
    citations = [row.get("citation_target_document_hit") for row in in_scope]
    calls = [call for row in cases for call in row.get("chat_calls", [])]
    call_counts = Counter(call.get("kind", "other") for call in calls)
    return {
        "cases": len(cases),
        "http_200": sum(row.get("http_status") == 200 for row in cases),
        "status_counts": dict(status_counts),
        "validated_answers": sum(row.get("status") == "answered"
                                  and row.get("validation_ok") is True for row in cases),
        "citation_target_document_hits": sum(value is True for value in citations),
        "in_scope_cases_with_expected_documents": len(in_scope),
        "out_of_scope_refusal_matches": sum(row.get("out_of_scope_refusal_match") is True
                                             for row in oos),
        "out_of_scope_controls": len(oos),
        "interrogations": sum(bool(row.get("interrogation")) for row in cases),
        "logical_chat_calls": len(calls),
        "chat_calls_by_kind": dict(call_counts),
        "provider_or_http_errors": sum(
            row.get("status") == "error"
            or row.get("http_status", 0) >= 500
            or bool(row.get("http_error"))
            or any(call.get("error") for call in row.get("chat_calls", []))
            for row in cases),
    }


def _clip(value, limit: int) -> str:
    text = re.sub(r"[\r\n\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def annotations(report: dict) -> list[tuple[str, dict]]:
    """Small check annotations: split answers and evidence to avoid GH truncation."""
    summary_events = [("summary", {
        "model": report.get("model"),
        "max_logical_chat_calls": report.get("max_logical_chat_calls"),
        "free_price_verified": (report.get("free_price_verification") or {}).get("verified"),
        "profile": report.get("profile"),
        "summary": report.get("summary") or summarize(report),
        "setup_error": report.get("setup_error"),
    })]
    case_events, answer_events, evidence_events = [], [], []
    for row in report.get("cases", []):
        inter = row.get("interrogation") or {}
        case_events.append(("case", {
            "id": row.get("id"),
            "question": _clip(row.get("question"), 220),
            "http": row.get("http_status"),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "language": row.get("language_actual"),
            "validation_ok": row.get("validation_ok"),
            "evidence_status": row.get("evidence_status"),
            "retrieved": row.get("retrieved"),
            "dropped_for_budget": row.get("dropped_for_budget"),
            "seconds": row.get("seconds"),
            "expected_topic_ids": row.get("expected_topic_ids", []),
            "expected_documents": row.get("expected_documents", []),
            "cited_documents": row.get("cited_documents", []),
            "sources": [{"document": source.get("document"),
                         "heading": source.get("heading"),
                         "unit_id": source.get("unit_id")}
                        for source in (row.get("sources") or [])[:5]],
            "citation_target_document_hit": row.get("citation_target_document_hit"),
            "interrogation": ({
                "classification": inter.get("classification"),
                "topics": (inter.get("topics") or [])[:3],
                "topic_hit": row.get("interrogation_topic_hit"),
                "requirements": (inter.get("requirements") or [])[:4],
                "confidence": inter.get("confidence"),
                "understood_as": _clip(row.get("understood_as"), 220),
            } if inter else None),
            "requirements_covered": (row.get("requirements_covered") or [])[:8],
            "requirements_missing": (row.get("requirements_missing") or [])[:8],
            "answered_requirements": (row.get("answered_requirements") or [])[:8],
            "unanswered_requirements": (row.get("unanswered_requirements") or [])[:8],
            "partial": row.get("partial", False),
            "refusal_reason": row.get("refusal_reason"),
            "referral": (_clip(json.dumps(row.get("referral"), ensure_ascii=False), 360)
                         if row.get("referral") else None),
            "expected_out_of_scope_refusal": row.get("expected_out_of_scope_refusal"),
            "out_of_scope_refusal_match": row.get("out_of_scope_refusal_match"),
            "chat_call_counts": row.get("chat_call_counts", {}),
            "raw_preview": _clip(row.get("raw_preview"), 240) or None,
        }))
        answer = _clip(row.get("answer"), 650)
        answer_parts = [answer[index:index + 400] for index in range(0, len(answer), 400)] or ["—"]
        for part_number, text in enumerate(answer_parts, start=1):
            answer_events.append(("answer", {
                "id": row.get("id"), "part": part_number,
                "parts": len(answer_parts), "text": text,
            }))
        for claim_number, claim in enumerate((row.get("claims") or [])[:3], start=1):
            evidence = [{
                "source_id": item.get("source_id"),
                "document": item.get("document"),
                "heading": item.get("heading"),
                "quote": _clip(item.get("quote"), 180),
            } for item in (claim.get("evidence") or [])[:2]]
            evidence_events.append(("evidence", {
                "id": row.get("id"), "claim_number": claim_number,
                "claim": _clip(claim.get("text"), 220), "evidence": evidence,
            }))
    return [*summary_events, *case_events, *answer_events, *evidence_events]


def _emit_annotations(report: dict) -> None:
    for kind, value in annotations(report):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        print(f"ANNO| phase9-answer-{kind} | {encoded}")


def emit_existing_report(path: Path) -> int:
    """Re-publish an existing JSON diagnostic without ingestion or API access."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != "phase9-answer-live-results-1.0":
        raise ValueError("not a Phase-9 answer-probe report")
    report.setdefault("summary", summarize(report))
    _emit_annotations(report)
    return 0


def render_markdown(report: dict) -> str:
    summary = report.get("summary") or summarize(report)
    lines = [
        "# Phase 9 — six-case end-to-end answer probe",
        "",
        "**Scope:** live `POST /answer` through the service route; vector retrieval, sufficiency commitment, and Phase-8 interrogation are enabled. This is the current active profile only (no historical A/B), and the six diagnostic questions have no gold final-answer text. Citation membership is a structural guard, not a complete semantic truth score.",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Model: `{report.get('model')}`",
        f"- Logical chat-call cap: `{report.get('max_logical_chat_calls')}`",
        f"- Free-price verification: `{(report.get('free_price_verification') or {}).get('verified')}`",
        f"- Profile: `{json.dumps(report.get('profile', {}), ensure_ascii=False)}`",
        f"- Index chunks: `{report.get('index_chunks')}`",
        f"- Summary: `{json.dumps(summary, ensure_ascii=False)}`",
    ]
    if report.get("setup_error"):
        lines.extend(["", f"**Setup error:** `{report['setup_error']}`"])
    for row in report.get("cases", []):
        lines.extend([
            "", f"## {row.get('id')} ({row.get('language_expected')})", "",
            f"**Question:** {row.get('question')}", "",
            f"**Result:** HTTP {row.get('http_status')}; `{row.get('status')}` / `{row.get('reason')}`; validation `{row.get('validation_ok')}`; evidence `{row.get('evidence_status')}`; retrieved `{row.get('retrieved')}`; answer seconds `{row.get('seconds')}`.",
            "", "**Answer:**", "", f"> {str(row.get('answer') or '—').replace(chr(10), ' ')}",
        ])
        if row.get("understood_as"):
            inter = row.get("interrogation") or {}
            lines.extend([
                "", f"**Interrogation:** `{inter.get('classification')}`; topics "
                f"`{', '.join(inter.get('topics', [])) or 'none'}`; target hit "
                f"`{row.get('interrogation_topic_hit')}`; understood as: "
                f"{row.get('understood_as')}",
            ])
        if row.get("claims"):
            lines.extend(["", "**Claims and supporting quotes:**"])
            for claim in row["claims"]:
                lines.extend(["", f"- {claim.get('text')}"])
                for ev in claim.get("evidence", []):
                    location = " — ".join(filter(None, [ev.get("document"),
                                                         ev.get("heading"),
                                                         ev.get("unit_id")]))
                    lines.append(f"  - `{location}`: “{ev.get('quote')}”")
        if row.get("raw_preview"):
            lines.extend(["", f"**Rejected raw preview:** `{_clip(row['raw_preview'], 1000)}`"])
        if row.get("http_error"):
            lines.extend(["", f"**HTTP error:** `{row['http_error']}`"])
    return "\n".join(lines) + "\n"


def _emit(report: dict, output: Path) -> None:
    from artifacts import write_json

    output.parent.mkdir(parents=True, exist_ok=True)
    report["summary"] = summarize(report)
    report["markdown"] = render_markdown(report)
    write_json(output, report)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(report["markdown"], encoding="utf-8")
    print(f"results: {output.relative_to(RAGLAB) if output.is_relative_to(RAGLAB) else output}")
    print(f"summary: {markdown_path.relative_to(RAGLAB) if markdown_path.is_relative_to(RAGLAB) else markdown_path}")
    _emit_annotations(report)


def _validate_only() -> int:
    plan = load_plan()
    import profiles

    state = profiles.default_state()
    data_dirs = profiles.data_dirs(state)
    entries, by_id, _by_display = active_catalog(data_dirs)
    validate_active_targets(plan, by_id)
    print(f"answer probe plan OK: {len(plan['cases'])} unchanged cases; "
          f"{len(entries)} active topics; all expected IDs resolve; "
          "no service, provider, embedding, or model calls")
    return 0


def run(validate_only: bool = False, output: Path = DEFAULT_OUTPUT) -> int:
    if validate_only:
        return _validate_only()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        print("live POST /answer calls are restricted to GitHub Actions; "
              "use --validate-only locally", file=sys.stderr)
        return 2

    plan = load_plan()
    output = Path(output)
    base = {
        "schema_version": "phase9-answer-live-results-1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "active-profile end-to-end POST /answer; no historical A/B",
        "model": EXPECTED_PROFILE["answer"]["model"],
        "provider": EXPECTED_PROFILE["answer"]["provider"],
        "max_logical_chat_calls": MAX_LOGICAL_CHAT_CALLS,
        "free_price_verification": None,
        "profile": None,
        "index_chunks": None,
        "cases": [],
    }
    missing = [name for name in ("NVIDIA_API_KEY", "XKIRO_API_KEY")
               if not os.environ.get(name, "").strip()]
    if missing:
        base["setup_error"] = "required secret(s) missing: " + ", ".join(missing)
        _emit(base, output)
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="phase9-answer-probe-") as tmp:
            temp_root = Path(tmp)
            os.environ["RAGLAB_CACHE_DIR"] = str(temp_root / "cache")
            os.environ["RAGLAB_ALLOW_PROFILE_SWITCH"] = "0"

            import config as cfg
            import profiles
            import service

            profile = profiles.default_state()
            if profile["embedding"] != profiles.SUPPORTED_EMBEDDING:
                raise ValueError("default embedding profile is not the pinned NVIDIA model")
            if profile["answer"] != profiles.SUPPORTED_ANSWER:
                raise ValueError("default answer profile is not the pinned xKiro model")
            profile["chunking"] = {"mode": "restructure", "size": 220, "overlap": 40}
            profile["retrieval"] = {
                "top_k": int(cfg.ANSWER_TOP_K),
                "mode": "vector",
                "lang_filter": None,
                "neighbor_radius": int(cfg.ANSWER_NEIGHBOR_RADIUS),
            }
            profile["data_dirs"] = [str(path) for path in profiles.data_dirs(profile)]
            data_dirs = profiles.data_dirs(profile)
            _entries, by_id, by_display = active_catalog(data_dirs)
            validate_active_targets(plan, by_id)

            overrides = {
                "CHROMA_DIR": temp_root / "chroma",
                "RESULTS_DIR": temp_root / "results",
                "ANSWER_CACHE_PATH": temp_root / "answer_cache.json",
                "SUFFICIENCY_FIELDS_ENABLED": True,
                "ANSWER_SUFFICIENCY_COMMITMENT": True,
                "REPHRASE_INTERROGATION_ENABLED": True,
            }
            local = profiles.build_lab_config(profile)
            for key, value in overrides.items():
                setattr(local, key, value)

            # Build/check the pinned free answer generator BEFORE any NVIDIA
            # embeddings are paid for. No key value or model fallback is used.
            generator = profiles.build_generator(local)
            chat_budget = getattr(generator.client, "budget", None)
            if not isinstance(chat_budget, dict) or "limit" not in chat_budget:
                raise RuntimeError("pinned gateway client does not expose its logical-call budget")
            chat_budget["limit"] = MAX_LOGICAL_CHAT_CALLS
            sys.path.insert(0, str(PHASE9))
            import run_topic_map_live_probe as map_probe
            base["free_price_verification"] = map_probe.price_verification(
                generator.client, profile["answer"]["model"])
            counter = CountingChatClient(generator.client)
            generator.client = counter

            base["profile"] = {
                "embedding": profile["embedding"],
                "answer": profile["answer"],
                "chunking": profile["chunking"],
                "retrieval": profile["retrieval"],
                "sufficiency_commitment": True,
                "phase8_interrogation": True,
            }
            app = service.create_app(
                profile, generator=generator, allow_profile_switch=False,
                config_overrides=overrides,
                documents_dir=str(temp_root / "documents"),
            )
            from fastapi.testclient import TestClient

            with TestClient(app) as client:
                health = client.get("/health")
                if health.status_code != 200 or health.json().get("status") != "ok":
                    raise RuntimeError("service health preflight failed")
                ingest = client.post("/ingest?reset=true")
                if ingest.status_code != 200:
                    raise RuntimeError(f"service ingest rejected: HTTP {ingest.status_code}")
                deadline = time.monotonic() + 1800
                status = {}
                while time.monotonic() < deadline:
                    status_response = client.get("/ingest/status")
                    if status_response.status_code != 200:
                        raise RuntimeError("could not read service ingest status")
                    status = status_response.json()
                    if status.get("state") != "running":
                        break
                    time.sleep(0.5)
                else:
                    raise TimeoutError("service ingest exceeded 30 minutes")
                if status.get("state") != "done":
                    raise RuntimeError("service ingest failed: " + str(status.get("error") or status))
                index_info = client.get("/health").json().get("index", {})
                base["index_chunks"] = index_info.get("count")

                for case in plan["cases"]:
                    start = len(counter.events)
                    payload = {}
                    http_status = 0
                    request_error = None
                    try:
                        response = client.post("/answer", json={
                            "question": case["question"],
                            # No query_lang override: exercise service language detection.
                            "mode": "vector",
                            "k": int(profile["retrieval"]["top_k"]),
                            "include_excerpts": False,
                        })
                        http_status = response.status_code
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = {}
                        if http_status != 200:
                            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
                            request_error = json.dumps(detail, ensure_ascii=False)[:1000]
                    except Exception as exc:  # retain later cases after a request-level failure
                        from nvidia_api import safe_error
                        request_error = safe_error(exc)
                    row = _case_result(
                        case, payload if isinstance(payload, dict) else {},
                        http_status=http_status, error=request_error, by_id=by_id,
                        by_display=by_display, calls=counter.events[start:])
                    base["cases"].append(row)

        _emit(base, output)
        if base["summary"]["provider_or_http_errors"]:
            return 1
        return 0
    except Exception as exc:
        from nvidia_api import safe_error
        base["setup_error"] = safe_error(exc)
        _emit(base, output)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="validate the plan/catalog locally without provider or model calls")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--emit-report", type=Path,
                        help="emit compact GitHub annotations from an existing report; no API calls")
    args = parser.parse_args()
    if args.emit_report:
        return emit_existing_report(args.emit_report)
    return run(validate_only=args.validate_only, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
