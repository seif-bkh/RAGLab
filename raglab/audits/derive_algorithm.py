#!/usr/bin/env python3
"""Derive the strict-constants algorithm from the LM repair log.

Owner-approved track (2026-09-28): the language model repairs the sentences
first and logs its notes; the file is then processed to DERIVE a strict
algorithm "إن أمكن، وإن تعذّر فلا بأس". This tool performs that derivation
on the current log (batch 1) and measures the result:

1. CONSTANTS — builds the frozen lexicon from the stored corpus itself
   (other-3 documents freq>=2  U  Loi freq>=2). No external resources.
2. DERIVED ALGORITHM (tier A) — the strict transform distilled from the
   batch-1 fix notes:
   v2 zone reconstruction (markers, digit islands, pre-split, terminal
   punctuation, tail stop)  +  boundary-fragment joins  +  adjacent
   unknown+fragment joins checked against the frozen lexicon  +  single-char
   prefix glue for {ل، ب، ك} (always prefixes in Arabic; {و، ف} are words
   and never glue).
3. MEASUREMENT — per entry: exact reproduction of the LM's «ب», positional
   token agreement, and the entry's fix-code class (pure-mechanical /
   lexicon-assisted / LM-required). Reported per entry and aggregated.

Usage: python audits/derive_algorithm.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

from loader import load_all  # noqa: E402
import restructure as rst  # noqa: E402
from llm_repair_check import (parse_entries, norm_ws,  # noqa: E402
                              agreement, is_exact)  # noqa: E402

PUNCT = "،.؛:()«»–—؟!…\"'٪%-|\u0640"
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
DIGIT_RE = re.compile(r"\d")
# Single Arabic letters that are ALWAYS prefixes (glue right); و and ف are
# conjunction words and never glue. Distilled from batch-1 notes (E12, E23).
PREFIX_LETTERS = set("لبك")
DERIVABLE_CODES = {"ح1", "ح2", "ح3", "ح6", "ح8"}
LEXICON_CODES = {"ح4"}
LM_CODES = {"ح5", "ح7", "ح9"}


def bare(t: str) -> str:
    return t.strip(PUNCT)


def build_lexicon() -> set:
    docs = load_all([HERE.parent.parent / "docs", HERE.parent / "data"])
    loi = [d for d in docs if d["name"].startswith("Loi")][0]
    others = [d for d in docs if not d["name"].startswith("Loi")]
    cnt: Counter = Counter()
    for d in others:
        for tok in d["text"].split():
            b = bare(tok)
            if len(b) >= 2:
                cnt[b] += 1
    loic: Counter = Counter()
    for tok in loi["text"].split():
        b = bare(tok)
        if len(b) >= 2:
            loic[b] += 1
    return ({w for w, c in cnt.items() if c >= 2}
            | {w for w, c in loic.items() if c >= 2})


def is_frag(tok: str, lex: set) -> bool:
    b = bare(tok)
    return bool(ARABIC_RE.search(b)) and 1 <= len(b) <= 4 and b not in lex


def is_unknown(tok: str, lex: set) -> bool:
    b = bare(tok)
    return (len(b) >= 2 and b not in lex
            and bool(ARABIC_RE.search(b)) and not DIGIT_RE.search(b))


def _try_join(a: str, b: str, lex: set):
    """Return the joined word if a+b or b+a is a longer lexicon word."""
    ba, bb = bare(a), bare(b)
    if not ba or not bb:
        return None
    for cand in (ba + bb, bb + ba):
        if cand in lex and len(cand) > max(len(ba), len(bb)):
            return cand
    return None


def _fusion_split(tok: str, lex: set):
    """K11: an unknown token that cleanly splits into two lexicon words is
    a fused pair (دفعتعتبر -> دفع تعتبر). Returns the pair or None."""
    b = bare(tok)
    if len(b) < 6 or b in lex:
        return None
    for cut in range(2, len(b) - 1):
        w1, w2 = b[:cut], b[cut:]
        if w1 in lex and w2 in lex:
            return w1, w2
    return None


SENT_PUNCT = {".", "؟", "؛", ":"}
# Ablation flags. Constants are ADMITTED on measured evidence only
# (batch 1, 32 entries): K7+K8 alone give exact 5/32 and 86.4% mean
# agreement; the split search (K9) and the join pass add nothing measurable
# (5/32, 86.4% both ways), and the ال-variant join / in-token fusion split
# measured NEGATIVE (85.6%). They stay in the code, off by default, as the
# experimental record of the derivation.
USE_K9 = False     # 2-split search with the frozen bigram table
USE_JOINS = False  # boundary/adjacent fragment joins + prefix glue


def derived_line(line: str, lex: set) -> str:
    """The DERIVED per-line transform (constants K1-K9 from the batch-1
    notes), built on the v2 zone reconstruction:

    K7  spaced-fused marker (الاول العنوان) -> swap to marker+ordinal
    K8  a mid-line standalone sentence-punct token is a segment anchor:
        each side is zone-reconstructed separately, segment order kept
        (the extractor reverses each side around the anchor)
    K9  multi-run lines: additionally try a 2-split of the stored segment
        (each part zone-reconstructed, order kept) and keep it only when
        the frozen bigram table scores it strictly higher
    """
    s = line.strip()
    if not s or s.startswith("|"):
        return s
    prefix = ""
    m = rst._SPACED_FUSED_MARKER_RE.match(s)
    if m:
        words = m.group(0).split()
        prefix = words[1] + " " + words[0]          # K7 swap
        s = s[m.end():].strip()
    else:
        m = (rst._MARKER_RE.match(s) or rst._FUSED_MARKER_RE.match(s))
        if m:
            prefix = m.group(0).rstrip()
            s = s[m.end():].strip()
            head = s.split()
            if head and head[0] in ("ـ", "—", "–", "-"):
                prefix += " " + head[0]
                s = " ".join(head[1:])
    if not s:
        return prefix

    # K8: split at standalone sentence-punct anchors
    toks = s.split()
    segments, cur = [], []
    for t in toks:
        if t in SENT_PUNCT:
            segments.append(("seg", cur))
            segments.append(("punct", [t]))
            cur = []
        else:
            cur.append(t)
    segments.append(("seg", cur))

    out_parts = []
    for kind, seg in segments:
        if kind == "punct":
            out_parts.append(seg[0])
            continue
        if not seg:
            continue
        seg_text = " ".join(seg)
        best = rst._reconstruct_line_visual_order(seg_text)
        best_score = rst._order_score(best.split())
        # K9: 2-split search with the frozen bigram/unigram table
        if USE_K9:
            stoks = seg
            for cut in range(2, len(stoks) - 1):
                a = rst._reconstruct_line_visual_order(" ".join(stoks[:cut]))
                b = rst._reconstruct_line_visual_order(" ".join(stoks[cut:]))
                cand = (a + " " + b).strip()
                # same word multiset as the plain reconstruction -> the score
                # difference isolates the bigram (order) evidence
                if cand.replace(" ", "") != best.replace(" ", ""):
                    continue
                if rst._order_score(cand.split()) > best_score + 1:
                    best, best_score = cand, rst._order_score(cand.split())
        out_parts.append(best)
    body = " ".join(out_parts)
    return (prefix + " " + body).strip() if prefix else body


def strict_repair(q_lines: list, lex: set) -> str:
    """The derived strict algorithm over one entry's stored lines: the
    per-line transform above + the join pass (boundary fragments and
    {ل ب ك} prefix glue checked against the frozen lexicon)."""
    recs = [derived_line(l, lex) for l in q_lines]
    toks = []
    for l in recs:
        parts = l.split()
        if USE_JOINS and len(parts) >= 3:
            # same-line first/last boundary join (البنك...ية / اعتي...ادية)
            joined = _try_join(parts[0], parts[-1], lex)
            if joined and (is_frag(parts[0], lex) or is_frag(parts[-1], lex)
                           or is_unknown(parts[0], lex)
                           or is_unknown(parts[-1], lex)):
                if len(bare(parts[-1])) > len(bare(parts[0])):
                    parts = parts[:-1] + [joined]
                else:
                    parts = [joined] + parts[1:]
        toks.extend(parts)

    for _ in range(4) if USE_JOINS else [0]:
        out, i, changed = [], 0, False
        while i < len(toks):
            t = toks[i]
            bt = bare(t)
            if i + 1 < len(toks):
                nx = toks[i + 1]
                bn = bare(nx)
                # single-char prefix glue {ل ب ك}
                if (len(bt) == 1 and bt in PREFIX_LETTERS and bn
                        and not is_frag(nx, lex)):
                    out.append(bt + nx)
                    i += 2
                    changed = True
                    continue
                # adjacent join (either order) via the frozen lexicon
                if (is_frag(t, lex) or is_unknown(t, lex)
                        or is_frag(nx, lex) or is_unknown(nx, lex)):
                    joined = _try_join(t, nx, lex)
                    if joined:
                        out.append(joined)
                        i += 2
                        changed = True
                        continue
            out.append(t)
            i += 1
        toks = out
        if not changed:
            break
    return norm_ws(" ".join(toks))


def v2_only(q_lines: list) -> str:
    return norm_ws(" ".join(
        rst._reconstruct_line_visual_order(l) for l in q_lines))


def entry_codes(m_lines: list) -> set:
    codes = set()
    for m in m_lines:
        codes.update(re.findall(r"ح\d", m))
        if "ش⚠" in m:
            codes.add("ش⚠")
    return codes


def classify(codes: set) -> str:
    if codes & LM_CODES or "ش⚠" in codes:
        return "LM"
    if codes & LEXICON_CODES:
        return "LEX"
    return "MECH"


exact = is_exact  # shared aligned metric (see llm_repair_check)


def main():
    lex = build_lexicon()
    md = (HERE / "Loi_2016-48_llm_repair.md").read_text(encoding="utf-8")
    entries = parse_entries(md)
    print(f"[derive] lexicon constants: {len(lex)} words")
    print(f"[derive] entries: {len(entries)}")
    global D_exact, D_agreement
    D_exact, D_agreement = exact, agreement

    rows = []
    n_exact_v2 = n_exact_strict = 0
    sum_v2 = sum_strict = 0.0
    tiers = Counter()
    for e in entries:
        lm = norm_ws("\n".join(e["b"]))
        a_v2 = v2_only(e["q"])
        a_strict = strict_repair(e["q"], lex)
        ex_v2 = exact(a_v2, lm)
        ex_strict = exact(a_strict, lm)
        r_v2, r_strict = agreement(a_v2, lm), agreement(a_strict, lm)
        n_exact_v2 += ex_v2
        n_exact_strict += ex_strict
        sum_v2 += r_v2
        sum_strict += r_strict
        codes = entry_codes(e["m"])
        tier = classify(codes)
        tiers[tier] += 1
        rows.append((e["id"].strip(), tier, ",".join(sorted(codes)) or "-",
                     f"{r_v2:.0%}", f"{r_strict:.0%}",
                     "نعم" if ex_strict else "لا"))

    print(f"[derive] {'ID':<7} {'فئة':<5} {'شفرة الملاحظات':<18} "
          f"{'v2':>4} {'مستنتجة':>8} {'تطابق':>5}")
    for r in rows:
        print(f"[derive] {r[0]:<7} {r[1]:<5} {r[2]:<18} "
              f"{r[3]:>4} {r[4]:>8} {r[5]:>5}")

    n = len(entries)
    print(f"[derive] SUMMARY v2-only: exact {n_exact_v2}/{n}, "
          f"mean agreement {sum_v2 / n:.1%}")
    print(f"[derive] SUMMARY derived (admitted constants K7+K8): "
          f"exact {n_exact_strict}/{n}, mean agreement {sum_strict / n:.1%}")
    print(f"[derive] entry classes: MECH(آلية بالكامل)={tiers['MECH']} "
          f"LEX(تحتاج معجمًا مجمدًا)={tiers['LEX']} "
          f"LM(تتطلب النموذج اللغوي)={tiers['LM']}")

    # ablation: the rejected constants, measured
    global USE_K9, USE_JOINS
    for label, k9, jo in (("K9 split search", True, False),
                          ("join pass", False, True),
                          ("K9+joins", True, True)):
        USE_K9, USE_JOINS = k9, jo
        ex = sum(1 for e in entries
                 if D_exact(strict_repair(e["q"], lex),
                            norm_ws("\n".join(e["b"]))))
        ag = sum(D_agreement(strict_repair(e["q"], lex),
                             norm_ws("\n".join(e["b"]))) for e in entries)
        print(f"[derive] ablation +{label:<14}: exact {ex}/{n}, "
              f"mean agreement {ag / n:.1%}")
    USE_K9, USE_JOINS = False, False


if __name__ == "__main__":
    main()
