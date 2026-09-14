"""Output PII scrubbing: known identifier patterns -> [LABEL] placeholders.

Applied at the RESPONSE boundary (after the citation gate: the gate must see
the raw verbatim text, the user must not). The patterns are deliberately
narrow for a banking corpus — emails, phone numbers, Tunisian RIB/IBAN and
CIN card numbers — everything else passes through untouched. This is a
display safety net, not a DLP product.

Order matters: emails first (no digit overlap), then RIB (long digit runs)
before phones (shorter digit runs), then contextual CIN.
"""

from __future__ import annotations

import re

SCRUB_LABELS = ("EMAIL", "PHONE", "RIB", "CIN")

_PATTERNS = (
    # email — unambiguous
    ("EMAIL", re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # Tunisian RIB: exactly 20 digits, optionally separated ("08 000 ... 12")
    # or as an IBAN ("TN59 2000 ..."). Standalone 20-digit runs are almost
    # never legitimate amounts or dates.
    ("RIB", re.compile(
        r"(?<!\d)(?:\d[ .\-]?){19}\d(?!\d)"
        r"|TN\d{2}(?:[ .\-]?\d){20}(?!\d)",
        re.IGNORECASE)),
    # phone: international (+NN or 00NN, then 8-12 digits) or Tunisian
    # national 0 + 8 digits
    ("PHONE", re.compile(
        r"(?<![\d+])(?:\+|00)\d{1,3}(?:[ .\-]?\d){7,11}(?!\d)"
        r"|(?<!\d)0(?:[ .\-]?\d){8}(?!\d)")),
    # CIN: 8 digits, only in CIN context (an 8-digit number alone could be an
    # amount, a year range, a count — scrubbing those would break answers);
    # the prefix words stay, only the number becomes [CIN]
    ("CIN", re.compile(
        r"(?i:\bCIN\b|\bcarte d'identit[ée](?: nationale)?\b)[^0-9]{0,24}?(\d{8})(?!\d)")),
)


def scrub_pii(text: str) -> str:
    """Replace known identifier patterns with their [LABEL] placeholder."""
    if not text:
        return text
    for label, pattern in _PATTERNS:
        if label == "CIN":
            # keep the prefix words, replace only the digit group (the first
            # occurrence in the match — the prefix is digit-free by construction)
            text = pattern.sub(
                lambda m: m.group(0).replace(m.group(1), "[CIN]", 1), text)
        else:
            text = pattern.sub(lambda _m, _label=label: f"[{_label}]", text)
    return text
