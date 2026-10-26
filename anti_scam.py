r"""
Shared scam and fraud detection for job postings.

Single source of truth for the local pre-filter that runs before any LLM
call. Imported by both `scraper.py` (to reject listings at ingestion) and
`scorer.py` (to short-circuit Gemini when a description is obviously
fraudulent).

Design notes
------------
- One pattern list. Adding a new fraud indicator means editing one file.

- All patterns run against a whitespace-normalized, lowercased copy of the
  text. This makes `"training deposit"` match `"training\n deposit"` and
  `"training   deposit"` without needing multi-line regexes.

- Patterns use `\b` word boundaries where the phrase could be a substring
  of a legitimate word, and no boundaries where the phrase is inherently
  distinctive (`wire transfer`).

- The module does not import from `models.py`. It returns a plain
  `tuple[bool, str]` so it can be used from anywhere without a Pydantic
  dependency.

- "Suspiciously short description" and "anonymous company" checks live here
  because they share the same call site and the same purpose: reject before
  spending tokens.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Pattern list
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, human-readable reason).
# Patterns are matched against a lowercased, whitespace-collapsed string.
SCAM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Payment and financial traps.
    (re.compile(r"\bapplication fee\b"), "Requests an application or processing fee."),
    (re.compile(r"\bprocessing fee\b"), "Requests a processing fee."),
    (re.compile(r"\bregistration fee\b"), "Requests a registration fee."),
    (re.compile(r"\btraining deposit\b"), "Requests an upfront training deposit."),
    (re.compile(r"\btraining fee\b"), "Requests an upfront training fee."),
    (re.compile(r"\bsecurity deposit\b"), "Requests a security deposit."),
    (re.compile(r"\bwire transfer\b"), "Requests a wire transfer."),
    (re.compile(r"\bcrypto payment\b"), "Requests cryptocurrency payment."),
    (re.compile(r"\bbitcoin payment\b"), "Requests Bitcoin payment."),
    (re.compile(r"\bpay for your own\b"), "Asks the candidate to pay for their own equipment."),
    (re.compile(r"\bbuy (your own )?equipment\b"), "Asks the candidate to purchase equipment upfront."),
    (re.compile(r"\bprovide your bank\b"), "Requests bank account details before hiring."),
    (re.compile(r"\bssn required before\b"), "Requests sensitive identity data before hiring."),

    # Communication-channel red flags.
    (re.compile(r"\btelegram recruiter\b"), "Directs recruitment to Telegram."),
    (re.compile(r"\btelegram contact\b"), "Directs recruitment to Telegram."),
    (re.compile(r"\bwhatsapp recruiter\b"), "Directs recruitment to WhatsApp."),
    (re.compile(r"\bwhatsapp (only|contact)\b"), "Directs recruitment to WhatsApp."),
    (re.compile(r"\bcontact (us |me )?on telegram\b"), "Directs recruitment to Telegram."),
    (re.compile(r"\bmessage (us |me )?on whatsapp\b"), "Directs recruitment to WhatsApp."),

    # Unrealistic compensation traps.
    (re.compile(r"\bno experience required.{0,20}\$\d{2,}/?\s?(hr|hour)"),
     "Unrealistic pay for no-experience role."),
    (re.compile(r"\bearn \$?\d{3,}.{0,20}(daily|per day)"),
     "Unrealistic daily earnings claim."),
    (re.compile(r"\bguaranteed (weekly |daily )?income\b"),
     "Guaranteed-income language typical of fraud."),

    # Credential and identity harvesting.
    (re.compile(r"\bsend (a )?copy of your (passport|id|license)\b"),
     "Requests ID documents before hiring."),
    (re.compile(r"\bprovide (your )?credit card\b"),
     "Requests credit card details."),
]


# Company names that mean "we are not telling you who we are."
_ANONYMOUS_COMPANY_NAMES = frozenset({
    "unknown",
    "confidential",
    "n/a",
    "na",
    "none",
    "nan",
    "null",
    "-",
    "",
})


# Minimum viable description length. Anything shorter is almost always a
# teaser that funnels to an external form or a scam landing page.
_MIN_DESCRIPTION_CHARS = 100


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase and collapse all whitespace to single spaces."""
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _is_anonymous_company(company: Optional[str]) -> bool:
    """Return True if the company field is missing or a placeholder."""
    if company is None:
        return True
    return _normalize(company) in _ANONYMOUS_COMPANY_NAMES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_listing(
    title: str,
    company: Optional[str],
    description: Optional[str],
) -> tuple[bool, str]:
    """Evaluate a job posting for high-confidence fraud indicators.

    Parameters
    ----------
    title
        Job title. Currently unused, but reserved for future title-based
        heuristics (for example, "data entry" + "no experience").
    company
        Company name. `None` and placeholder values are treated as anonymous.
    description
        Full job description text. Whitespace is normalized internally.

    Returns
    -------
    tuple[bool, str]
        `(True, reason)` if the listing is flagged as fraudulent.
        `(False, "")` if the listing passes all local checks.
    """
    if _is_anonymous_company(company):
        return True, "Anonymous or missing company name."

    if not description:
        return True, "Empty job description."

    normalized = _normalize(description)

    if len(normalized) < _MIN_DESCRIPTION_CHARS:
        return True, (
            f"Description suspiciously short ({len(normalized)} chars, "
            f"minimum {_MIN_DESCRIPTION_CHARS})."
        )

    for pattern, reason in SCAM_PATTERNS:
        if pattern.search(normalized):
            return True, reason

    return False, ""


__all__ = ["SCAM_PATTERNS", "check_listing"]