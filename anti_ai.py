r"""
Anti-AI text protocol and Typst sanitization.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# ---------------------------------------------------------------------------
# Typst escaping
# ---------------------------------------------------------------------------

_TYPST_ESCAPE_RE = re.compile(r'([\[\]#$*_`@<>~\\])')

# ---------------------------------------------------------------------------
# Pre-escape cleanup patterns
# ---------------------------------------------------------------------------

_AI_AS_AL_RE = re.compile(r"\b(Al|al)\b")
_LINE_START_HASH_RE = re.compile(r"^[#]+\s*", re.MULTILINE)
_TRAILING_BACKSLASH_RE = re.compile(r"\\+\s*$", re.MULTILINE)
_EM_DASH_RE = re.compile(r"[\u2014\u2013]")   # em-dash and en-dash
_WHITESPACE_COLLAPSE_RE = re.compile(r"[ \t]{2,}")
_DUP_COMMA_RE = re.compile(r",\s*,")

# Catches [b1], [b. 1], [ b2 ], etc.
_BULLET_REF_RE = re.compile(r"\[b\.?\s*\d+\]", re.IGNORECASE)


# Lowercase or mixed-case tech acronyms that must render uppercase.
_ACRONYM_FIXES: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bapis\b", re.I), "APIs"),
    (re.compile(r"\bapi\b", re.I), "API"),
    (re.compile(r"\baws\b", re.I), "AWS"),
    (re.compile(r"\bgcp\b", re.I), "GCP"),
    (re.compile(r"\bazure\b", re.I), "Azure"),
    (re.compile(r"\bsql\b", re.I), "SQL"),
    (re.compile(r"\bnosql\b", re.I), "NoSQL"),
    (re.compile(r"\bhtml\b", re.I), "HTML"),
    (re.compile(r"\bcss\b", re.I), "CSS"),
    (re.compile(r"\bjson\b", re.I), "JSON"),
    (re.compile(r"\bhttp\b", re.I), "HTTP"),
    (re.compile(r"\bhttps\b", re.I), "HTTPS"),
    (re.compile(r"\burl\b", re.I), "URL"),
    (re.compile(r"\burls\b", re.I), "URLs"),
    (re.compile(r"\bui\b", re.I), "UI"),
    (re.compile(r"\bux\b", re.I), "UX"),
    (re.compile(r"\bllms\b", re.I), "LLMs"),
    (re.compile(r"\bllm\b", re.I), "LLM"),
    (re.compile(r"\brag\b", re.I), "RAG"),
    (re.compile(r"\bmcp\b", re.I), "MCP"),
    (re.compile(r"\bgrpc\b", re.I), "gRPC"),
    (re.compile(r"\brest\b", re.I), "REST"),
    (re.compile(r"\bsdk\b", re.I), "SDK"),
    (re.compile(r"\bsdks\b", re.I), "SDKs"),
    (re.compile(r"\bci/cd\b", re.I), "CI/CD"),
    (re.compile(r"\bcicd\b", re.I), "CI/CD"),
    (re.compile(r"\bdevops\b", re.I), "DevOps"),
    (re.compile(r"\bsaas\b", re.I), "SaaS"),
    (re.compile(r"\bpaas\b", re.I), "PaaS"),
    (re.compile(r"\biaas\b", re.I), "IaaS"),
    (re.compile(r"\bmlops\b", re.I), "MLOps"),
    (re.compile(r"\bllmops\b", re.I), "LLMOps"),
]


# ---------------------------------------------------------------------------
# Banned vocabulary
# ---------------------------------------------------------------------------

_BANNED_VERBS: List[re.Pattern[str]] = [
    re.compile(r"\barchitected\b", re.I),
    re.compile(r"\barchitecting\b", re.I),
    re.compile(r"\bspearheaded\b", re.I),
    re.compile(r"\bspearheading\b", re.I),
    re.compile(r"\borchestrated\b", re.I),
    re.compile(r"\borchestrating\b", re.I),
    re.compile(r"\bchampioned\b", re.I),
    re.compile(r"\bchampioning\b", re.I),
    re.compile(r"\bleverage\b", re.I),
    re.compile(r"\bleverages\b", re.I),
    re.compile(r"\bleveraged\b", re.I),
    re.compile(r"\bleveraging\b", re.I),
    re.compile(r"\bdelved\b", re.I),
    re.compile(r"\bdelving\b", re.I),
    re.compile(r"\bsynergized\b", re.I),
    re.compile(r"\bsynergizing\b", re.I),
    re.compile(r"\bharmonized\b", re.I),
    re.compile(r"\bharmonizing\b", re.I),
    re.compile(r"\bcatalyzed\b", re.I),
    re.compile(r"\bcatalyzing\b", re.I),
    re.compile(r"\bstreamlined\b", re.I),
    re.compile(r"\bstreamlining\b", re.I),
    re.compile(r"\bengineered\b", re.I),
    re.compile(r"\bpivotal\b", re.I),
]

_BANNED_BUZZWORDS: List[re.Pattern[str]] = [
    re.compile(r"\bresults[- ]driven\b", re.I),
    re.compile(r"\bdetail[- ]oriented\b", re.I),
    re.compile(r"\bpassionate\b", re.I),
    re.compile(r"\bcutting[- ]edge\b", re.I),
    re.compile(r"\bforward[- ]thinking\b", re.I),
    re.compile(r"\bvisionary\b", re.I),
    re.compile(r"\bdynamic\b", re.I),
    re.compile(r"\bseamlessly\b", re.I),
    re.compile(r"\brobustly\b", re.I),
    re.compile(r"\btransformative\b", re.I),
    re.compile(r"\bsynergy\b", re.I),
    re.compile(r"\bparadigm\b", re.I),
    re.compile(r"\bbest[- ]in[- ]class\b", re.I),
    re.compile(r"\bgroundbreaking\b", re.I),
    re.compile(r"\brelentless\b", re.I),
    re.compile(r"\bstrategic[- ]maven\b", re.I),
    re.compile(r"\binnovative\b", re.I),
]

_RULE_OF_THREE_RE = re.compile(
    r"\b([A-Za-z][\w\-]{2,}),\s+([A-Za-z][\w\-]{2,}),\s+and\s+([A-Za-z][\w\-]{2,})\b"
)
_COLON_LEAD_IN_RE = re.compile(r":\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Entropy classification
# ---------------------------------------------------------------------------

_CONTEXT_FIRST_STARTERS = (
    "due to", "when ", "after ", "before ", "because ", "following ",
    "since ", "while ", "as ", "in response",
)

_DUTY_STARTERS = (
    "maintained", "updated", "supported", "handled", "managed",
    "documented", "reviewed", "assisted", "wrote",
)

_METRIC_MARKERS = (
    "%", "reduced", "cut ", "improved", "increased", "decreased",
    "saved", "halved", "doubled", "tripled",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_typst(text: object) -> str:
    """Prepare arbitrary text for injection into a Typst content block."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    s = _AI_AS_AL_RE.sub("AI", s)
    s = _LINE_START_HASH_RE.sub("", s)
    
    # Strip LLM bullet citations like [b1] or [b. 2]
    s = _BULLET_REF_RE.sub("", s)

    s = _EM_DASH_RE.sub(", ", s)
    s = _DUP_COMMA_RE.sub(",", s)
    s = _WHITESPACE_COLLAPSE_RE.sub(" ", s)
    s = _TRAILING_BACKSLASH_RE.sub("", s)

    for pattern, replacement in _ACRONYM_FIXES:
        s = pattern.sub(replacement, s)

    s = "\n".join(line.strip() for line in s.split("\n")).strip()

    # Escape Typst active markup. Must be last.
    s = _TYPST_ESCAPE_RE.sub(r"\\\1", s)

    return s


def validate_no_banned(text: str) -> List[str]:
    findings: List[str] = []

    for pattern in _BANNED_VERBS:
        for match in pattern.finditer(text):
            findings.append(f"banned verb: '{match.group(0)}'")

    for pattern in _BANNED_BUZZWORDS:
        for match in pattern.finditer(text):
            findings.append(f"banned buzzword: '{match.group(0)}'")

    for match in _RULE_OF_THREE_RE.finditer(text):
        findings.append(f"rule-of-three triplet: '{match.group(0)}'")

    if _COLON_LEAD_IN_RE.search(text):
        findings.append("colon-gated lead-in at end of line")

    if _EM_DASH_RE.search(text):
        findings.append("em-dash or en-dash present")

    seen = set()
    unique: List[str] = []
    for item in findings:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _classify_bullet(bullet: str, tool_keywords: Iterable[str]) -> str:
    b = bullet.strip().lower()

    for starter in _CONTEXT_FIRST_STARTERS:
        if b.startswith(starter):
            return "context"

    for tool in tool_keywords:
        if b.startswith(tool.lower()):
            return "tool"

    for marker in _METRIC_MARKERS:
        if marker in b:
            return "metric"

    for starter in _DUTY_STARTERS:
        if b.startswith(starter):
            return "duty"

    return "other"


def validate_entropy(
    bullets: Iterable[str],
    tool_keywords: Iterable[str] = (),
) -> List[str]:
    bullet_list = [b.strip() for b in bullets if b and b.strip()]
    warnings: List[str] = []

    if len(bullet_list) < 2:
        return warnings

    tool_list = list(tool_keywords)
    tags = [_classify_bullet(b, tool_list) for b in bullet_list]

    for i in range(1, len(tags)):
        if tags[i] == tags[i - 1] and tags[i] != "other":
            warnings.append(
                f"consecutive bullets share pattern '{tags[i]}' "
                f"(indices {i - 1} and {i})"
            )

    if len(bullet_list) >= 3:
        counts = [len(b.split()) for b in bullet_list]
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        std = variance ** 0.5
        if std < 3.0:
            warnings.append(
                f"low burstiness: bullet word-count stddev = {std:.2f} "
                f"(counts: {counts})"
            )

    return warnings


__all__ = [
    "sanitize_typst",
    "validate_no_banned",
    "validate_entropy",
]