"""
Comprehensive verification for the CV Builder pipeline.

Runs the full 3-pass DIS pipeline end-to-end, compiles the PDF, then
validates the generated output against the Anti-AI protocol, the expected
section structure, and the absence of LLM formatting artifacts.

Fails with a nonzero exit code if any check is violated.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import NoReturn

from dotenv import load_dotenv
from pypdf import PdfReader

from anti_ai import sanitize_typst, validate_no_banned
from cv_builder import generate_tailored_cv
from llm import LLMClient
from models import BridgeConfig, MasterFacts, MatchAnalysis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("VerifyCVBuilder")

OUTPUT_PDF = "_test_output_resume.pdf"
MAX_PIPELINE_SECONDS = 180

REQUIRED_SECTIONS = [
    "PROFESSIONAL SUMMARY",
    "TECHNICAL SKILLS",
    "PROFESSIONAL EXPERIENCE",
    "EDUCATION",
    "CERTIFICATIONS",
]

REQUIRED_ROLES = [
    "TELUS Digital",
    "Independent Software Consultant",
    "Elimu Digital TV",
    "Self-Employed",
]

SKILL_CATEGORY_HINTS = [
    "Languages",
    "Backend",
    "Frontend",
    "Cloud",
    "DevOps",
    "AI",
    "Data",
    "Databases",
]

# Matches sanitizer's regex exactly. Catches [b1], [b. 1], [b 2], etc.
_LEAKED_BULLET_REF = re.compile(r"\[b\.?\s*\d+\]", re.IGNORECASE)
_LEAKED_SUMMARY_PLACEHOLDER = re.compile(r"\[summary\]", re.IGNORECASE)
_LEAKED_SKILLS_PLACEHOLDER = re.compile(r"\[skills", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fail(message: str) -> NoReturn:
    """Log a failure and exit with a nonzero code. NoReturn tells Pylance flow halts."""
    logger.error(f"[FAIL] {message}")
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    """Log pass or fail for a single assertion."""
    if condition:
        logger.info(f"[OK]   {message}")
    else:
        fail(message)


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF using pypdf."""
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 70)
    logger.info("CV Builder Full Pipeline Verification")
    logger.info("=" * 70)

    # 1. Prerequisites
    if os.path.exists(OUTPUT_PDF):
        os.remove(OUTPUT_PDF)
    logger.info(f"[*] Removed stale {OUTPUT_PDF} (if present).")

    if not os.getenv("GEMINI_API_KEY"):
        fail("GEMINI_API_KEY is not set in environment.")
    logger.info("[OK]   GEMINI_API_KEY is set.")

    # 2. Load inputs
    facts: MasterFacts
    try:
        with open("master_facts.json", "r", encoding="utf-8") as f:
            facts = MasterFacts.model_validate(json.load(f))
        logger.info(f"[OK]   Loaded master_facts.json ({facts.profile.name})")
    except Exception as e:
        fail(f"Failed to load master_facts.json: {e}")

    bridge_config: BridgeConfig
    try:
        with open("bridge_roles.json", "r", encoding="utf-8") as f:
            bridge_config = BridgeConfig.model_validate(json.load(f))
        logger.info(
            f"[OK]   Loaded bridge_roles.json ({len(bridge_config.project_pool)} projects)"
        )
    except Exception as e:
        fail(f"Failed to load bridge_roles.json: {e}")

    if not os.path.exists("cv_template.typ"):
        fail("cv_template.typ is missing from project root.")
    logger.info("[OK]   cv_template.typ exists.")

    # 3. Initialize LLM client
    client: LLMClient
    try:
        client = LLMClient()
        logger.info("[OK]   LLMClient initialized.")
    except Exception as e:
        fail(f"Failed to initialize LLMClient: {e}")

    # 4. Build a realistic JD
    job_description = (
        "We are hiring a Senior Full-Stack and AI Systems Engineer to join our "
        "platform team in Dubai. You will build production web applications and "
        "distributed microservices using C# .NET 8, Python FastAPI, TypeScript, "
        "and Next.js. Experience with Kubernetes, PostgreSQL, and RAG workflows "
        "using LangChain and vector databases is required. Familiarity with AWS "
        "and Azure is preferred. The role involves working closely with ML "
        "engineers to integrate LLM-powered features into customer-facing apps."
    )
    logger.info(f"[OK]   JD length: {len(job_description)} chars.")

    # 5. Build a mock MatchAnalysis to exercise the seed path
    match_analysis = MatchAnalysis(
        status="ok",
        is_scam_risk=False,
        scam_reason=None,
        fit_score=9.2,
        missing_critical_skills=[],
        skill_bridge_statements=[
            "C# .NET background directly maps to the JD's .NET 8 requirement.",
            "Existing LangChain and vector DB work covers the RAG requirement.",
        ],
        selected_bullet_ids=["b1", "b2", "b3"],
        tailored_summary=(
            "Lead Full-Stack and AI Systems Engineer with deep experience in "
            "Python, TypeScript, and .NET. Focused on shipping production RAG "
            "pipelines and scalable backend services."
        ),
    )
    logger.info("[OK]   Constructed mock MatchAnalysis for seed path.")

    # 6. Run the pipeline (timed)
    logger.info("-" * 70)
    logger.info("Running full 3-pass DIS pipeline (Groq primary, Gemini fallback)...")
    logger.info("-" * 70)

    start_time = time.time()
    success: bool
    try:
        success = generate_tailored_cv(
            job_description=job_description,
            master_facts=facts,
            bridge_config=bridge_config,
            llm_client=client,
            match_analysis=match_analysis,
            output_pdf_path=OUTPUT_PDF,
        )
    except Exception as e:
        fail(f"generate_tailored_cv raised an exception: {e}")
    elapsed = time.time() - start_time

    check(success, "Pipeline returned True.")
    check(
        elapsed <= MAX_PIPELINE_SECONDS,
        f"Pipeline completed in {elapsed:.1f}s (limit {MAX_PIPELINE_SECONDS}s).",
    )
    check(os.path.exists(OUTPUT_PDF), f"{OUTPUT_PDF} was written to disk.")

    pdf_size = os.path.getsize(OUTPUT_PDF)
    check(pdf_size > 10_000, f"PDF size is reasonable ({pdf_size} bytes).")

    # 7. Extract PDF text
    text: str
    try:
        text = extract_pdf_text(OUTPUT_PDF)
        logger.info(f"[OK]   Extracted {len(text)} characters from PDF.")
    except Exception as e:
        fail(f"Failed to extract PDF text: {e}")

    text_upper = text.upper()
    text_lower = text.lower()

    # 8. Required sections present
    for section in REQUIRED_SECTIONS:
        check(section in text_upper, f"Section '{section}' appears in PDF.")

    # 9. Required roles present (case-insensitive)
    for role in REQUIRED_ROLES:
        check(role.lower() in text_lower, f"Role '{role}' appears in PDF.")

    # 10. Candidate contact info present
    check(
        facts.profile.name.lower() in text_lower,
        "Candidate name appears in PDF.",
    )
    check(
        facts.profile.email.lower() in text_lower,
        "Candidate email appears in PDF.",
    )
    check(facts.profile.phone in text, "Candidate phone appears in PDF.")

    # 11. Forbidden artifact checks
    if _LEAKED_SUMMARY_PLACEHOLDER.search(text):
        fail("Summary placeholder '[summary]' leaked into PDF.")
    logger.info("[OK]   No '[summary]' placeholder leaked.")

    if _LEAKED_SKILLS_PLACEHOLDER.search(text):
        fail("Skills placeholder '[skills...' leaked into PDF.")
    logger.info("[OK]   No '[skills...' placeholder leaked.")

    bullet_leak = _LEAKED_BULLET_REF.search(text)
    if bullet_leak:
        fail(f"Bullet reference leaked into PDF: '{bullet_leak.group(0)}'")
    logger.info("[OK]   No bullet reference citations leaked into PDF.")

    check("{" not in text, "No raw JSON opening brace leaked.")
    check("}" not in text, "No raw JSON closing brace leaked.")
    check("\\n" not in text, "No literal backslash-n leaked.")

    # 12. Anti-AI protocol
    banned_findings = validate_no_banned(text)
    if banned_findings:
        for finding in banned_findings[:10]:
            logger.error(f"       - {finding}")
        fail(f"PDF contains {len(banned_findings)} anti-AI violations.")
    logger.info("[OK]   PDF passes the Anti-AI protocol (zero violations).")

    # 13. Certifications populated
    cert_count = sum(
        1
        for c in facts.profile.certifications
        if c.split(" - ")[0].split(" (")[0].strip() in text
    )
    check(
        cert_count >= 2,
        f"At least 2 certifications rendered (found {cert_count}).",
    )

    # 14. Timeline dates present
    check("Mar 2024" in text or "MAR 2024" in text_upper, "TELUS start date present.")
    check("Sep 2022" in text or "SEP 2022" in text_upper, "Consulting start date present.")
    check("May 2022" in text or "MAY 2022" in text_upper, "Elimu start date present.")
    check("Jan 2020" in text or "JAN 2020" in text_upper, "Freelance start date present.")

    # 15. Skills section has category formatting
    category_hits = sum(1 for cat in SKILL_CATEGORY_HINTS if cat in text)
    check(
        category_hits >= 1,
        f"Skills section has at least one category label ({category_hits} hits).",
    )

    # 16. Sanitizer self-check
    sanitizer_cases = [
        "Built a system [b1].",
        "Optimized queries [b. 2] for speed.",
        "Scaled infra [b3] and [b4].",
    ]
    for case in sanitizer_cases:
        sanitized = sanitize_typst(case)
        check(
            not _LEAKED_BULLET_REF.search(sanitized),
            f"sanitize_typst strips bullet refs from: '{case[:40]}'",
        )

    # Summary
    logger.info("=" * 70)
    logger.info("ALL CHECKS PASSED")
    logger.info(f"Output PDF:        {os.path.abspath(OUTPUT_PDF)}")
    logger.info(f"Pipeline elapsed:  {elapsed:.1f}s")
    logger.info(f"PDF size:          {pdf_size} bytes")
    logger.info(f"Extracted chars:   {len(text)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()