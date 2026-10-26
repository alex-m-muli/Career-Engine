"""
Career Engine Browser Autofill Module.

Uses Playwright to attach to a persistent Chrome profile, navigate to a
job application URL, pre-fill standard fields from master_facts.json,
answer custom questions from question_bank.json, upload the tailored PDF,
and pause for human review before submission.

Relies on headed mode and a real user profile to bypass anti-bot layers
such as Cloudflare Turnstile and DataDome.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, NoReturn

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page, BrowserContext, Error as PlaywrightError
from pypdf import PdfReader

import db
from models import ApplicationRecord, MasterFacts

logger = logging.getLogger("Autofill")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUESTION_BANK_PATH = "question_bank.json"
DEFAULT_TIMEOUT_MS = 15000
HUMAN_REVIEW_DELAY_SEC = 0

# Fields we can fill without asking the user
STANDARD_FIELDS = {
    "name": ["first name", "last name", "full name", "your name"],
    "email": ["email", "e-mail", "email address"],
    "phone": ["phone", "mobile", "telephone", "contact number"],
    "location": ["location", "city", "current city", "where are you based"],
    "linkedin": ["linkedin", "linkedin profile", "linkedin url"],
    "github": ["github", "github profile", "portfolio", "website"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _load_question_bank() -> Dict[str, Any]:
    """Load question_bank.json and return a dict keyed by lowercase keyword."""
    if not os.path.exists(QUESTION_BANK_PATH):
        logger.warning(f"{QUESTION_BANK_PATH} not found. Custom questions will not be pre-filled.")
        return {}
    with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build a flat keyword -> answer map
    bank = {}
    for q in data.get("questions", []):
        for kw in q.get("keywords", []):
            bank[kw.lower()] = q["answer"]
    return bank


def _lookup_answer(question_text: str, bank: Dict[str, Any]) -> Optional[str]:
    """Find the best matching answer for a question using keyword overlap."""
    q_lower = question_text.lower()
    best_match = None
    best_score = 0
    for kw, answer in bank.items():
        if kw in q_lower:
            # Longer keyword matches are more specific
            score = len(kw)
            if score > best_score:
                best_score = score
                best_match = answer
    return best_match


def _chime() -> None:
    """Play a system chime to alert the user for manual review."""
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    except ImportError:
        # Non-Windows fallback: print a bell character
        print("\a", end="", flush=True)


# ---------------------------------------------------------------------------
# Core automation
# ---------------------------------------------------------------------------

def _fill_standard_fields(page: Page, facts: MasterFacts) -> None:
    """Attempt to fill common fields on the page."""
    profile = facts.profile
    field_map = {
        "first name": profile.name.split()[0],
        "last name": " ".join(profile.name.split()[1:]),
        "full name": profile.name,
        "email": profile.email,
        "phone": profile.phone,
        "location": profile.location,
        "linkedin": profile.linkedin,
        "github": profile.github,
    }

    for label, value in field_map.items():
        try:
            # Use get_by_label for robust accessibility-based location
            locator = page.get_by_label(label, exact=False).first
            if locator.is_visible(timeout=1000):
                locator.fill(value)
                logger.info(f"  [OK] Filled '{label}' -> {value}")
        except PlaywrightError:
            # Field not present, skip silently
            pass
        except Exception as e:
            logger.debug(f"  [SKIP] Could not fill '{label}': {e}")


def _fill_custom_questions(page: Page, bank: Dict[str, Any]) -> None:
    """Find all visible text inputs and textareas, and answer them from the bank."""
    # Generic locator for text inputs and textareas
    inputs = page.locator("input[type='text'], input[type='email'], textarea").all()
    for inp in inputs:
        try:
            if not inp.is_visible(timeout=500):
                continue
            # Try to find a nearby label
            label_text = ""
            # Method 1: aria-label
            aria = inp.get_attribute("aria-label")
            if aria:
                label_text = aria
            else:
                # Method 2: associated <label for="...">
                input_id = inp.get_attribute("id")
                if input_id:
                    label = page.locator(f"label[for='{input_id}']").first
                    if label.is_visible(timeout=500):
                        label_text = label.inner_text()
                # Method 3: parent container text
                if not label_text:
                    parent = inp.locator("xpath=..")
                    label_text = parent.inner_text()

            if not label_text:
                continue

            answer = _lookup_answer(label_text, bank)
            if answer:
                inp.fill(answer)
                logger.info(f"  [OK] Answered custom question '{label_text[:40]}...'")
        except Exception as e:
            logger.debug(f"  [SKIP] Could not process input: {e}")


def _upload_resume(page: Page, pdf_path: str) -> None:
    """Attempt to upload the PDF resume."""
    if not os.path.exists(pdf_path):
        logger.warning(f"  [WARN] Resume PDF not found: {pdf_path}")
        return

    try:
        # Common file input selectors
        file_input = page.locator("input[type='file']").first
        if file_input.is_visible(timeout=2000):
            file_input.set_input_files(pdf_path)
            logger.info(f"  [OK] Uploaded resume: {pdf_path}")
        else:
            logger.warning("  [WARN] No visible file input found for resume upload.")
    except Exception as e:
        logger.warning(f"  [WARN] Resume upload failed: {e}")


def autofill_application(
    job_url: str,
    pdf_path: str,
    facts: MasterFacts,
    user_data_dir: str,
) -> bool:
    """
    Launch a persistent Chrome context, navigate to the job URL, fill fields,
    and pause for human review.
    """
    bank = _load_question_bank()

    logger.info("=" * 70)
    logger.info("STARTING AUTOFILL SESSION")
    logger.info(f"Job URL: {job_url}")
    logger.info(f"Resume:  {pdf_path}")
    logger.info("=" * 70)

    try:
        with sync_playwright() as p:
            # Launch persistent context to use the real Chrome profile
            context: BrowserContext = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--start-maximized",
                ],
                viewport={"width": 1920, "height": 1080},
            )

            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(DEFAULT_TIMEOUT_MS)

            logger.info(f"[*] Navigating to {job_url}...")
            page.goto(job_url, wait_until="domcontentloaded")
            logger.info("[OK] Page loaded.")

            # Wait for form to render (dynamic ATS pages need extra time)
            page.wait_for_timeout(3000)

            # 1. Fill standard fields
            logger.info("[*] Filling standard fields...")
            _fill_standard_fields(page, facts)

            # 2. Fill custom questions
            logger.info("[*] Filling custom questions...")
            _fill_custom_questions(page, bank)

            # 3. Upload resume
            logger.info("[*] Uploading resume...")
            _upload_resume(page, pdf_path)

            # 4. Human review breakpoint
            logger.info("=" * 70)
            logger.info("HUMAN REVIEW REQUIRED")
            logger.info("=" * 70)
            logger.info("Please review the form in the browser window.")
            logger.info("When you are ready to submit, press ENTER in this terminal.")
            logger.info("To skip this application, type 's' and press ENTER.")
            logger.info("=" * 70)

            _chime()

            user_input = input("Press ENTER to proceed, or type 's' to skip: ").strip().lower()
            
            if user_input == "s":
                logger.info("[SKIP] User skipped submission.")
                context.close()
                return False

            # 5. The user has clicked submit manually (we do NOT auto-submit)
            logger.info("[OK] Application marked as submitted by user.")
            
            # Record in DB
            record = ApplicationRecord(
                job_url=job_url,
                company="Manual Review",  # Ideally parsed from page
                title="Manual Review",
                fit_score=0.0,
                pdf_path=pdf_path,
                status="applied",
            )
            db.record_application(record)
            logger.info("[OK] Application recorded in database.")

            context.close()
            return True

    except Exception as e:
        logger.error(f"Autofill failed: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> NoReturn:
    load_dotenv()
    _setup_logging()

    parser = argparse.ArgumentParser(description="Autofill a job application.")
    parser.add_argument("--url", required=True, help="Job application URL.")
    parser.add_argument("--pdf", required=True, help="Path to the tailored PDF resume.")
    parser.add_argument("--facts", default="master_facts.json", help="Path to master_facts.json.")
    parser.add_argument(
        "--profile-dir",
        default=os.getenv("CHROME_USER_DATA_DIR", ""),
        help="Path to Chrome user data directory.",
    )
    args = parser.parse_args()

    if not args.profile_dir:
        logger.error("CHROME_USER_DATA_DIR is not set in .env and no --profile-dir provided.")
        sys.exit(1)

    if not os.path.exists(args.facts):
        logger.error(f"Facts file not found: {args.facts}")
        sys.exit(1)

    with open(args.facts, "r", encoding="utf-8") as f:
        facts = MasterFacts.model_validate(json.load(f))

    success = autofill_application(
        job_url=args.url,
        pdf_path=args.pdf,
        facts=facts,
        user_data_dir=args.profile_dir,
    )

    if success:
        logger.info("Autofill session completed successfully.")
        sys.exit(0)
    else:
        logger.warning("Autofill session ended without submission.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nAutofill interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)