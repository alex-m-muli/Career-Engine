"""
Career Engine CLI Orchestrator.

Ties together the scraper, scorer, database, and CV builder into a single
pipeline. Fetches jobs, filters out scams and low-fit roles, generates
tailored PDFs for high-fit roles, and records everything in SQLite.

Usage:
    python main.py --search "AI Engineer" --location "Dubai"
    python main.py --search "Full Stack" --location "Remote" --threshold 8.5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List, NoReturn

from dotenv import load_dotenv

import db
from cv_builder import generate_tailored_cv
from llm import LLMClient
from models import ApplicationRecord, BridgeConfig, MasterFacts
from scorer import analyze_job_fit
from scraper import fetch_recent_jobs

logger = logging.getLogger("CareerEngine")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("career_engine.log", encoding="utf-8"),
        ],
    )


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Career Engine: Scrape, Score, Tailor, and Apply."
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        default="Full Stack Engineer",
        help="Job title query (default: 'Full Stack Engineer').",
    )
    parser.add_argument(
        "--location", "-l",
        type=str,
        default="Dubai",
        help="Location constraint (default: 'Dubai').",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="United Arab Emirates",
        help="Indeed country identifier (default: 'United Arab Emirates').",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=8.0,
        help="Minimum fit score to generate a CV (default: 8.0).",
    )
    parser.add_argument(
        "--results", "-r",
        type=int,
        default=10,
        help="Results wanted per job board (default: 10).",
    )
    parser.add_argument(
        "--hours-old",
        type=int,
        default=72,
        help="Maximum job posting age in hours (default: 72).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without generating PDFs or writing to the DB.",
    )
    return parser.parse_args()


def main() -> NoReturn:
    load_dotenv()
    _setup_logging()
    args = _parse_args()

    logger.info("=" * 70)
    logger.info("CAREER ENGINE PIPELINE START")
    logger.info(f"Search: '{args.search}' | Location: '{args.location}'")
    logger.info(f"Threshold: {args.threshold}/10 | Dry Run: {args.dry_run}")
    logger.info("=" * 70)

    # 1. Validate environment
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set. Please check your .env file.")
        sys.exit(1)

    # 2. Load core data
    try:
        facts_data = _load_json("master_facts.json")
        facts = MasterFacts.model_validate(facts_data)
        
        bridge_data = _load_json("bridge_roles.json")
        bridge_config = BridgeConfig.model_validate(bridge_data)
    except Exception as e:
        logger.error(f"Failed to load configuration files: {e}")
        sys.exit(1)

    # 3. Initialize infrastructure
    if not args.dry_run:
        db.init_db()
        
    try:
        llm_client = LLMClient()
    except Exception as e:
        logger.error(f"Failed to initialize LLM Client: {e}")
        sys.exit(1)

    # 4. Scrape jobs
    logger.info("-" * 70)
    logger.info("STEP 1: Scraping Jobs")
    logger.info("-" * 70)
    
    jobs = fetch_recent_jobs(
        search_term=args.search,
        location=args.location,
        country_indeed=args.country,
        results_wanted=args.results,
        hours_old=args.hours_old,
    )

    if not jobs:
        logger.warning("No jobs returned from scraper. Exiting.")
        sys.exit(0)

    logger.info(f"Scraped {len(jobs)} unique, non-scam jobs.")

    # 5. Process each job
    logger.info("-" * 70)
    logger.info("STEP 2: Scoring & Generating CVs")
    logger.info("-" * 70)

    applied_count = 0
    skipped_scam = 0
    skipped_low_fit = 0
    skipped_duplicate = 0
    failed_count = 0

    for i, job in enumerate(jobs, 1):
        logger.info(f"\n[{i}/{len(jobs)}] {job.title} at {job.company} ({job.site})")

        # A. Deduplication check
        if not args.dry_run and db.is_applied(job.job_url):
            logger.info(f"  -> [SKIP] Already applied to this job.")
            skipped_duplicate += 1
            continue

        # B. Score the job
        analysis = analyze_job_fit(job.description, facts, llm_client)
        
        if analysis.status == "error":
            logger.error(f"  -> [FAIL] Scoring failed: {analysis.scam_reason}")
            failed_count += 1
            continue
            
        if analysis.status == "scam" or analysis.is_scam_risk:
            logger.warning(f"  -> [SKIP] Scam risk detected: {analysis.scam_reason}")
            skipped_scam += 1
            continue

        logger.info(f"  -> Fit Score: {analysis.fit_score}/10.0")

        if analysis.fit_score < args.threshold:
            logger.info(f"  -> [SKIP] Below threshold ({args.threshold}).")
            skipped_low_fit += 1
            continue

        # C. Generate CV
        safe_company = "".join(c for c in job.company if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        safe_title = "".join(c for c in job.title if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        output_filename = f"outputs/{safe_company}_{safe_title}.pdf"

        logger.info(f"  -> [ACTION] Generating CV -> {output_filename}")

        if args.dry_run:
            logger.info(f"  -> [DRY RUN] Would generate CV here.")
            applied_count += 1
            continue

        success = generate_tailored_cv(
            job_description=job.description,
            master_facts=facts,
            bridge_config=bridge_config,
            llm_client=llm_client,
            match_analysis=analysis,
            output_pdf_path=output_filename,
        )

        if success:
            record = ApplicationRecord(
                job_url=job.job_url,
                company=job.company,
                title=job.title,
                fit_score=analysis.fit_score,
                pdf_path=output_filename,
                status="pending",
            )
            db.record_application(record)
            applied_count += 1
            logger.info(f"  -> [SUCCESS] CV generated and recorded.")
        else:
            logger.error(f"  -> [FAIL] CV generation failed.")
            failed_count += 1

        # D. Pacing between jobs to avoid rate limits
        if i < len(jobs):
            logger.debug("Sleeping 5s between jobs...")
            time.sleep(5)

    # 6. Summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total Jobs Analyzed:   {len(jobs)}")
    logger.info(f"CVs Generated/Ready:   {applied_count}")
    logger.info(f"Skipped (Duplicates):  {skipped_duplicate}")
    logger.info(f"Skipped (Scams):       {skipped_scam}")
    logger.info(f"Skipped (Low Fit):     {skipped_low_fit}")
    logger.info(f"Failures:              {failed_count}")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("[DRY RUN] No PDFs were written and no database records were created.")

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)