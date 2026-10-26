"""
Scraper Module for Career Engine.

Utilizes `python-jobspy` to aggregate live job postings across LinkedIn, Indeed,
Glassdoor, ZipRecruiter, Bayt, and Google Jobs into a standardized internal data format.
Includes robust error handling, Gulf region support, and pre-LLM anti-scam filtering.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, List, Optional

import pandas as pd
from jobspy import scrape_jobs

from anti_scam import check_listing
from models import JobPosting

logger = logging.getLogger("JobScraper")


def _sanitize_text(text: Any) -> str:
    """
    Cleans raw job description text by removing excessive whitespace, 
    non-printable characters, and malformed HTML.
    """
    if text is None or pd.isna(text):
        return ""
    text_str = str(text)
    # Remove HTML tags
    text_str = re.sub(r"<[^>]+>", " ", text_str)
    # Collapse whitespace
    text_str = re.sub(r"\s+", " ", text_str)
    return text_str.strip()


def fetch_recent_jobs(
    search_term: str = "Full Stack Engineer",
    location: str = "Remote",
    site_names: Optional[List[str]] = None,
    results_wanted: int = 10,
    hours_old: int = 72,
    country_indeed: str = "USA",
) -> List[JobPosting]:
    """
    Scrapes recent job postings across specified platforms with regional and anti-scam filtering.

    Args:
        search_term (str): Job title query (e.g., 'Full Stack Engineer').
        location (str): Location constraint (e.g., 'Dubai', 'Riyadh', 'Remote').
        site_names (Optional[List[str]]): Job boards to scrape. Supported: linkedin, indeed, 
            glassdoor, zip_recruiter, bayt, google.
        results_wanted (int): Max results per board.
        hours_old (int): Maximum age of postings in hours.
        country_indeed (str): Country identifier for Indeed/Glassdoor (e.g., 'United Arab Emirates').

    Returns:
        List[JobPosting]: Validated, anti-scam filtered job records.
    """
    if site_names is None:
        site_names = ["linkedin", "indeed", "zip_recruiter", "glassdoor"]

    logger.info(
        f"Initiating job scrape | Term: '{search_term}' | Location: '{location}' | Country Indeed: '{country_indeed}'"
    )

    try:
        jobs_df: pd.DataFrame = scrape_jobs(
            site_name=site_names,
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed,
            is_remote=(location == "Remote"),
            fetch_description=True,
            verbose=0,
        )

        if jobs_df is None or jobs_df.empty:
            logger.warning("No job listings returned for the specified parameters.")
            return []

        standardized_jobs: List[JobPosting] = []
        seen_jobs: set[tuple[str, str, str]] = set()

        for _, row in jobs_df.iterrows():
            # 1. Safe field extraction (handling pandas NaN)
            raw_url = str(row["job_url"]).strip() if pd.notna(row.get("job_url")) else ""
            raw_title = str(row["title"]).strip() if pd.notna(row.get("title")) else ""
            raw_company = str(row["company"]).strip() if pd.notna(row.get("company")) else "Unknown"
            raw_location = str(row["location"]).strip() if pd.notna(row.get("location")) else location
            
            if not raw_url or not raw_title:
                continue

            # 2. Deduplication check
            dedup_key = (raw_company.lower(), raw_title.lower(), raw_location.lower())
            if dedup_key in seen_jobs:
                logger.debug(f"Skipping duplicate listing: {raw_title} at {raw_company}")
                continue
            seen_jobs.add(dedup_key)

            # 3. Sanitize description
            sanitized_desc = _sanitize_text(row.get("description"))

            # 4. Anti-Scam Inspection (delegated to anti_scam.py)
            is_scam, scam_reason = check_listing(raw_title, raw_company, sanitized_desc)
            if is_scam:
                logger.warning(f"[REJECTED SCAM] {raw_title} at {raw_company} | Reason: {scam_reason}")
                continue

            # 5. Safe boolean extraction
            is_remote = bool(row["is_remote"]) if pd.notna(row.get("is_remote")) else False

            # 6. Deterministic ID generation
            job_id = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()[:16]

            job_entry = JobPosting(
                id=job_id,
                site=str(row.get("site", "unknown")),
                title=raw_title,
                company=raw_company,
                location=raw_location,
                job_url=raw_url,
                description=sanitized_desc,
                date_posted=str(row.get("date_posted", "")),
                is_remote=is_remote,
            )

            standardized_jobs.append(job_entry)

        logger.info(f"Successfully scraped and verified {len(standardized_jobs)} non-scam job postings.")
        return standardized_jobs

    except Exception as e:
        logger.error(f"Failed to execute job scrape: {str(e)}", exc_info=True)
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("[*] Testing Multi-Title Scrape (UAE / Dubai)...")
    
    # Define the titles you want to search for
    target_titles = ["Full Stack Engineer", "AI Engineer", "Python Developer"]
    
    # Define the boards (Omitting ZipRecruiter and Google as they are mostly US-centric)
    target_boards = ["indeed", "linkedin", "bayt"]
    
    all_results: List[JobPosting] = []
    
    for title in target_titles:
        print(f"\n--- Searching for: {title} ---")
        jobs = fetch_recent_jobs(
            search_term=title,
            location="Dubai",
            country_indeed="United Arab Emirates",
            site_names=target_boards,
            results_wanted=5, # Small number to avoid rate limiting during the test
            hours_old=168,    # Past 7 days
        )
        all_results.extend(jobs)
        
    # Deduplicate across all search terms
    unique_jobs = {job.id: job for job in all_results}.values()
    
    print(f"\n[✓] Scraped {len(unique_jobs)} unique verified jobs across {len(target_titles)} titles.")
    for job in unique_jobs:
        print(f"  [{job.site}] {job.title} at {job.company} | {job.location}")