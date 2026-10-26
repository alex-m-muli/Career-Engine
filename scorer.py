"""
Scorer Module for Career Engine.

Evaluates job descriptions against candidate master facts using the shared
LLMClient (Gemini 3.8 Flash with Groq fallback). Computes precision fit scores,
identifies skill gaps, and generates dynamic skill bridges.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from anti_scam import check_listing
from llm import LLMClient
from models import MatchAnalysis, MasterFacts
from prompts import SCORER_SYSTEM_INSTRUCTION, build_scorer_prompt

logger = logging.getLogger("FitScorer")


def _error_analysis(reason: str) -> MatchAnalysis:
    """Helper to generate a standardized error response."""
    return MatchAnalysis(
        status="error",
        is_scam_risk=False,
        scam_reason=reason,
        fit_score=0.0,
        missing_critical_skills=[],
        skill_bridge_statements=[],
        selected_bullet_ids=[],
        tailored_summary="Evaluation failed due to an internal error.",
    )


def _scam_analysis(reason: str) -> MatchAnalysis:
    """Helper to generate a standardized scam response."""
    return MatchAnalysis(
        status="scam",
        is_scam_risk=True,
        scam_reason=reason,
        fit_score=0.0,
        missing_critical_skills=[],
        skill_bridge_statements=[],
        selected_bullet_ids=[],
        tailored_summary="Rejected due to recruitment fraud/scam indicators.",
    )


def analyze_job_fit(
    job_description: str,
    master_facts: MasterFacts,
    llm_client: LLMClient,
) -> MatchAnalysis:
    """
    Analyzes a job description against candidate master facts.

    Args:
        job_description (str): Sanitized plain text job description.
        master_facts (MasterFacts): Candidate facts model.
        llm_client (LLMClient): Initialized LLM client with retries and fallbacks.

    Returns:
        MatchAnalysis: Strongly typed evaluation response object.
    """
    # 1. Fast local scam check (zero API cost)
    is_scam, scam_reason = check_listing("Target Role", "Target Company", job_description)
    if is_scam:
        logger.warning(f"[SCAM DETECTED LOCAL] {scam_reason}")
        return _scam_analysis(scam_reason)

    # 2. Build prompt
    prompt = build_scorer_prompt(job_description, master_facts)

    # 3. Invoke LLM Client
    try:
        logger.info("Invoking LLM for fit analysis...")
        analysis = llm_client.generate_json(
            prompt=prompt,
            response_model=MatchAnalysis,
            system_instruction=SCORER_SYSTEM_INSTRUCTION,
            max_tokens=1500,
            temperature=0.1,
        )
        
        # Defensive check: LLM might have hallucinated "scam" despite local passing
        if analysis.is_scam_risk:
            logger.warning(f"[SCAM DETECTED LLM] {analysis.scam_reason}")
            analysis.status = "scam"
        else:
            analysis.status = "ok"

        logger.info(
            f"Analysis Complete | Fit Score: {analysis.fit_score}/10.0 | "
            f"Scam Risk: {analysis.is_scam_risk} | Status: {analysis.status}"
        )
        return analysis

    except Exception as e:
        logger.error(f"Error executing LLM fit evaluation: {str(e)}", exc_info=True)
        return _error_analysis(f"Evaluation Error: {str(e)}")


if __name__ == "__main__":
    import json
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("[*] Testing Scorer module standalone run...")
    
    with open("master_facts.json", "r", encoding="utf-8") as f:
        sample_facts = MasterFacts.model_validate(json.load(f))

    sample_jd = """
    We are looking for a Senior Full-Stack Engineer with strong TypeScript, React, Next.js, and Azure Cloud experience.
    Must have experience building REST and microservice backends with Python or Node.js.
    Knowledge of Kubernetes and Docker is highly preferred.
    """

    try:
        client = LLMClient()
        res = analyze_job_fit(sample_jd, sample_facts, client)
        
        print(f"\n[✓] Standalone Test Complete!")
        print(f"    Status: {res.status}")
        print(f"    Fit Score: {res.fit_score}/10.0")
        print(f"    Selected Bullets: {res.selected_bullet_ids}")
        print(f"    Missing Skills: {res.missing_critical_skills}")
        print(f"    Summary: {res.tailored_summary[:100]}...")
    except Exception as e:
        print(f"\n[X] Test Failed: {e}")