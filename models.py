"""
Pydantic v2 models for the Career Engine.

Central data contract file. Every module imports from here.

Design notes
------------
- Models sent to an LLM as a structured-output schema use `extra="forbid"`.
  This makes Pydantic emit `additionalProperties: false`, which Groq's
  strict Structured Outputs mode requires. Without it, Groq can hang
  indefinitely on some requests.

- Those same LLM-facing models avoid field defaults so that every field
  lands in the JSON Schema `required` array. OpenAI/Groq strict mode
  rejects schemas with optional fields. Fields that may be absent use
  `Optional[T]` and accept explicit null instead.

- Models that ingest user-edited data (`master_facts.json`,
  `bridge_roles.json`) use `extra="ignore"` so additional keys do not
  break loading.

- SQLite row models keep defaults for ergonomics. They are never sent
  to an LLM, so strict-mode rules do not apply.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Scraper output
# ---------------------------------------------------------------------------

class JobPosting(BaseModel):
    """A sanitized, scam-filtered job posting returned by the scraper."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Stable SHA-256 prefix of the job URL.")
    site: str = Field(description="Source board: 'linkedin', 'indeed', etc.")
    title: str
    company: str
    location: str
    job_url: str
    description: str
    date_posted: str = ""
    is_remote: bool = False


# ---------------------------------------------------------------------------
# Scorer output
# ---------------------------------------------------------------------------

AnalysisStatus = Literal["ok", "error", "scam"]


class MatchAnalysis(BaseModel):
    """Structured evaluation of candidate fit against a target job description."""

    model_config = ConfigDict(extra="forbid")

    status: AnalysisStatus = Field(
        description="'ok' if the analysis completed, 'error' if the scorer failed, 'scam' if fraud was detected."
    )
    is_scam_risk: bool = Field(
        description="True if the posting shows scam, fraud, fee requests, or phishing characteristics."
    )
    scam_reason: Optional[str] = Field(
        description="Short explanation when is_scam_risk is true, otherwise null."
    )
    fit_score: float = Field(
        description="Match score from 0.0 to 10.0 based on core skills, transferable equivalencies, and experience level."
    )
    missing_critical_skills: List[str] = Field(
        description="Mandatory skills absent from the candidate's full taxonomy."
    )
    skill_bridge_statements: List[str] = Field(
        description="Statements articulating how existing skills bridge specific JD requirements without fabricating experience."
    )
    selected_bullet_ids: List[str] = Field(
        description="IDs from the reference bullet library that best align with this role. Advisory only; the generator may ignore them."
    )
    tailored_summary: str = Field(
        description="3-sentence summary positioning the candidate for this JD without fabricating facts."
    )


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

PassGroup = Literal["recent", "bridge", "foundational"]


class TimelineRole(BaseModel):
    """A single role in the deterministic career timeline."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Stable identifier, e.g. 'role_telus'.")
    company: str
    role: str
    location: str
    period: str = Field(description="Human-readable date range, e.g. 'Mar 2024 - Present'.")
    pass_group: PassGroup


# ---------------------------------------------------------------------------
# Bridge configuration
# ---------------------------------------------------------------------------

class UmbrellaRole(TimelineRole):
    """The single consulting role that covers the gap period."""
    pass


class Project(BaseModel):
    """A project template in the consulting pool."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    description: str
    keywords: List[str]


class BridgeConfig(BaseModel):
    """The schema for bridge_roles.json."""

    model_config = ConfigDict(extra="ignore")

    umbrella_role: UmbrellaRole
    project_pool: List[Project]


# ---------------------------------------------------------------------------
# LLM synthesis payloads
# ---------------------------------------------------------------------------

class RoleBullets(BaseModel):
    """Explicit object model to satisfy Groq's strict JSON schema requirements."""
    model_config = ConfigDict(extra="forbid")
    
    role_id: str = Field(description="The exact ID of the role (e.g., 'role_telus').")
    bullets: List[str] = Field(description="3 to 4 achievement bullet strings.")


class SkillCategory(BaseModel):
    """Structured output for skills to prevent LLM formatting hallucinations."""
    model_config = ConfigDict(extra="forbid")
    
    category: str = Field(description="e.g., Languages, Backend, Frontend, Cloud/DevOps, AI/Data.")
    items: List[str] = Field(description="List of specific technologies in this category.")


class Pass1Payload(BaseModel):
    """Output of Pass 1: executive summary and categorized skills."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="4-sentence human-sounding professional summary.")
    skills: List[SkillCategory] = Field(description="Array of structured skill categories.")


class Pass2Payload(BaseModel):
    """Output of Pass 2: bullets for recent and bridge roles."""

    model_config = ConfigDict(extra="forbid")

    role_bullets: List[RoleBullets] = Field(description="Array of role bullet objects.")


class Pass3Payload(BaseModel):
    """Output of Pass 3: bullets for foundational roles and certifications."""

    model_config = ConfigDict(extra="forbid")

    role_bullets: List[RoleBullets] = Field(description="Array of role bullet objects.")
    certifications: List[str] = Field(description="4-5 certification strings aligned with the target role.")


# ---------------------------------------------------------------------------
# Master facts (user-edited data)
# ---------------------------------------------------------------------------

class EducationInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    degree: str
    institution: str
    graduation: str


class ProfileInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    title: str
    secondaryTitle: Optional[str] = None
    email: str
    phone: str
    location: str
    linkedin: str
    github: str
    education: EducationInfo
    certifications: List[str] = Field(default_factory=list)


class ExperienceBullet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    category: str
    text: str
    keywords: List[str] = Field(default_factory=list)


class CareerTimelineEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    company: Optional[str] = None
    period: str
    location: str
    description: str = ""


class MasterFacts(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile: ProfileInfo
    career_timeline: List[CareerTimelineEntry] = Field(default_factory=list)
    skills_taxonomy: Dict[str, Any] = Field(default_factory=dict)
    experience_bullets: List[ExperienceBullet] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Application tracking
# ---------------------------------------------------------------------------

ApplicationStatus = Literal["applied", "skipped", "failed", "pending"]


class ApplicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_url: str
    company: str
    title: str
    fit_score: float
    pdf_path: str = ""
    submitted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    status: ApplicationStatus = "pending"


__all__ = [
    "JobPosting",
    "AnalysisStatus",
    "MatchAnalysis",
    "PassGroup",
    "TimelineRole",
    "UmbrellaRole",
    "Project",
    "BridgeConfig",
    "RoleBullets",
    "SkillCategory",
    "Pass1Payload",
    "Pass2Payload",
    "Pass3Payload",
    "EducationInfo",
    "ProfileInfo",
    "ExperienceBullet",
    "CareerTimelineEntry",
    "MasterFacts",
    "ApplicationStatus",
    "ApplicationRecord",
]