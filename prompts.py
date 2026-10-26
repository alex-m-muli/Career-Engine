"""
Prompt generation module for the Career Engine.
"""

from __future__ import annotations

import json
from typing import List

from models import BridgeConfig, MasterFacts, TimelineRole

SYSTEM_INSTRUCTION = """
You are an expert human executive resume writer and senior technical talent consultant.
Your objective is to generate resume text that sounds 100% written by an experienced human software engineer.

STRICT ANTI-AI PROTOCOL (ZERO TOLERANCE):
1. BANNED VERBS: Architected, Spearheaded, Orchestrated, Championed, Leveraged, Delved, Pivotal, Synergized, Harmonized, Catalyzed, Streamlined, Engineered.
2. BANNED BUZZWORDS: Results-driven, Detail-oriented, Passionate, Cutting-edge, Forward-thinking, Visionary, Dynamic, Seamlessly, Robustly, Transformative, Synergy, Paradigm, Best-in-class, Groundbreaking, Relentless, Strategic maven, Innovative.
3. BANNED PUNCTUATION: Do NOT use em-dashes (-) or en-dashes (–). Do NOT use colon-gated lead-ins (e.g., "Key accomplishments include:"). Do NOT use rhythmic rule-of-three triplets (e.g., "scalable, secure, and performant").
4. STRUCTURAL ENTROPY: Rotate bullet openings across these four styles:
   - Context/Problem-First: "Due to database query timeouts during morning surges, rewrote PostgreSQL queries..."
   - Tool/Stack-First: "Used Redis caching to lower database read load on high-traffic account balances."
   - Simple Duty/Maintenance: "Maintained core backend services running on .NET 8, updating middleware..."
   - Direct Action + Grounded Metric: "Split a monolithic script into microservices, cutting memory usage by 35%..."
   Never place two bullets with the same opening pattern consecutively.
5. CADENCE & FRICTION: Mix short 8-12 word punchy statements with detailed 22+ word technical breakdowns. Explicitly cite real-world engineering constraints (rate limits, memory leaks, legacy migrations).
6. NO FLUFF: State actions directly. Remove qualitative adverbs like "seamlessly" or "effortlessly".
7. DO NOT INCLUDE REFERENCE IDs: Never append citation tags like [b1] or [b2] to your generated text.

You must output STRICT JSON matching the requested schema. Do not include markdown formatting, explanations, or conversational filler outside the JSON block.
"""

SCORER_SYSTEM_INSTRUCTION = """
You are an expert technical recruitment and resume matching engine. Your task is to objectively evaluate candidate fit against a target job description.

STRICT CONSTRAINTS:
1. NEVER fabricate metrics or candidate experience not supported by master facts.
2. Score from 0.0 to 10.0:
   - Exact core skill matches = 100% value.
   - Transferable equivalencies (e.g. candidate has AWS, job needs Azure) = 85% value.
   - Unmatched core requirements decrease fit score proportionally.
3. Identify critical missing skills NOT found in core, transferable, or adjacent taxonomy.
4. Select 3-5 existing bullet IDs from the `experience_bullets` that maximize job-relevant keywords.
5. Flag scam risk if the posting mentions upfront fees, wire transfers, crypto, or suspicious recruitment methods.
6. Set the `status` field to "ok".

You must output STRICT JSON matching the requested schema. Do not include markdown formatting.
"""

def build_pass1_prompt(job_description: str, master_facts: MasterFacts) -> str:
    taxonomy = master_facts.skills_taxonomy
    profile = master_facts.profile

    return f"""
TARGET JOB DESCRIPTION:
{job_description[:4000]}

CANDIDATE PROFILE:
Name: {profile.name}
Current Title: {profile.title}

CANDIDATE SKILLS TAXONOMY:
{json.dumps(taxonomy, indent=2)}

OUTPUT REQUIREMENTS:
Generate a tailored professional summary and structured technical skills taxonomy for this candidate targeting the job description above.

1. "summary": A 4-sentence human-sounding professional summary string. Highlight core engineering strengths, practical RAG/AI experience, and cloud infrastructure background. Do not invent facts. Do not use banned words. Do NOT wrap the summary in brackets or quotes.
2. "skills": A JSON array of objects. Each object must have two keys: "category" (e.g., Languages, Backend, Frontend, Cloud/DevOps, AI/Data) and "items" (an array of specific technology strings). 
   - Do NOT format the items with colons, asterisks, bullet points, or markdown. Just the raw technology names (e.g., "Python", "React").
   - Tailor the list to the JD by prioritizing the most relevant technologies. Do not dump every skill.
   - Python will format this into the resume later.
"""

def build_pass2_prompt(
    job_description: str,
    recent_role: TimelineRole,
    bridge_role: TimelineRole,
    bridge_config: BridgeConfig,
    master_facts: MasterFacts,
) -> str:
    projects_json = json.dumps([p.model_dump() for p in bridge_config.project_pool], indent=2)
    ref_bullets_json = json.dumps([b.model_dump() for b in master_facts.experience_bullets], indent=2)

    return f"""
TARGET JOB DESCRIPTION:
{job_description[:4000]}

REFERENCE BULLETS (Use these for factual grounding and tone):
{ref_bullets_json}

You are generating achievement bullets for the following roles:

ROLE 1 (Recent):
{json.dumps(recent_role.model_dump(), indent=2)}

ROLE 2 (Consulting):
{json.dumps(bridge_role.model_dump(), indent=2)}

PROJECT POOL FOR ROLE 2:
Select the 3 most relevant projects from this pool to form the basis of the bullets for Role 2. Tailor the descriptions to the Job Description.
{projects_json}

OUTPUT REQUIREMENTS:
Return a strict JSON object containing a "role_bullets" array. Each object in the array must have a "role_id" and a "bullets" array.

- For "{recent_role.id}": Write 3-4 high-impact bullets based on the reference bullets provided. Focus on the technical friction, metrics, and the required stack from the JD.
- For "{bridge_role.id}": Write 3-4 bullets. Use the selected projects from the pool as the factual anchor. Adapt the descriptions to sound like direct client engagements, emphasizing the specific tools and outcomes relevant to the JD.

Strictly apply the Anti-AI protocol. Rotate bullet openings. Do not use banned verbs or buzzwords. Do NOT append reference IDs like [b1] to your text.
"""

def build_pass3_prompt(
    job_description: str,
    foundational_roles: List[TimelineRole],
    master_facts: MasterFacts,
) -> str:
    ref_bullets_json = json.dumps([b.model_dump() for b in master_facts.experience_bullets], indent=2)
    roles_json = json.dumps([r.model_dump() for r in foundational_roles], indent=2)

    return f"""
TARGET JOB DESCRIPTION:
{job_description[:4000]}

REFERENCE BULLETS (Use these for factual grounding and tone):
{ref_bullets_json}

You are generating achievement bullets for the following foundational roles:
{roles_json}

OUTPUT REQUIREMENTS:
Return a strict JSON object with two keys: "role_bullets" (an array of objects) and "certifications" (an array of strings).

1. "role_bullets": An array of objects. Each object must have a "role_id" and a "bullets" array. Focus on foundational software engineering, backend work, and early technical challenges. Tailor the language to align with the target JD where appropriate, but do not fabricate senior-level achievements for early roles.
2. "certifications": An array of 4 to 5 professional certification strings. You may select from the candidate's existing profile or suggest highly relevant industry certifications based on the JD. Format them cleanly (e.g., "AWS Certified Solutions Architect - Associate").

Strictly apply the Anti-AI protocol. Rotate bullet openings. Do not use banned verbs or buzzwords. Do NOT append reference IDs like [b1] to your text.
"""

def build_scorer_prompt(job_description: str, master_facts: MasterFacts) -> str:
    taxonomy = master_facts.skills_taxonomy
    bullets = master_facts.experience_bullets

    return f"""
TARGET JOB DESCRIPTION:
{job_description[:6000]}

CANDIDATE TAXONOMY:
{json.dumps(taxonomy, indent=2)}

CANDIDATE EXPERIENCE BULLETS:
{json.dumps([b.model_dump() for b in bullets], indent=2)}
"""