"""
CV Builder Module for Career Engine.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from string import Template
from typing import Dict, List, Optional

from anti_ai import sanitize_typst, validate_entropy, validate_no_banned
from llm import LLMClient
from models import (
    BridgeConfig,
    MasterFacts,
    MatchAnalysis,
    Pass1Payload,
    Pass2Payload,
    Pass3Payload,
    RoleBullets,
    TimelineRole,
)
from prompts import (
    SYSTEM_INSTRUCTION,
    build_pass1_prompt,
    build_pass2_prompt,
    build_pass3_prompt,
)

logger = logging.getLogger("CVBuilder")


def _get_typst_binary() -> str:
    """Finds the Typst binary across platforms, falling back to local executable."""
    typst = shutil.which("typst")
    if typst:
        return typst
    if os.path.exists("typst.exe"):
        return "./typst.exe"
    raise RuntimeError("Typst CLI not found. Please install it or place typst.exe in the project root.")


def _strip_brackets(text: str) -> str:
    """Strips surrounding brackets that LLMs sometimes hallucinate in summaries."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1].strip()
    return text


def _extract_timeline(master_facts: MasterFacts, bridge_config: BridgeConfig) -> List[TimelineRole]:
    """Constructs the deterministic, non-overlapping timeline."""
    timeline: List[TimelineRole] = []

    # 1. Current Anchor (TELUS)
    timeline.append(
        TimelineRole(
            id="role_telus",
            company="TELUS Digital",
            role="Lead Full-Stack & AI Systems Engineer",
            location="Nairobi, Kenya",
            period="Mar 2024 - Present",
            pass_group="recent",
        )
    )

    # 2. Gap Bridge (Independent Consulting)
    timeline.append(bridge_config.umbrella_role)

    # 3. Foundational Roles (from master_facts)
    elimu = next((r for r in master_facts.career_timeline if r.company and "Elimu" in r.company), None)
    if elimu:
        timeline.append(
            TimelineRole(
                id="role_elimu",
                company=elimu.company or "Elimu Digital TV",
                role=elimu.role,
                location=elimu.location,
                period=elimu.period,
                pass_group="foundational",
            )
        )

    freelance = next((r for r in master_facts.career_timeline if "Self-Employed" in r.role or "Freelance" in r.role), None)
    if freelance:
        timeline.append(
            TimelineRole(
                id="role_freelance",
                company=freelance.company or "Self-Employed",
                role=freelance.role,
                location=freelance.location,
                period=freelance.period,
                pass_group="foundational",
            )
        )

    return timeline


def _validate_bullets(role_bullets: List[RoleBullets], expected_role_ids: List[str], tools: List[str]) -> List[str]:
    """Validates that all roles are present and bullets pass Anti-AI checks."""
    findings = []
    bullets_dict = {rb.role_id: rb.bullets for rb in role_bullets}
    
    for r_id in expected_role_ids:
        if r_id not in bullets_dict:
            findings.append(f"Missing role ID: {r_id}")
            continue
            
        bullets = bullets_dict[r_id]
        if len(bullets) < 3:
            findings.append(f"Role {r_id} has only {len(bullets)} bullets, expected 3-4.")
            
        for b in bullets:
            banned = validate_no_banned(b)
            findings.extend([f"Role {r_id}: {f}" for f in banned])
            
        entropy = validate_entropy(bullets, tools)
        findings.extend([f"Role {r_id}: {e}" for e in entropy])
        
    return findings


def _execute_pass_with_retry(
    pass_name: str,
    prompt: str,
    llm_client: LLMClient,
    response_model: type,
    role_ids: List[str],
    tools: List[str],
    max_tokens: int,
) -> type:
    """Executes an LLM pass and re-prompts once if anti-AI validation fails."""
    logger.info(f"[{pass_name}] Generating...")
    
    # First attempt
    payload = llm_client.generate_json(
        prompt=prompt,
        response_model=response_model,
        system_instruction=SYSTEM_INSTRUCTION,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    
    if hasattr(payload, "role_bullets"):
        findings = _validate_bullets(payload.role_bullets, role_ids, tools)
        if findings:
            logger.warning(f"[{pass_name}] Anti-AI validation failed. Retrying with specific feedback...")
            
            # Format the specific offending text for the LLM
            feedback_str = "\n".join(f"- {f}" for f in findings[:10])
            
            retry_prompt = (
                f"{prompt}\n\n"
                f"WARNING: Your previous response violated the Anti-AI protocol. "
                f"You must fix these exact issues and return the corrected JSON:\n{feedback_str}\n\n"
                f"CRITICAL: Replace banned verbs like 'Architected' with 'Built'. Replace 'Engineered' with 'Developed'. "
                f"Vary the opening words of consecutive bullets. Ensure bullet lengths vary significantly."
            )
            
            payload = llm_client.generate_json(
                prompt=retry_prompt,
                response_model=response_model,
                system_instruction=SYSTEM_INSTRUCTION,
                max_tokens=max_tokens,
                temperature=0.5, # Higher temp to force different phrasing
            )
            findings = _validate_bullets(payload.role_bullets, role_ids, tools)
            if findings:
                logger.error(f"[{pass_name}] Validation failed again. Findings: {findings}")
                logger.error(f"Proceeding, but manual review required.")
                
    return payload


def generate_tailored_cv(
    job_description: str,
    master_facts: MasterFacts,
    bridge_config: BridgeConfig,
    llm_client: LLMClient,
    match_analysis: Optional[MatchAnalysis] = None,
    output_pdf_path: str = "output_resume.pdf",
) -> bool:
    """
    Orchestrates the 3-pass DIS pipeline and compiles the PDF.
    """
    try:
        typst_bin = _get_typst_binary()
        timeline = _extract_timeline(master_facts, bridge_config)
        
        recent_role = next(r for r in timeline if r.pass_group == "recent")
        bridge_role = next(r for r in timeline if r.pass_group == "bridge")
        foundational_roles = [r for r in timeline if r.pass_group == "foundational"]

        tools = master_facts.skills_taxonomy.get("direct_core", [])

        # --- PASS 1 ---
        p1_prompt = build_pass1_prompt(job_description, master_facts)
        if match_analysis and match_analysis.tailored_summary:
            p1_prompt += f"\n\nREFERENCE SUMMARY FROM SCORER:\n{match_analysis.tailored_summary}\n(Improve upon this, but maintain the exact same facts.)"
            
        # 3000 tokens to prevent truncation
        pass1 = _execute_pass_with_retry("Pass 1", p1_prompt, llm_client, Pass1Payload, [], tools, max_tokens=3000)

        # --- PASS 2 ---
        p2_prompt = build_pass2_prompt(job_description, recent_role, bridge_role, bridge_config, master_facts)
        pass2 = _execute_pass_with_retry("Pass 2", p2_prompt, llm_client, Pass2Payload, [recent_role.id, bridge_role.id], tools, max_tokens=3000)

        # --- PASS 3 ---
        p3_prompt = build_pass3_prompt(job_description, foundational_roles, master_facts)
        pass3 = _execute_pass_with_retry("Pass 3", p3_prompt, llm_client, Pass3Payload, [r.id for r in foundational_roles], tools, max_tokens=3000)

        # --- ASSEMBLY ---
        logger.info("[Assembly] Merging payloads and compiling Typst PDF...")
        
        all_bullets: Dict[str, List[str]] = {}
        for rb in pass2.role_bullets:
            all_bullets[rb.role_id] = rb.bullets
        for rb in pass3.role_bullets:
            all_bullets[rb.role_id] = rb.bullets
        
        exp_blocks = []
        for role in timeline:
            bullets = all_bullets.get(role.id, ["[Missing bullets - manual review required]"])
            bullet_lines = "\n".join(f"- {sanitize_typst(_strip_brackets(b))}" for b in bullets)
            
            block = f"""
*{sanitize_typst(role.role)}* \\
*{sanitize_typst(role.company)}* | {sanitize_typst(role.location)} #h(1fr) _{sanitize_typst(role.period)}_
{bullet_lines}
#v(6pt)
"""
            exp_blocks.append(block)
            
        experience_section_str = "\n".join(exp_blocks)

        # Format the structured skills into a Typst string
        skills_lines = []
        if pass1.skills:
            for cat in pass1.skills:
                # Join the items with commas, then sanitize the whole string
                items_str = ", ".join(sanitize_typst(i) for i in cat.items)
                # Bold the category name safely and add a Typst line break (\)
                skills_lines.append(f"*{sanitize_typst(cat.category)}:* {items_str} \\")
        skills_text = "\n".join(skills_lines)

        # Strip brackets from the summary
        summary_text = sanitize_typst(_strip_brackets(pass1.summary))

        edu = master_facts.profile.education
        edu_str = (
            f"*{sanitize_typst(edu.degree)}* \\\n"
            f"{sanitize_typst(edu.institution)} | Graduated: {sanitize_typst(edu.graduation)}"
        )

        certs_str = ""
        if pass3.certifications:
            certs_str = " • ".join([sanitize_typst(c) for c in pass3.certifications])

        with open("cv_template.typ", "r", encoding="utf-8") as f:
            template_str = f.read()
            
        t = Template(template_str)
        typst_content = t.safe_substitute(
            name=sanitize_typst(master_facts.profile.name),
            title=sanitize_typst(master_facts.profile.title),
            location=sanitize_typst(master_facts.profile.location),
            phone=sanitize_typst(master_facts.profile.phone),
            email_raw=master_facts.profile.email,
            email_display=sanitize_typst(master_facts.profile.email),
            linkedin=master_facts.profile.linkedin,
            github=master_facts.profile.github,
            summary=summary_text,
            skills_section=skills_text,
            experience_section=experience_section_str,
            education_section=edu_str,
            certifications_section=certs_str,
        )

        output_dir = os.path.dirname(os.path.abspath(output_pdf_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            typ_path = os.path.join(tmpdir, "resume.typ")
            with open(typ_path, "w", encoding="utf-8") as f:
                f.write(typst_content)

            result = subprocess.run(
                [typst_bin, "compile", typ_path, output_pdf_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"Typst compilation failed:\n{result.stderr}")
                return False

        logger.info(f"[*] Successfully generated: {output_pdf_path}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Typst compilation timed out after 30 seconds.")
        return False
    except Exception as e:
        logger.error(f"Failed to generate tailored CV: {str(e)}", exc_info=True)
        return False