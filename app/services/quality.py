import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from anthropic import Anthropic

from app.config import settings
from app.models import (
    ThesisVersion, AnalysisSnapshot, Flag,
    FlagTypeEnum, SeverityEnum
)

RUBRIC_CONFIG = {
    "introduction": {
        "weight": 0.15,
        "criteria": "Problem clearly stated, motivation justified, contribution stated upfront"
    },
    "related_work": {
        "weight": 0.15,
        "criteria": "Adequate coverage, positions the thesis against prior work (not just a list)"
    },
    "methodology": {
        "weight": 0.25,
        "criteria": "Reproducible detail, justified design choices, limitations acknowledged"
    },
    "results": {
        "weight": 0.20,
        "criteria": "Results support claims, appropriate presentation, statistical rigor where relevant"
    },
    "discussion": {
        "weight": 0.15,
        "criteria": "Interprets results honestly, addresses limitations, doesn't overclaim"
    },
    "conclusion": {
        "weight": 0.10,
        "criteria": "Summarizes contribution accurately, doesn't introduce new unsupported claims"
    }
}

def evaluate_section_llm(
    section_name: str,
    section_text: str,
    degree_level: str,
    field: str
) -> Dict[str, Any]:
    
    criteria = RUBRIC_CONFIG[section_name]["criteria"]
    
    # If mock key or test mode, return a mocked response
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or api_key == "mock_api_key_for_testing" or settings.ENV == "test":
        # Realistic mock: score based on section text length and content
        text_len = len(section_text.strip())
        
        if text_len < 50:
            # Very short section — low score
            return {
                "score": 35,
                "strengths": [],
                "weaknesses": [{
                    "issue": f"Section '{section_name}' is too brief and lacks sufficient detail.",
                    "evidence_excerpt": section_text.strip()[:200] or "Section content is minimal.",
                    "severity": "critical",
                    "suggested_fix": f"Expand the {section_name} section with more analysis and evidence."
                }]
            }
        elif text_len < 300:
            # Short section — moderate score
            return {
                "score": 62,
                "strengths": ["Section is present and structured."],
                "weaknesses": [{
                    "issue": f"Section '{section_name}' could benefit from more depth.",
                    "evidence_excerpt": section_text.strip()[:200],
                    "severity": "moderate",
                    "suggested_fix": f"Add more detailed analysis and supporting evidence to {section_name}."
                }]
            }
        else:
            # Adequate section — good score
            return {
                "score": 82,
                "strengths": [
                    f"Section '{section_name}' is well-developed with sufficient detail.",
                    "Good use of evidence and structured argumentation."
                ],
                "weaknesses": []
            }

    system_prompt = (
        f"You are reviewing the {section_name} section of a {degree_level} thesis in {field}.\n\n"
        f"Score this section 0-100 against these criteria:\n"
        f"{criteria}\n\n"
        f"Rules:\n"
        f"- Every score must be justified by quoting or closely paraphrasing specific sentences from\n"
        f"  the text — never a bare number with no anchor.\n"
        f"- Distinguish \"weak because underdeveloped\" from \"weak because factually/logically flawed\" —\n"
        f"  these need different kinds of fixes.\n"
        f"- List strengths too, not just weaknesses — a 0-100 score with only criticism reads as\n"
        f"  harsher than intended.\n"
        f"- Do not invent problems not evidenced in the text. If the section is genuinely solid, say so\n"
        f"  plainly rather than manufacturing a critique to seem thorough.\n"
        f"- Output strict JSON only."
    )

    user_content = (
        f"Section: {section_name}\n"
        f"Full text:\n"
        f"{section_text}\n\n"
        f"Return JSON:\n"
        f"{{\n"
        f"  \"score\": 0-100,\n"
        f"  \"strengths\": [\"...\", \"...\"],\n"
        f"  \"weaknesses\": [\n"
        f"    {{\"issue\": \"...\", \"evidence_excerpt\": \"exact quoted sentence\", \"severity\": \"critical|moderate|low\", \"suggested_fix\": \"...\"}}\n"
        f"  ]\n"
        f"}}"
    )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.0
    )

    raw_res = response.content[0].text.strip()
    try:
        if "```json" in raw_res:
            raw_res = raw_res.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_res:
            raw_res = raw_res.split("```")[1].split("```")[0].strip()
        return json.loads(raw_res)
    except Exception as e:
        raise ValueError(f"Failed to parse Quality Review LLM output: {e}. Raw: {raw_res}")

def run_quality_review_pipeline(db: Session, version: ThesisVersion) -> AnalysisSnapshot:
    raw_text = version.raw_text or ""
    section_map = version.section_map or {}
    degree_level = version.thesis.degree_level or "Master"
    field = version.thesis.field or "General"

    section_scores = {}
    total_flags = 0
    flags_to_create = []

    # Look up previous snapshot for incremental optimization
    prev_snapshot = None
    if version.version_number > 1:
        prev_version = db.query(ThesisVersion).filter(
            ThesisVersion.thesis_id == version.thesis_id,
            ThesisVersion.version_number == version.version_number - 1
        ).first()
        if prev_version:
            prev_snapshot = db.query(AnalysisSnapshot).filter(
                AnalysisSnapshot.version_id == prev_version.id,
                AnalysisSnapshot.tool_type == "quality_review"
            ).first()

    # Iterate through the 6 standard sections
    for sec_name, config in RUBRIC_CONFIG.items():
        is_changed = True
        if prev_snapshot and version.diff_from_prev and "changed_sections" in version.diff_from_prev:
            changed_secs = version.diff_from_prev.get("changed_sections", [])
            if sec_name not in changed_secs:
                is_changed = False

        if not is_changed and prev_snapshot and prev_snapshot.scores and "sections" in prev_snapshot.scores:
            prev_sec_data = prev_snapshot.scores["sections"].get(sec_name)
            if prev_sec_data:
                score = prev_sec_data.get("score", 0)
                section_scores[sec_name] = {
                    "score": score,
                    "flag_count": 0
                }
                # Retrieve and clone previous flags for this section
                prev_flags = db.query(Flag).filter(
                    Flag.snapshot_id == prev_snapshot.id,
                    Flag.page_ref == f"Section: {sec_name}"
                ).all()
                for pf in prev_flags:
                    flag = Flag(
                        type=pf.type,
                        severity=pf.severity,
                        message=pf.message,
                        evidence_excerpt=pf.evidence_excerpt,
                        page_ref=pf.page_ref
                    )
                    flags_to_create.append((flag, sec_name))
                    section_scores[sec_name]["flag_count"] += 1
                    total_flags += 1
                continue

        if sec_name in section_map:
            start, end = section_map[sec_name]
            section_text = raw_text[start:end].strip()
            
            # Run section evaluation
            eval_res = evaluate_section_llm(sec_name, section_text, degree_level, field)
            score = int(eval_res.get("score", 0))
            weaknesses = eval_res.get("weaknesses", [])
        else:
            # If section is completely missing, it receives a score of 0
            score = 0
            weaknesses = [{
                "issue": f"Section '{sec_name}' is missing from the thesis.",
                "evidence_excerpt": "N/A - Section not detected.",
                "severity": "critical",
                "suggested_fix": f"Add a dedicated {sec_name} section to address the required rubric criteria."
            }]

        section_scores[sec_name] = {
            "score": score,
            "flag_count": 0
        }

        # If score is below threshold (<70), we convert weaknesses to Flags
        if score < 70:
            for w in weaknesses:
                evidence = w.get("evidence_excerpt", "")
                
                # Non-negotiable: raise error if evidence_excerpt is null/empty
                # We enforce this at the application layer by raising a ValueError
                if not evidence or not str(evidence).strip():
                    raise ValueError("Flag must carry a non-empty evidence excerpt.")
                
                severity_str = w.get("severity", "moderate").lower()
                if severity_str not in ["critical", "moderate", "low"]:
                    severity_str = "moderate"
                    
                flag_msg = f"Issue: {w.get('issue', '')} | Fix: {w.get('suggested_fix', '')}"
                
                flag = Flag(
                    type=FlagTypeEnum.quality,
                    severity=SeverityEnum(severity_str),
                    message=flag_msg,
                    evidence_excerpt=evidence,
                    page_ref=f"Section: {sec_name}"
                )
                flags_to_create.append((flag, sec_name))
                section_scores[sec_name]["flag_count"] += 1
                total_flags += 1

    # Calculate weighted overall score
    weighted_sum = 0.0
    for sec_name, info in section_scores.items():
        weight = RUBRIC_CONFIG[sec_name]["weight"]
        weighted_sum += info["score"] * weight

    overall_score = int(round(weighted_sum))

    # Write AnalysisSnapshot
    snapshot = AnalysisSnapshot(
        version_id=version.id,
        tool_type="quality_review",
        scores={
            "overall": overall_score,
            "sections": section_scores
        }
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    # Attach flags to snapshot and save
    for flag, sec_name in flags_to_create:
        flag.snapshot_id = snapshot.id
        db.add(flag)
    
    if flags_to_create:
        db.commit()

    return snapshot
