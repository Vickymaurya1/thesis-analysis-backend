import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from anthropic import Anthropic

from app.config import settings
from app.models import ThesisVersion, Flag, AnalysisSnapshot, ReviewerSimSession

def get_thesis_analysis_summary(db: Session, version: ThesisVersion) -> str:
    # Gather flags
    flags = db.query(Flag).join(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == version.id
    ).all()
    
    # Gather snapshots
    snapshots = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == version.id
    ).all()

    summary_lines = []
    summary_lines.append(f"Thesis Title: {version.thesis.title}")
    summary_lines.append(f"Field: {version.thesis.field or 'General'}")
    summary_lines.append(f"Degree Level: {version.thesis.degree_level or 'Master'}")
    summary_lines.append("\nQuality Snapshots:")
    for snap in snapshots:
        summary_lines.append(f"- Tool: {snap.tool_type} | Scores/Metadata: {json.dumps(snap.scores)}")
        
    summary_lines.append("\nAutomated Review Flags:")
    for f in flags:
        summary_lines.append(f"- Type: {f.type} | Severity: {f.severity} | Message: {f.message} | Excerpt: '{f.evidence_excerpt}' | ID: {f.id}")
        
    return "\n".join(summary_lines)

def run_practice_viva_turn(
    db: Session,
    session: ReviewerSimSession,
    user_reply: Optional[str] = None
) -> Dict[str, Any]:
    version = db.query(ThesisVersion).filter(
        ThesisVersion.thesis_id == session.thesis_id
    ).order_by(ThesisVersion.version_number.desc()).first()
    
    if not version:
        raise ValueError("No thesis version uploaded yet.")

    analysis_summary = get_thesis_analysis_summary(db, version)
    
    # Initialize transcript if empty
    transcript = session.transcript or []
    
    # Append user reply if present
    if user_reply:
        transcript.append({"role": "user", "content": user_reply})

    # Mock mode fallback
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or api_key == "mock_api_key_for_testing" or settings.ENV == "test":
        if not transcript:
            # First question
            question = "Could you justify the structured pruning method you chose in your methodology?"
        else:
            # Feedback + follow-up
            question = "That's a reasonable justification, but how do you verify the threshold doesn't lose accuracy?"
            
        first_flag = db.query(Flag).join(AnalysisSnapshot).filter(
            AnalysisSnapshot.version_id == version.id
        ).first()
        ref_flag_id = first_flag.id if first_flag else "mock-flag-uuid"
        
        examiner_msg = {
            "role": "assistant",
            "content": question
        }
        transcript.append(examiner_msg)
        session.transcript = transcript
        db.commit()
        
        return {
            "examiner_message": question,
            "referenced_flag_id": ref_flag_id
        }

    # Format history for Anthropic message schema
    messages = []
    for turn in transcript:
        messages.append({
            "role": "user" if turn["role"] == "user" else "assistant",
            "content": turn["content"]
        })

    system_prompt = (
        f"You are simulating a rigorous but fair thesis viva examiner for a {version.thesis.degree_level or 'Master'} student in {version.thesis.field or 'General'}.\n\n"
        "Here is the context about their thesis and automated review findings:\n"
        "------------------\n"
        f"{analysis_summary}\n"
        "------------------\n\n"
        "Ask one probing question at a time, in the voice of an examiner — not a chatbot assistant. Prioritize questions about:\n"
        "- Weak points already flagged by the quality/citation/novelty reviews (test whether the student can defend or has already fixed them)\n"
        "- Methodology choices that need justification\n"
        "- Claims in the conclusion that may overreach the evidence\n\n"
        "After the student answers, give brief examiner-style feedback before asking the next question. "
        "Stay in character as an examiner throughout — do not break into a tutor voice.\n\n"
        "Output strict JSON only with the following format:\n"
        "{\n"
        "  \"examiner_message\": \"your examiner question or response text\",\n"
        "  \"referenced_flag_id\": \"the UUID of the automated flag this question probes, or null if it is a general question\"\n"
        "}"
    )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system=system_prompt,
        messages=messages if messages else [{"role": "user", "content": "Hello, I am ready for my practice viva."}],
        temperature=0.7
    )

    raw_res = response.content[0].text.strip()
    try:
        if "```json" in raw_res:
            raw_res = raw_res.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_res:
            raw_res = raw_res.split("```")[1].split("```")[0].strip()
        parsed_res = json.loads(raw_res)
        examiner_message = parsed_res.get("examiner_message", raw_res)
        referenced_flag_id = parsed_res.get("referenced_flag_id")
    except Exception:
        examiner_message = raw_res
        referenced_flag_id = None
    
    # Save assistant turn
    transcript.append({"role": "assistant", "content": examiner_message})
    session.transcript = transcript
    db.commit()

    return {
        "examiner_message": examiner_message,
        "referenced_flag_id": referenced_flag_id
    }

def generate_teacher_viva_report(
    db: Session,
    session: ReviewerSimSession
) -> Dict[str, Any]:
    # Check if report already generated
    if session.report:
        return session.report

    version = db.query(ThesisVersion).filter(
        ThesisVersion.thesis_id == session.thesis_id
    ).order_by(ThesisVersion.version_number.desc()).first()
    
    if not version:
        raise ValueError("No thesis version uploaded yet.")

    analysis_summary = get_thesis_analysis_summary(db, version)

    # Mock mode fallback
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or api_key == "mock_api_key_for_testing" or settings.ENV == "test":
        first_flag = db.query(Flag).join(AnalysisSnapshot).filter(
            AnalysisSnapshot.version_id == version.id
        ).first()
        ref_flag_id = first_flag.id if first_flag else "mock-flag-uuid"
        
        report_data = {
            "overall_assessment": "The thesis introduces a structured attention pruning mechanism. It displays reasonable grounding but has minor citation gaps.",
            "strengths": [
                "Clear methodology definition",
                "Solid evaluation speedups reported"
            ],
            "weaknesses": [
                {
                    "point": "Lacks comprehensive coverage in attention-pruning literature review.",
                    "evidence_excerpt": "We propose a novel attention head pruning mechanism.",
                    "related_flag_id": ref_flag_id
                }
            ],
            "suggested_questions": [
                "How do you justify the 0.15 similarity threshold chosen for paper clustering?",
                "Did you verify model accuracy retains 98%+ baseline after structured pruning?"
            ]
        }
        session.report = report_data
        db.commit()
        return report_data

    system_prompt = (
        f"You are drafting a mock external examiner's report for a {version.thesis.degree_level or 'Master'} thesis in {version.thesis.field or 'General'}, to help the supervising teacher prepare for the actual viva/defense.\n\n"
        "Here is the context about their thesis and automated review findings:\n"
        "------------------\n"
        f"{analysis_summary}\n"
        "------------------\n\n"
        "Write a structured report:\n"
        "- Overall assessment (2-3 sentences)\n"
        "- Strengths (bulleted)\n"
        "- Weaknesses (bulleted, each tied to specific evidence — reuse existing flags where relevant rather than re-deriving new criticism from scratch)\n"
        "- Suggested viva questions (5-8 questions an external examiner would likely ask)\n\n"
        "Rules:\n"
        "- This report is a preparation aid for the supervisor, not a grade or official verdict.\n"
        "- Ground every weakness in specific text evidence, same standard as the other tools.\n"
        "- Output strict JSON only."
    )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": "Please generate the mock reviewer report based on the provided analysis summary."}],
        temperature=0.0
    )

    raw_res = response.content[0].text.strip()
    try:
        if "```json" in raw_res:
            raw_res = raw_res.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_res:
            raw_res = raw_res.split("```")[1].split("```")[0].strip()
        report_json = json.loads(raw_res)
        
        session.report = report_json
        db.commit()
        return report_json
    except Exception as e:
        raise ValueError(f"Failed to parse Reviewer Report JSON output: {e}. Raw: {raw_res}")
