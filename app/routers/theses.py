import os
import shutil
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from sqlalchemy.orm import Session
from typing import List, Optional, Any, Dict
from pydantic import BaseModel

from app.database import get_db
from app.models import (
    User, Thesis, ThesisVersion, CitationRecord, AnalysisSnapshot,
    Flag, FlagTypeEnum, RoleEnum, ReviewerSimSession, ThesisStatusEnum
)
from app.schemas import (
    ThesisCreate, ThesisResponse, ThesisVersionResponse,
    CitationResponse, DashboardResponse, FlagResponse,
    ReviewerSimSessionCreate, ReviewerSimMessageRequest, LiteratureReviewRequest,
    ReviewerSimSessionResponse
)
from app.dependencies import (
    get_current_user, require_feature_access, AccessLevel,
    assert_can_view_thesis, assert_owns_thesis
)
from app.permissions import can
from app.limiter import limiter, get_user_key
from app.services.audit import log_audit_action
from app.services.ingestion import run_ingestion_pipeline
from app.services.plagiarism import mask_plagiarism_flag_message
from app.services.orchestrator import run_on_thesis_update
from app.services.reviewersim import run_practice_viva_turn, generate_teacher_viva_report
from app.services.literature import run_literature_clustering_and_synthesis

router = APIRouter(prefix="/theses", tags=["theses"])

class ThesisStatusUpdate(BaseModel):
    status: str

@router.post("", response_model=ThesisResponse, status_code=status.HTTP_201_CREATED)
def create_thesis(
    thesis_in: ThesisCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != RoleEnum.student:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can create theses."
        )
        
    advisor_id = thesis_in.advisor_id
    if advisor_id:
        advisor = db.query(User).filter(User.id == advisor_id, User.role == RoleEnum.teacher).first()
        if not advisor:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Advisor not found or is not a teacher"
            )

    new_thesis = Thesis(
        owner_id=current_user.id,
        advisor_id=advisor_id,
        title=thesis_in.title,
        field=thesis_in.field,
        degree_level=thesis_in.degree_level
    )
    db.add(new_thesis)
    db.commit()
    db.refresh(new_thesis)
    return new_thesis

@router.get("/{thesis_id}", response_model=ThesisResponse)
def get_thesis(
    thesis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
        
    assert_can_view_thesis(current_user, thesis)
    return thesis

@router.get("/{thesis_id}/versions", response_model=List[ThesisVersionResponse])
def get_thesis_versions(
    thesis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
        
    assert_can_view_thesis(current_user, thesis)
    versions = db.query(ThesisVersion).filter(ThesisVersion.thesis_id == thesis_id).order_by(ThesisVersion.version_number.asc()).all()
    return versions

@router.post("/{thesis_id}/versions", response_model=ThesisVersionResponse, status_code=status.HTTP_201_CREATED)
async def upload_thesis_version(
    thesis_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
        
    # Only owner student can upload a version
    assert_owns_thesis(current_user, thesis)

    # 10MB limit validation
    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds maximum limit of 10MB"
        )
    
    file.file.seek(0, os.SEEK_END)
    actual_size = file.file.tell()
    file.file.seek(0)
    if actual_size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds maximum limit of 10MB"
        )
    
    last_version = db.query(ThesisVersion).filter(
        ThesisVersion.thesis_id == thesis_id
    ).order_by(ThesisVersion.version_number.desc()).first()
    
    next_ver_num = (last_version.version_number + 1) if last_version else 1

    # Create local uploads folder
    uploads_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Save file locally
    temp_filename = f"version_{next_ver_num}_{file.filename}"
    file_path = os.path.join(uploads_dir, temp_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_version = ThesisVersion(
        thesis_id=thesis_id,
        version_number=next_ver_num,
        file_ref=file_path,
        raw_text="",  # Extracted in ingestion pipeline
        diff_from_prev=None
    )
    db.add(new_version)
    
    # Update current version reference
    thesis.current_version_id = new_version.id
    
    # Log upload_version in the same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="upload_version",
        resource_type="ThesisVersion",
        resource_id=new_version.id,
        details={"filename": file.filename, "size": actual_size}
    )
    db.commit()
    db.refresh(new_version)
    
    # Run text ingestion pipeline
    await run_ingestion_pipeline(db, new_version, file_path)
    
    # Trigger full orchestrator update (incremental + notifications)
    await run_on_thesis_update(db, new_version)
    
    return new_version

@router.get("/{thesis_id}/citations", response_model=List[CitationResponse])
def get_thesis_citations(
    thesis_id: str,
    version_id: Optional[str] = None,
    current_user: User = Depends(require_feature_access("citation_verification", AccessLevel.VIEW)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
        
    assert_can_view_thesis(current_user, thesis)
    
    target_version_id = version_id or thesis.current_version_id
    if not target_version_id:
        return []
        
    citations = db.query(CitationRecord).filter(
        CitationRecord.version_id == target_version_id
    ).all()
    return citations

# ---------- FLAG RETRIEVAL WITH STUDENT PRIVACY MASKING ----------

@router.get("/{thesis_id}/flags", response_model=List[FlagResponse])
def get_thesis_flags(
    thesis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
    assert_can_view_thesis(current_user, thesis)
    
    if not thesis.current_version_id:
        return []
        
    snapshots = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == thesis.current_version_id
    ).all()
    snapshot_ids = [s.id for s in snapshots]
    
    flags = db.query(Flag).filter(Flag.snapshot_id.in_(snapshot_ids)).all()
    
    result_flags = []
    for flag in flags:
        msg = mask_plagiarism_flag_message(flag.message, current_user.role) if flag.type == FlagTypeEnum.plagiarism else flag.message
        result_flags.append({
            "id": flag.id,
            "snapshot_id": flag.snapshot_id,
            "type": flag.type,
            "severity": flag.severity,
            "message": msg,
            "evidence_excerpt": flag.evidence_excerpt,
            "page_ref": flag.page_ref,
            "resolved": flag.resolved,
            "resolved_by": flag.resolved_by,
            "resolved_at": flag.resolved_at,
            "created_at": flag.created_at
        })
            
    return result_flags

# ---------- EXTENDED LIVE DASHBOARD ENDPOINT ----------

@router.get("/{thesis_id}/dashboard", response_model=DashboardResponse)
def get_thesis_dashboard(
    thesis_id: str,
    current_user: User = Depends(require_feature_access("progress_tracker", AccessLevel.VIEW)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
        
    assert_can_view_thesis(current_user, thesis)
    
    # Default values if no version uploaded yet
    empty_res = {
        "overall_score": 0,
        "overall_quality_score": 0,
        "sections": {
            "introduction": {"score": 0, "flag_count": 0, "status": "ok"},
            "related_work": {"score": 0, "flag_count": 0, "status": "ok"},
            "methodology": {"score": 0, "flag_count": 0, "status": "ok"},
            "results": {"score": 0, "flag_count": 0, "status": "ok"},
            "discussion": {"score": 0, "flag_count": 0, "status": "ok"},
            "conclusion": {"score": 0, "flag_count": 0, "status": "ok"}
        },
        "citation_summary": {"total": 0, "verified": 0, "flagged": 0},
        "integrity_summary": {"plagiarism_flags": 0, "novelty_flags": 0},
        "recent_activity": [],
        "milestones": [
            {"title": "Draft Chapter 1 & 2", "status": "pending"},
            {"title": "Ingestion & Validation", "status": "pending"},
            {"title": "Advisor Link & Approval", "status": "done" if thesis.advisor_id else "pending"}
        ]
    }

    if not thesis.current_version_id:
        return empty_res

    # 1. Quality Scores & Flags
    latest_quality_snapshot = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == thesis.current_version_id,
        AnalysisSnapshot.tool_type == "quality_review"
    ).order_by(AnalysisSnapshot.generated_at.desc()).first()

    overall_quality_score = 0
    sections = {}
    
    if latest_quality_snapshot:
        overall_quality_score = latest_quality_snapshot.scores.get("overall", 0)
        sections_data = latest_quality_snapshot.scores.get("sections", {})
        for sec_name in ["introduction", "related_work", "methodology", "results", "discussion", "conclusion"]:
            data = sections_data.get(sec_name, {"score": 0, "flag_count": 0})
            score = data.get("score", 0)
            fc = data.get("flag_count", 0)
            
            # Status rules:
            if score >= 80 and fc == 0:
                sec_status = "ok"
            elif score < 60 or fc >= 2:
                sec_status = "critical"
            else:
                sec_status = "warning"
                
            sections[sec_name] = {
                "score": score,
                "flag_count": fc,
                "status": sec_status
            }
    else:
        # Populate empty sections if snapshot doesn't exist yet
        for sec_name in ["introduction", "related_work", "methodology", "results", "discussion", "conclusion"]:
            sections[sec_name] = {"score": 0, "flag_count": 0, "status": "ok"}

    # 2. Citation Summary
    citations = db.query(CitationRecord).filter(
        CitationRecord.version_id == thesis.current_version_id
    ).all()
    
    # A citation is flagged if it fails verification on any validation check
    flagged_citations = [c for c in citations if not c.exists_in_bib or not c.format_ok or c.supports_claim == "refutes"]
    total_citations = len(citations)
    flagged_count = len(flagged_citations)
    verified_count = total_citations - flagged_count

    # 3. Integrity Summary (plagiarism & novelty flags count)
    snapshots = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == thesis.current_version_id
    ).all()
    snapshot_ids = [s.id for s in snapshots]
    
    plagiarism_flags_count = db.query(Flag).filter(
        Flag.snapshot_id.in_(snapshot_ids),
        Flag.type == FlagTypeEnum.plagiarism
    ).count()

    novelty_flags_count = db.query(Flag).filter(
        Flag.snapshot_id.in_(snapshot_ids),
        Flag.type == FlagTypeEnum.novelty
    ).count()

    # 4. Recent Activity (last 3 versions)
    versions = db.query(ThesisVersion).filter(
        ThesisVersion.thesis_id == thesis.id
    ).order_by(ThesisVersion.version_number.desc()).limit(3).all()
    
    recent_activity = []
    for ver in versions:
        ver_snapshots = db.query(AnalysisSnapshot).filter(AnalysisSnapshot.version_id == ver.id).all()
        ver_snap_ids = [vs.id for vs in ver_snapshots]
        flags_count = db.query(Flag).filter(Flag.snapshot_id.in_(ver_snap_ids)).count()
        recent_activity.append({
            "version": ver.version_number,
            "uploaded_at": ver.uploaded_at,
            "new_flags_count": flags_count
        })

    # 5. Milestones
    milestones = [
        {"title": "Draft Chapter 1 & 2", "status": "done"},
        {"title": "Ingestion & Validation", "status": "done"},
        {"title": "Advisor Link & Approval", "status": "done" if thesis.advisor_id else "pending"}
    ]

    return {
        "overall_score": overall_quality_score,
        "overall_quality_score": overall_quality_score,
        "sections": sections,
        "citation_summary": {
            "total": total_citations,
            "verified": verified_count,
            "flagged": flagged_count
        },
        "integrity_summary": {
            "plagiarism_flags": plagiarism_flags_count,
            "novelty_flags": novelty_flags_count
        },
        "recent_activity": recent_activity,
        "milestones": milestones
    }

@router.post("/{thesis_id}/quality-review/trigger", status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour", key_func=get_user_key)
def trigger_quality_review(
    request: Request,
    thesis_id: str,
    current_user: User = Depends(require_feature_access("quality_review", AccessLevel.FULL)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
        
    assert_can_view_thesis(current_user, thesis)
    
    if not thesis.current_version_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No versions uploaded yet"
        )
        
    version = db.query(ThesisVersion).filter(ThesisVersion.id == thesis.current_version_id).first()
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current thesis version not found"
        )
        
    from app.services.quality import run_quality_review_pipeline
    snapshot = run_quality_review_pipeline(db, version)
    
    log_audit_action(
        db,
        user_id=current_user.id,
        action="trigger_pipeline",
        resource_type="AnalysisSnapshot",
        resource_id=snapshot.id,
        details={
            "pipeline_type": "quality_review",
            "flags_created": [f.id for f in snapshot.flags],
            "flag_count": len(snapshot.flags)
        }
    )
    db.commit()
    
    return snapshot.scores

# ---------- PHASE 2 PLAGIARISM & NOVELTY ENDPOINTS ----------

def mask_verdicts_for_student(verdicts: List[Dict[str, Any]], user_role: str) -> List[Dict[str, Any]]:
    if user_role == RoleEnum.student:
        masked = []
        for v in verdicts:
            v_copy = dict(v)
            if v_copy.get("source_type") == "internal_thesis":
                v_copy["source_title"] = "[RESTRICTED - Details restricted to advisor]"
                v_copy["source_thesis_id"] = None
            masked.append(v_copy)
        return masked
    return verdicts

@router.post("/{thesis_id}/plagiarism/trigger", status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour", key_func=get_user_key)
def trigger_plagiarism_review(
    request: Request,
    thesis_id: str,
    current_user: User = Depends(require_feature_access("plagiarism_monitor", AccessLevel.FULL)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
    assert_can_view_thesis(current_user, thesis)
    
    if not thesis.current_version_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No versions uploaded yet"
        )
        
    version = db.query(ThesisVersion).filter(ThesisVersion.id == thesis.current_version_id).first()
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current version not found"
        )
        
    snapshot = run_plagiarism_review_pipeline(db, version)
    
    # Log trigger_pipeline in same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="trigger_pipeline",
        resource_type="AnalysisSnapshot",
        resource_id=snapshot.id,
        details={
            "pipeline_type": "plagiarism",
            "flags_created": [f.id for f in snapshot.flags],
            "flag_count": len(snapshot.flags)
        }
    )
    db.commit()
    
    # Return masked verdicts if student
    verdicts = snapshot.scores.get("verdicts", [])
    masked_verdicts = mask_verdicts_for_student(verdicts, current_user.role)
    return {
        "flag_count": snapshot.scores.get("flag_count", 0),
        "verdicts": masked_verdicts
    }

@router.post("/{thesis_id}/novelty/trigger", status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour", key_func=get_user_key)
def trigger_novelty_review(
    request: Request,
    thesis_id: str,
    current_user: User = Depends(require_feature_access("novelty_detection", AccessLevel.FULL)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found"
        )
    assert_can_view_thesis(current_user, thesis)
    
    if not thesis.current_version_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No versions uploaded yet"
        )
        
    version = db.query(ThesisVersion).filter(ThesisVersion.id == thesis.current_version_id).first()
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current version not found"
        )
        
    snapshot = run_novelty_review_pipeline(db, version)
    
    # Log trigger_pipeline in same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="trigger_pipeline",
        resource_type="AnalysisSnapshot",
        resource_id=snapshot.id,
        details={
            "pipeline_type": "novelty",
            "flags_created": [f.id for f in snapshot.flags],
            "flag_count": len(snapshot.flags)
        }
    )
    db.commit()
    
    return snapshot.scores

# ---------- PHASE 3 NEW REMAINING TOOL ENDPOINTS ----------

@router.post("/{thesis_id}/reviewer-sim/session", response_model=ReviewerSimSessionResponse)
@limiter.limit("10/hour", key_func=get_user_key)
def create_reviewer_sim_session(
    request: Request,
    thesis_id: str,
    body: ReviewerSimSessionCreate,
    current_user: User = Depends(require_feature_access("reviewer_simulation", AccessLevel.FULL)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thesis not found")
        
    assert_can_view_thesis(current_user, thesis)
    
    # Mode-per-role validations
    if body.mode == "student_practice" and current_user.role != RoleEnum.student:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can create viva practice chat sessions."
        )
    if body.mode == "teacher_report" and current_user.role != RoleEnum.teacher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers can trigger reviewer report generation."
        )

    session = ReviewerSimSession(
        thesis_id=thesis_id,
        user_id=current_user.id,
        mode=body.mode,
        transcript=[],
        report=None
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Initialize the first examiner question for practice viva chat mode
    if body.mode == "student_practice":
        run_practice_viva_turn(db, session)
        
    # Log viva_session_created in same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="viva_session_created",
        resource_type="ReviewerSimSession",
        resource_id=session.id,
        details={"mode": body.mode}
    )
    db.commit()
    return session

@router.post("/{thesis_id}/reviewer-sim/session/{sess_id}/message")
@limiter.limit("10/hour", key_func=get_user_key)
def post_reviewer_sim_message(
    request: Request,
    thesis_id: str,
    sess_id: str,
    body: ReviewerSimMessageRequest,
    current_user: User = Depends(require_feature_access("reviewer_simulation", AccessLevel.FULL)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thesis not found")
        
    assert_can_view_thesis(current_user, thesis)
    
    session = db.query(ReviewerSimSession).filter(
        ReviewerSimSession.id == sess_id,
        ReviewerSimSession.thesis_id == thesis_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
    # Security: must own the session to post messages
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")
        
    if session.mode != "student_practice":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Can only post messages on student viva practice sessions.")

    result = run_practice_viva_turn(db, session, user_reply=body.message)
    
    # Log viva_message_sent in same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="viva_message_sent",
        resource_type="ReviewerSimSession",
        resource_id=session.id,
        details={"referenced_flag_id": result.get("referenced_flag_id")}
    )
    db.commit()
    return result

@router.post("/{thesis_id}/reviewer-sim/report")
@limiter.limit("10/hour", key_func=get_user_key)
def trigger_reviewer_report(
    request: Request,
    thesis_id: str,
    current_user: User = Depends(require_feature_access("reviewer_simulation", AccessLevel.FULL)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thesis not found")
        
    assert_can_view_thesis(current_user, thesis)
    
    # Mode-per-role security block (Only teachers can request Reports)
    if current_user.role != RoleEnum.teacher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers can generate mock examiner reports."
        )

    # Find or create a teacher_report session
    session = db.query(ReviewerSimSession).filter(
        ReviewerSimSession.thesis_id == thesis_id,
        ReviewerSimSession.user_id == current_user.id,
        ReviewerSimSession.mode == "teacher_report"
    ).first()
    
    if not session:
        session = ReviewerSimSession(
            thesis_id=thesis_id,
            user_id=current_user.id,
            mode="teacher_report",
            transcript=None,
            report=None
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
    report = generate_teacher_viva_report(db, session)
    
    # Log trigger_reviewer_report in same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="trigger_reviewer_report",
        resource_type="ReviewerSimSession",
        resource_id=session.id
    )
    db.commit()
    return report

@router.post("/{thesis_id}/literature-review/draft")
@limiter.limit("10/hour", key_func=get_user_key)
def trigger_literature_review_draft(
    request: Request,
    thesis_id: str,
    body: LiteratureReviewRequest,
    current_user: User = Depends(require_feature_access("literature_review", AccessLevel.FULL)),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thesis not found")
        
    # Only student owner can request paper fetches & drafts
    assert_owns_thesis(current_user, thesis)

    if not thesis.current_version_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No versions uploaded yet.")
        
    version = db.query(ThesisVersion).filter(ThesisVersion.id == thesis.current_version_id).first()
    if not version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Current version not found")

    result = run_literature_clustering_and_synthesis(db, version)
    
    # Log trigger_literature_review in same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="trigger_literature_review",
        resource_type="ThesisVersion",
        resource_id=version.id,
        details={"cluster_count": len(result.get("clusters", []))}
    )
    db.commit()
    return result

# Map FlagTypeEnum → permission matrix feature key for per-tool flag resolution.
# Flag resolution requires FULL access on the flag's underlying tool, not a blanket
# progress_tracker check. This preserves the original matrix intent: teachers with VIEW
# on citation_verification can see citation flags but not resolve them; teachers with FULL
# on plagiarism_monitor can resolve plagiarism flags.
FLAG_TYPE_TO_FEATURE = {
    FlagTypeEnum.citation: "citation_verification",
    FlagTypeEnum.plagiarism: "plagiarism_monitor",
    FlagTypeEnum.quality: "quality_review",
    FlagTypeEnum.novelty: "novelty_detection",
}

@router.post("/{thesis_id}/flags/{flag_id}/resolve")
def resolve_flag(
    thesis_id: str,
    flag_id: str,
    resolve: bool = True,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thesis not found")
    assert_can_view_thesis(current_user, thesis)

    # Only teachers (advisors) or admins can resolve flags
    if current_user.role not in [RoleEnum.teacher, RoleEnum.admin]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only advisors or admins can resolve/dismiss flags."
        )

    flag = db.query(Flag).filter(Flag.id == flag_id).first()
    if not flag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flag not found")

    # Per-tool permission check: teacher must have FULL on the flag's underlying tool
    required_feature = FLAG_TYPE_TO_FEATURE.get(flag.type)
    if required_feature and not can(current_user.role.value, required_feature, AccessLevel.FULL):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You have VIEW access on {required_feature}, which does not permit flag resolution. "
                   f"Only users with FULL access on {required_feature} can resolve {flag.type.value} flags."
        )

    old_resolved = flag.resolved
    flag.resolved = resolve
    if resolve:
        flag.resolved_by = current_user.id
        flag.resolved_at = datetime.utcnow()
    else:
        flag.resolved_by = None
        flag.resolved_at = None

    # Write audit log in the same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="flag_resolved",
        resource_type="Flag",
        resource_id=flag.id,
        details={"before": old_resolved, "after": resolve, "flag_type": flag.type.value}
    )
    db.commit()
    db.refresh(flag)
    return {"message": "Flag resolution status updated", "flag_id": flag.id, "resolved": flag.resolved}

@router.post("/{thesis_id}/status")
def update_thesis_status(
    thesis_id: str,
    body: ThesisStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if not thesis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thesis not found")
    assert_can_view_thesis(current_user, thesis)

    if current_user.role == RoleEnum.student:
        if current_user.id != thesis.owner_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your thesis")
        if body.status not in ["draft", "in_review"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Students can only set thesis status to draft or in_review."
            )
    elif current_user.role == RoleEnum.teacher:
        if thesis.advisor_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not supervise this student")
    elif current_user.role != RoleEnum.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    try:
        new_status = ThesisStatusEnum(body.status)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status value")

    old_status = thesis.status
    thesis.status = new_status

    # Log thesis status change in the same transaction
    log_audit_action(
        db,
        user_id=current_user.id,
        action="thesis_status_changed",
        resource_type="Thesis",
        resource_id=thesis.id,
        details={"before": old_status.value if old_status else None, "after": new_status.value}
    )
    db.commit()
    db.refresh(thesis)
    return {"message": "Thesis status updated", "thesis_id": thesis.id, "status": thesis.status.value}
