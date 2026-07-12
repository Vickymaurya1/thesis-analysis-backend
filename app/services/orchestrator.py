import traceback
from sqlalchemy.orm import Session
from app.models import ThesisVersion, Flag, AnalysisSnapshot, Notification, SeverityEnum
from app.services.citation import run_citation_verification_pipeline
from app.services.quality import run_quality_review_pipeline
from app.services.plagiarism import run_plagiarism_review_pipeline
from app.services.novelty import run_novelty_review_pipeline

async def run_on_thesis_update(db: Session, version: ThesisVersion):
    # Run pipelines sequentially, each wrapped to prevent one failure from blocking the rest
    try:
        await run_citation_verification_pipeline(db, version)
        print(f"Orchestrator: Citation verification completed for version {version.id}")
    except Exception as e:
        print(f"Orchestrator: Citation verification FAILED for version {version.id}")
        traceback.print_exc()

    try:
        run_plagiarism_review_pipeline(db, version)
        print(f"Orchestrator: Plagiarism review completed for version {version.id}")
    except Exception as e:
        print(f"Orchestrator: Plagiarism review FAILED for version {version.id}")
        traceback.print_exc()

    try:
        run_quality_review_pipeline(db, version)
        print(f"Orchestrator: Quality review completed for version {version.id}")
    except Exception as e:
        print(f"Orchestrator: Quality review FAILED for version {version.id}")
        traceback.print_exc()

    try:
        run_novelty_review_pipeline(db, version)
        print(f"Orchestrator: Novelty review completed for version {version.id}")
    except Exception as e:
        print(f"Orchestrator: Novelty review FAILED for version {version.id}")
        traceback.print_exc()

    # Write notifications
    try:
        write_notifications_for_new_flags(db, version)
    except Exception as e:
        print(f"Orchestrator: Notification writing FAILED for version {version.id}")
        traceback.print_exc()

def write_notifications_for_new_flags(db: Session, version: ThesisVersion):
    thesis = version.thesis
    if not thesis:
        return

    from app.services.plagiarism import mask_plagiarism_flag_message
    from app.models import FlagTypeEnum

    # Find all flags created for this version
    flags = db.query(Flag).join(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == version.id
    ).all()

    notifications_to_add = []
    
    for flag in flags:
        # Student message masking for plagiarism flags
        student_flag_msg = flag.message
        if flag.type == FlagTypeEnum.plagiarism:
            student_flag_msg = mask_plagiarism_flag_message(flag.message, "student")

        # Determine routing based on severity
        if flag.severity == SeverityEnum.critical:
            # Immediate notification for student
            notifications_to_add.append(Notification(
                user_id=thesis.owner_id,
                type="critical",
                message=f"Critical issue flagged in version {version.version_number}: {student_flag_msg[:100]}",
                related_flag_id=flag.id,
                batched=False
            ))
            
            # Immediate notification for advisor (if linked)
            if thesis.advisor_id:
                notifications_to_add.append(Notification(
                    user_id=thesis.advisor_id,
                    type="critical",
                    message=f"Critical issue flagged in your student's thesis ('{thesis.title}'): {flag.message[:100]}",
                    related_flag_id=flag.id,
                    batched=False
                ))
        else:
            # Moderate/Low -> batched daily digest for student only
            notifications_to_add.append(Notification(
                user_id=thesis.owner_id,
                type="digest",
                message=f"Issue flagged in version {version.version_number}: {student_flag_msg[:100]}",
                related_flag_id=flag.id,
                batched=True
            ))

    if notifications_to_add:
        db.add_all(notifications_to_add)
        db.commit()
