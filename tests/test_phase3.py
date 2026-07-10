import pytest
import datetime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import (
    User, Thesis, ThesisVersion, CorpusDocument, CorpusSourceEnum,
    Flag, FlagTypeEnum, AnalysisSnapshot, SemanticScholarSearch,
    ReviewerSimSession, Notification, SeverityEnum, RoleEnum
)
from app.services.orchestrator import run_on_thesis_update

@pytest.mark.asyncio
async def test_incremental_re_analysis_caching(db: Session):
    student = User(email="inc@uni.edu", password_hash="hash", role=RoleEnum.student, name="Student")
    db.add(student)
    db.commit()

    thesis = Thesis(owner_id=student.id, title="Incremental Thesis", field="CS")
    db.add(thesis)
    db.commit()

    # Version 1: standard content
    v1 = ThesisVersion(
        thesis_id=thesis.id,
        version_number=1,
        raw_text="Introduction\nWe propose structured pruning mechanisms.\nMethodology\nWe train our models.",
        section_map={"introduction": [0, 40], "methodology": [40, 80]}
    )
    db.add(v1)
    db.commit()
    db.refresh(v1)

    from app.models import Chunk
    # Chunks for V1
    c1 = Chunk(version_id=v1.id, chunk_index=0, content="We propose structured pruning mechanisms.", section_label="introduction", embedding=[0.1]*1024)
    c2 = Chunk(version_id=v1.id, chunk_index=1, content="We train our models.", section_label="methodology", embedding=[0.1]*1024)
    db.add_all([c1, c2])
    db.commit()

    # Seed an external paper to be similarity matched during novelty checks
    peer = CorpusDocument(
        source_type=CorpusSourceEnum.external_paper,
        title="Reference Pruning Method",
        chunk_text="A structured attention head pruning approach.",
        embedding=[0.1]*1024
    )
    db.add(peer)
    db.commit()

    # Seed initial snapshots
    await run_on_thesis_update(db, v1)

    # Check that initial snapshot is stored
    snap_q1 = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == v1.id,
        AnalysisSnapshot.tool_type == "quality_review"
    ).first()
    assert snap_q1 is not None

    snap_n1 = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == v1.id,
        AnalysisSnapshot.tool_type == "novelty_detection"
    ).first()
    assert snap_n1 is not None

    # Version 2: Update ONLY methodology, leaving introduction unchanged!
    v2 = ThesisVersion(
        thesis_id=thesis.id,
        version_number=2,
        raw_text="Introduction\nWe propose structured pruning mechanisms.\nMethodology\nWe train our models with optimized parameters.",
        section_map={"introduction": [0, 40], "methodology": [40, 95]},
        diff_from_prev={
            "changed_sections": ["methodology"],
            "changed_paragraphs": ["We train our models with optimized parameters."]
        }
    )
    db.add(v2)
    db.commit()
    db.refresh(v2)

    c3 = Chunk(version_id=v2.id, chunk_index=0, content="We propose structured pruning mechanisms.", section_label="introduction", embedding=[0.1]*1024)
    c4 = Chunk(version_id=v2.id, chunk_index=1, content="We train our models with optimized parameters.", section_label="methodology", embedding=[0.1]*1024)
    db.add_all([c3, c4])
    db.commit()

    # Trigger orchestrator
    await run_on_thesis_update(db, v2)

    snap_q2 = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == v2.id,
        AnalysisSnapshot.tool_type == "quality_review"
    ).first()
    assert snap_q2 is not None

    # Verify that the unchanged "introduction" section reused the previous version's score
    assert snap_q2.scores["sections"]["introduction"]["score"] == snap_q1.scores["sections"]["introduction"]["score"]

    # Verify novelty detection reused the cached claims (since the claim chunk is identical)
    snap_n2 = db.query(AnalysisSnapshot).filter(
        AnalysisSnapshot.version_id == v2.id,
        AnalysisSnapshot.tool_type == "novelty_detection"
    ).first()
    assert snap_n2 is not None
    assert snap_n2.scores["novelty_score"] == snap_n1.scores["novelty_score"]


def test_notification_routing(db: Session):
    student = User(email="ns@uni.edu", password_hash="hash", role=RoleEnum.student, name="Student")
    advisor = User(email="na@uni.edu", password_hash="hash", role=RoleEnum.teacher, name="Advisor")
    db.add_all([student, advisor])
    db.commit()

    thesis = Thesis(owner_id=student.id, advisor_id=advisor.id, title="Alert Thesis")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(thesis_id=thesis.id, version_number=1)
    db.add(version)
    db.commit()

    # Seed snap
    snapshot = AnalysisSnapshot(version_id=version.id, tool_type="quality_review")
    db.add(snapshot)
    db.commit()

    # Flag 1: Critical -> Immediate
    flag_c = Flag(
        snapshot_id=snapshot.id,
        type=FlagTypeEnum.quality,
        severity=SeverityEnum.critical,
        message="Critical issue.",
        evidence_excerpt="Sample critical text"
    )
    # Flag 2: Moderate -> Batched
    flag_m = Flag(
        snapshot_id=snapshot.id,
        type=FlagTypeEnum.quality,
        severity=SeverityEnum.moderate,
        message="Moderate issue.",
        evidence_excerpt="Sample moderate text"
    )
    db.add_all([flag_c, flag_m])
    db.commit()

    from app.services.orchestrator import write_notifications_for_new_flags
    write_notifications_for_new_flags(db, version)

    # Verify critical generated 2 immediate notifications (student + advisor)
    student_notifs = db.query(Notification).filter(
        Notification.user_id == student.id,
        Notification.type == "critical"
    ).all()
    assert len(student_notifs) == 1
    assert student_notifs[0].batched is False

    advisor_notifs = db.query(Notification).filter(
        Notification.user_id == advisor.id,
        Notification.type == "critical"
    ).all()
    assert len(advisor_notifs) == 1
    assert advisor_notifs[0].batched is False

    # Verify moderate generated 1 batched notification for student only
    student_digests = db.query(Notification).filter(
        Notification.user_id == student.id,
        Notification.type == "digest"
    ).all()
    assert len(student_digests) == 1
    assert student_digests[0].batched is True

    # Verify advisor got 0 digest notifications
    advisor_digests = db.query(Notification).filter(
        Notification.user_id == advisor.id,
        Notification.type == "digest"
    ).all()
    assert len(advisor_digests) == 0


def test_reviewer_simulation_endpoints(client: TestClient, db: Session):
    # Setup roles
    s_res = client.post("/auth/register", json={
        "email": "student_sim@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Student",
        "university": "Uni",
        "department": "CS",
        "degree": "BTech",
        "field_of_study": "CS"
    })
    student_id = s_res.json()["id"]

    t_res = client.post("/auth/register", json={
        "email": "teacher_sim@uni.edu",
        "password": "pass",
        "role": "teacher",
        "name": "Teacher",
        "university": "Uni",
        "department": "CS",
        "designation": "Professor"
    })
    teacher_id = t_res.json()["id"]

    # Logins
    s_login = client.post("/auth/login", data={"username": "student_sim@uni.edu", "password": "pass"})
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}

    t_login = client.post("/auth/login", data={"username": "teacher_sim@uni.edu", "password": "pass"})
    t_headers = {"Authorization": f"Bearer {t_login.json()['access_token']}"}

    # Setup thesis
    thesis = Thesis(owner_id=student_id, advisor_id=teacher_id, title="Sim Thesis")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(thesis_id=thesis.id, version_number=1, raw_text="Introduction content.")
    db.add(version)
    db.commit()
    
    thesis.current_version_id = version.id
    db.commit()

    # 1. Student Viva Practice endpoint tests
    # Verification: Student trying to start a teacher report -> Forbidden
    r_bad = client.post(
        f"/theses/{thesis.id}/reviewer-sim/session",
        json={"mode": "teacher_report"},
        headers=s_headers
    )
    assert r_bad.status_code == 403

    # Student starting student chat mode -> Success
    r_ok = client.post(
        f"/theses/{thesis.id}/reviewer-sim/session",
        json={"mode": "student_practice"},
        headers=s_headers
    )
    print("DEBUG REVIEWER SIM SESS:", r_ok.status_code, r_ok.text)
    assert r_ok.status_code == 200
    sess_id = r_ok.json()["id"]
    # The first question is automatically populated in mock mode
    assert len(r_ok.json()["transcript"]) == 1
    assert "justify" in r_ok.json()["transcript"][0]["content"]

    # Student posts a response
    r_msg = client.post(
        f"/theses/{thesis.id}/reviewer-sim/session/{sess_id}/message",
        json={"message": "I chose structured pruning because of inference gains."},
        headers=s_headers
    )
    assert r_msg.status_code == 200
    assert "accuracy" in r_msg.json()["examiner_message"]

    # 2. Teacher Mock Report endpoint tests
    # Verification: Student trying to request a Report -> Forbidden
    r_rep_bad = client.post(f"/theses/{thesis.id}/reviewer-sim/report", headers=s_headers)
    assert r_rep_bad.status_code == 403

    # Teacher requests report -> Success
    r_rep_ok = client.post(f"/theses/{thesis.id}/reviewer-sim/report", headers=t_headers)
    assert r_rep_ok.status_code == 200
    assert "overall_assessment" in r_rep_ok.json()
    assert len(r_rep_ok.json()["strengths"]) > 0


def test_literature_review_assistant(client: TestClient, db: Session):
    s_res = client.post("/auth/register", json={
        "email": "student_lit@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Student",
        "university": "Uni",
        "department": "CS",
        "degree": "BTech",
        "field_of_study": "CS"
    })
    student_id = s_res.json()["id"]

    s_login = client.post("/auth/login", data={"username": "student_lit@uni.edu", "password": "pass"})
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}

    thesis = Thesis(owner_id=student_id, title="Lit Thesis", field="CS")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(
        thesis_id=thesis.id,
        version_number=1,
        raw_text="Introduction\nWe propose a novel attention head pruning mechanism.",
        section_map={"introduction": [0, 60]}
    )
    db.add(version)
    db.commit()
    thesis.current_version_id = version.id
    db.commit()

    from app.models import Chunk
    chunk = Chunk(
        version_id=version.id,
        chunk_index=0,
        content="We propose a novel attention head pruning mechanism.",
        section_label="introduction",
        embedding=[0.1]*1024
    )
    db.add(chunk)
    db.commit()

    # Seed external papers in CorpusDocument
    # Paper A: similarity 0.95 to thesis vector (our seed)
    p_a = CorpusDocument(
        source_type=CorpusSourceEnum.external_paper,
        external_id="paper_a",
        title="Fast attention head pruning",
        chunk_text="Pruning attention heads dynamically.",
        embedding=[0.1]*1024
    )
    # Paper B: similarity 0.98 to Paper A (should join Paper A's cluster)
    p_b = CorpusDocument(
        source_type=CorpusSourceEnum.external_paper,
        external_id="paper_b",
        title="Pruning Transformers at Scale",
        chunk_text="Transformer model compression via structured pruning.",
        embedding=[0.1]*1024
    )
    db.add_all([p_a, p_b])
    db.commit()

    # Trigger literature review synthesis
    r_lit = client.post(
        f"/theses/{thesis.id}/literature-review/draft",
        json={"topics": ["pruning"]},
        headers=s_headers
    )
    assert r_lit.status_code == 200
    clusters = r_lit.json()["clusters"]
    
    # We should have exactly 1 cluster since Paper B is near-identical to Paper A and got consumed
    assert len(clusters) == 1
    assert len(clusters[0]["papers"]) == 2
    assert "synthesis_paragraph" in clusters[0]
    assert "Fast attention head pruning" in clusters[0]["synthesis_paragraph"]


def test_dashboard_visual_metrics(client: TestClient, db: Session):
    s_res = client.post("/auth/register", json={
        "email": "student_dash@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Student",
        "university": "Uni",
        "department": "CS",
        "degree": "BTech",
        "field_of_study": "CS"
    })
    student_id = s_res.json()["id"]

    s_login = client.post("/auth/login", data={"username": "student_dash@uni.edu", "password": "pass"})
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}

    thesis = Thesis(owner_id=student_id, title="Dashboard Thesis")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(thesis_id=thesis.id, version_number=1)
    db.add(version)
    db.commit()
    thesis.current_version_id = version.id
    db.commit()

    # Add Quality Review Snapshot
    snapshot = AnalysisSnapshot(
        version_id=version.id,
        tool_type="quality_review",
        scores={
            "overall": 82,
            "sections": {
                "introduction": {"score": 85, "flag_count": 0},
                "methodology": {"score": 58, "flag_count": 2}
            }
        }
    )
    db.add(snapshot)
    db.commit()

    # Add a mock plagiarism flag to check metrics aggregation
    flag = Flag(
        snapshot_id=snapshot.id,
        type=FlagTypeEnum.plagiarism,
        severity=SeverityEnum.critical,
        message="Copy detected",
        evidence_excerpt="Copied snippet"
    )
    db.add(flag)
    db.commit()

    # Fetch dashboard
    res = client.get(f"/theses/{thesis.id}/dashboard", headers=s_headers)
    assert res.status_code == 200
    data = res.json()
    
    # Assert keys match Phase 3 specification
    assert data["overall_quality_score"] == 82
    assert data["sections"]["introduction"]["status"] == "ok"
    assert data["sections"]["methodology"]["status"] == "critical"
    assert data["integrity_summary"]["plagiarism_flags"] == 1
    assert len(data["recent_activity"]) == 1
    assert data["recent_activity"][0]["version"] == 1
    assert len(data["milestones"]) > 0

def test_reviewer_simulation_cross_role_access(client: TestClient, db: Session):
    # Setup student
    s_res = client.post("/auth/register", json={
        "email": "student_sim_sec@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Student",
        "university": "Uni",
        "department": "CS",
        "degree": "BTech",
        "field_of_study": "CS"
    })
    student_id = s_res.json()["id"]

    # Setup teacher
    t_res = client.post("/auth/register", json={
        "email": "teacher_sim_sec@uni.edu",
        "password": "pass",
        "role": "teacher",
        "name": "Teacher",
        "university": "Uni",
        "department": "CS",
        "designation": "Professor"
    })
    teacher_id = t_res.json()["id"]

    s_login = client.post("/auth/login", data={"username": "student_sim_sec@uni.edu", "password": "pass"})
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}

    t_login = client.post("/auth/login", data={"username": "teacher_sim_sec@uni.edu", "password": "pass"})
    t_headers = {"Authorization": f"Bearer {t_login.json()['access_token']}"}

    thesis = Thesis(owner_id=student_id, advisor_id=teacher_id, title="Security Thesis")
    db.add(thesis)
    db.commit()

    # 1. Student trying to generate teacher report -> 403 Forbidden
    res1 = client.post(f"/theses/{thesis.id}/reviewer-sim/report", headers=s_headers)
    assert res1.status_code == 403

    # 2. Teacher trying to create student practice viva -> 403 Forbidden
    res2 = client.post(
        f"/theses/{thesis.id}/reviewer-sim/session",
        json={"mode": "student_practice"},
        headers=t_headers
    )
    assert res2.status_code == 403


def test_internal_send_digests_security(client: TestClient, db: Session):
    from app.config import settings

    # 1. Access without token -> 401
    res1 = client.post("/internal/send-digests")
    assert res1.status_code == 401

    # 2. Access with wrong token -> 401
    res2 = client.post(
        "/internal/send-digests",
        headers={"X-Internal-Token": "wrongtoken"}
    )
    assert res2.status_code == 401

    # 3. Access with correct token -> 200
    res3 = client.post(
        "/internal/send-digests",
        headers={"X-Internal-Token": settings.INTERNAL_SECRET_TOKEN}
    )
    assert res3.status_code == 200
    assert "sent_digests_count" in res3.json()
