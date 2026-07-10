import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import (
    User, Thesis, ThesisVersion, Flag, FlagTypeEnum, AnalysisSnapshot,
    SeverityEnum, RoleEnum, AuditLog, ReviewerSimSession
)

def test_upload_file_size_limit(client: TestClient, db: Session):
    """Test that files > 10MB are rejected with 400."""
    # Register student
    reg = client.post("/auth/register", json={
        "email": "size@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Size Student",
        "university": "Test Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert reg.status_code == 201, f"Registration failed: {reg.text}"

    s_login = client.post("/auth/login", data={"username": "size@uni.edu", "password": "pass"})
    assert s_login.status_code == 200, f"Login failed: {s_login.text}"
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}

    # Setup thesis
    student = db.query(User).filter(User.email == "size@uni.edu").first()
    thesis = Thesis(owner_id=student.id, title="Size Test Thesis")
    db.add(thesis)
    db.commit()

    # Upload file > 10MB (11MB of mock bytes) -> must be rejected before PDF parsing
    large_file = io.BytesIO(b"a" * (11 * 1024 * 1024))
    res = client.post(
        f"/theses/{thesis.id}/versions",
        files={"file": ("thesis.pdf", large_file, "application/pdf")},
        headers=s_headers
    )
    assert res.status_code == 400
    assert "exceeds maximum limit of 10MB" in res.json()["detail"]

def test_db_integrity_exception_handler(client: TestClient, db: Session):
    """Test that duplicate email registration returns clean 400."""
    # Register a user first (valid student with all required fields)
    res1 = client.post("/auth/register", json={
        "email": "dup@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Dup Student",
        "university": "Test Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert res1.status_code == 201, f"First registration failed: {res1.text}"

    # Register duplicate email -> caught by explicit check in register endpoint
    res2 = client.post("/auth/register", json={
        "email": "dup@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Dup Student Two",
        "university": "Test Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert res2.status_code == 400
    assert "already registered" in res2.json()["detail"].lower()

def test_thesis_status_change_and_audit(client: TestClient, db: Session):
    """Test thesis status transitions, role restrictions, and audit logging."""
    # Register student & teacher
    r1 = client.post("/auth/register", json={
        "email": "stud_stat@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Status Student",
        "university": "Test Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert r1.status_code == 201, f"Student registration failed: {r1.text}"

    r2 = client.post("/auth/register", json={
        "email": "teach_stat@uni.edu",
        "password": "pass",
        "role": "teacher",
        "name": "Status Teacher",
        "university": "Test Uni",
        "department": "CS",
        "designation": "Professor"
    })
    assert r2.status_code == 201, f"Teacher registration failed: {r2.text}"
    
    s_login = client.post("/auth/login", data={"username": "stud_stat@uni.edu", "password": "pass"})
    assert s_login.status_code == 200
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}
    
    t_login = client.post("/auth/login", data={"username": "teach_stat@uni.edu", "password": "pass"})
    assert t_login.status_code == 200
    t_headers = {"Authorization": f"Bearer {t_login.json()['access_token']}"}

    student = db.query(User).filter(User.email == "stud_stat@uni.edu").first()
    teacher = db.query(User).filter(User.email == "teach_stat@uni.edu").first()

    thesis = Thesis(owner_id=student.id, advisor_id=teacher.id, title="Status Thesis")
    db.add(thesis)
    db.commit()

    # 1. Student changes status to "in_review" -> Allowed (students can set draft or in_review)
    res_s = client.post(f"/theses/{thesis.id}/status", json={"status": "in_review"}, headers=s_headers)
    assert res_s.status_code == 200, f"Student status change failed: {res_s.text}"
    assert res_s.json()["status"] == "in_review"

    # Verify audit log exists
    audit_entry = db.query(AuditLog).filter(
        AuditLog.action == "thesis_status_changed",
        AuditLog.resource_id == thesis.id
    ).first()
    assert audit_entry is not None
    assert audit_entry.user_id == student.id
    assert audit_entry.details["before"] == "draft"
    assert audit_entry.details["after"] == "in_review"

    # 2. Student tries to set "defended" -> Forbidden (students can only set draft/in_review)
    res_s_bad = client.post(f"/theses/{thesis.id}/status", json={"status": "defended"}, headers=s_headers)
    assert res_s_bad.status_code == 403

    # 3. Teacher sets to "defended" -> Allowed
    res_t = client.post(f"/theses/{thesis.id}/status", json={"status": "defended"}, headers=t_headers)
    assert res_t.status_code == 200, f"Teacher status change failed: {res_t.text}"

    # Verify new audit log exists
    audit_entry_2 = db.query(AuditLog).filter(
        AuditLog.action == "thesis_status_changed",
        AuditLog.resource_id == thesis.id
    ).order_by(AuditLog.created_at.desc()).first()
    assert audit_entry_2 is not None
    assert audit_entry_2.user_id == teacher.id
    assert audit_entry_2.details["before"] == "in_review"
    assert audit_entry_2.details["after"] == "defended"

def test_flag_resolution_per_tool_permission(client: TestClient, db: Session):
    """Test per-tool flag resolution: teacher can resolve quality flags (FULL_PLUS_COMMENT)
    but cannot resolve citation flags (only VIEW on citation_verification)."""
    # Setup users
    r1 = client.post("/auth/register", json={
        "email": "stud_flag@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Flag Student",
        "university": "Test Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert r1.status_code == 201

    r2 = client.post("/auth/register", json={
        "email": "teach_flag@uni.edu",
        "password": "pass",
        "role": "teacher",
        "name": "Flag Teacher",
        "university": "Test Uni",
        "department": "CS",
        "designation": "Professor"
    })
    assert r2.status_code == 201
    
    s_login = client.post("/auth/login", data={"username": "stud_flag@uni.edu", "password": "pass"})
    assert s_login.status_code == 200
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}
    
    t_login = client.post("/auth/login", data={"username": "teach_flag@uni.edu", "password": "pass"})
    assert t_login.status_code == 200
    t_headers = {"Authorization": f"Bearer {t_login.json()['access_token']}"}

    student = db.query(User).filter(User.email == "stud_flag@uni.edu").first()
    teacher = db.query(User).filter(User.email == "teach_flag@uni.edu").first()

    # Setup thesis (student owns, teacher advises)
    thesis = Thesis(owner_id=student.id, advisor_id=teacher.id, title="Flag Thesis")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(thesis_id=thesis.id, version_number=1)
    db.add(version)
    db.commit()

    snapshot = AnalysisSnapshot(version_id=version.id, tool_type="quality_review", scores={"overall": 70})
    db.add(snapshot)
    db.commit()

    # --- Quality flag: teacher has FULL_PLUS_COMMENT on quality_review → CAN resolve ---
    quality_flag = Flag(
        snapshot_id=snapshot.id,
        type=FlagTypeEnum.quality,
        severity=SeverityEnum.moderate,
        message="Weak intro",
        evidence_excerpt="Intro content here is too brief",
        resolved=False
    )
    db.add(quality_flag)
    db.commit()

    # Student trying to resolve → Forbidden (not a teacher/admin)
    res_s = client.post(f"/theses/{thesis.id}/flags/{quality_flag.id}/resolve", json={}, headers=s_headers)
    assert res_s.status_code == 403

    # Teacher resolves quality flag → Success (FULL_PLUS_COMMENT ≥ FULL)
    res_t = client.post(f"/theses/{thesis.id}/flags/{quality_flag.id}/resolve?resolve=true", headers=t_headers)
    assert res_t.status_code == 200, f"Teacher quality flag resolve failed: {res_t.text}"
    assert res_t.json()["resolved"] is True

    # Verify flag state and audit log
    db.refresh(quality_flag)
    assert quality_flag.resolved is True
    assert quality_flag.resolved_by == teacher.id

    audit = db.query(AuditLog).filter(
        AuditLog.action == "flag_resolved",
        AuditLog.resource_id == quality_flag.id
    ).first()
    assert audit is not None
    assert audit.user_id == teacher.id
    assert audit.details["before"] is False
    assert audit.details["after"] is True
    assert audit.details["flag_type"] == "quality"

    # --- Citation flag: teacher has VIEW on citation_verification → CANNOT resolve ---
    citation_flag = Flag(
        snapshot_id=snapshot.id,
        type=FlagTypeEnum.citation,
        severity=SeverityEnum.moderate,
        message="Citation not found in bibliography",
        evidence_excerpt="Smith et al. (2023) is referenced but not in bibliography",
        resolved=False
    )
    db.add(citation_flag)
    db.commit()

    # Teacher tries to resolve citation flag → Forbidden (VIEW on citation_verification)
    res_t_cit = client.post(f"/theses/{thesis.id}/flags/{citation_flag.id}/resolve?resolve=true", headers=t_headers)
    assert res_t_cit.status_code == 403, f"Expected 403 for citation flag, got: {res_t_cit.status_code} {res_t_cit.text}"
    assert "citation_verification" in res_t_cit.json()["detail"]
    assert "VIEW" in res_t_cit.json()["detail"]

    # Verify citation flag remains unresolved
    db.refresh(citation_flag)
    assert citation_flag.resolved is False

def test_pipeline_trigger_audit_logging(client: TestClient, db: Session):
    """Test that pipeline triggers write audit logs with flag details."""
    # Setup users
    r1 = client.post("/auth/register", json={
        "email": "pipe_stud@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Pipeline Student",
        "university": "Test Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert r1.status_code == 201

    s_login = client.post("/auth/login", data={"username": "pipe_stud@uni.edu", "password": "pass"})
    assert s_login.status_code == 200
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}

    student = db.query(User).filter(User.email == "pipe_stud@uni.edu").first()

    thesis = Thesis(owner_id=student.id, title="Pipeline Audit Thesis")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(thesis_id=thesis.id, version_number=1)
    db.add(version)
    db.commit()

    thesis.current_version_id = version.id
    db.commit()

    # Trigger Quality Review
    res = client.post(f"/theses/{thesis.id}/quality-review/trigger", headers=s_headers)
    assert res.status_code == 201, f"Trigger failed: {res.text}"

    # Verify trigger audit logs details contains flags list and count
    audit = db.query(AuditLog).filter(
        AuditLog.action == "trigger_pipeline",
        AuditLog.user_id == student.id
    ).first()
    assert audit is not None
    assert audit.details["pipeline_type"] == "quality_review"
    assert "flags_created" in audit.details
    assert "flag_count" in audit.details
    assert isinstance(audit.details["flags_created"], list)

def test_internal_reindex_endpoint(client: TestClient):
    """Test that /internal/reindex requires correct auth token."""
    # 1. Reindex with missing header token -> 401
    res_bad = client.post("/internal/reindex")
    assert res_bad.status_code == 401

    # 2. Reindex with incorrect token -> 401
    res_wrong = client.post("/internal/reindex", headers={"X-Internal-Token": "badtoken"})
    assert res_wrong.status_code == 401

    # 3. Reindex with correct token -> 200
    from app.config import settings
    res_ok = client.post("/internal/reindex", headers={"X-Internal-Token": settings.INTERNAL_SECRET_TOKEN})
    assert res_ok.status_code == 200
    assert "Reindex executed successfully" in res_ok.json()["message"]

def test_rate_limiting_auth(client: TestClient):
    """Test that auth login is rate-limited to 5/minute (IP-based)."""
    from app.limiter import limiter
    limiter.enabled = True
    try:
        for _ in range(5):
            res = client.post("/auth/login", data={"username": "notexist_limit@uni.edu", "password": "any"})
            assert res.status_code == 401

        res_limit = client.post("/auth/login", data={"username": "notexist_limit@uni.edu", "password": "any"})
        assert res_limit.status_code == 429
    finally:
        limiter.enabled = False

def test_rate_limiting_llm_trigger(client: TestClient, db: Session):
    """Test that LLM-trigger endpoints (quality-review/trigger) are rate-limited
    to 10/hour per user. Verifies the 11th request returns 429."""
    from app.limiter import limiter

    # Setup student with thesis + version
    r1 = client.post("/auth/register", json={
        "email": "ratelim_stud@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Rate Student",
        "university": "Test Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert r1.status_code == 201

    s_login = client.post("/auth/login", data={"username": "ratelim_stud@uni.edu", "password": "pass"})
    assert s_login.status_code == 200
    s_headers = {"Authorization": f"Bearer {s_login.json()['access_token']}"}

    student = db.query(User).filter(User.email == "ratelim_stud@uni.edu").first()

    thesis = Thesis(owner_id=student.id, title="Rate Limit Thesis")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(thesis_id=thesis.id, version_number=1)
    db.add(version)
    db.commit()

    thesis.current_version_id = version.id
    db.commit()

    limiter.enabled = True
    try:
        # Make 10 successful trigger requests (the per-hour limit)
        for i in range(10):
            res = client.post(f"/theses/{thesis.id}/quality-review/trigger", headers=s_headers)
            assert res.status_code == 201, f"Request {i+1} failed unexpectedly: {res.status_code} {res.text}"

        # The 11th request should hit the 10/hour user-based rate limit
        res_limit = client.post(f"/theses/{thesis.id}/quality-review/trigger", headers=s_headers)
        assert res_limit.status_code == 429, f"Expected 429 on 11th trigger, got: {res_limit.status_code} {res_limit.text}"
    finally:
        limiter.enabled = False
