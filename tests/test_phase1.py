import pytest
import os
import uuid
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import User, Thesis, ThesisVersion, AdvisorLink, Flag, RoleEnum, SupportEnum, SeverityEnum, FlagTypeEnum
from app.config import settings
from app.services.quality import run_quality_review_pipeline
from app.services.ingestion import detect_sections_regex, detect_sections_llm, extract_candidate_headings, chunk_section_text, get_embeddings

def test_conditional_registration(client: TestClient):
    # Test Student Signup (Missing degree/field)
    res = client.post("/auth/register", json={
        "email": "stud1@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Student",
        "university": "Uni",
        "department": "CS"
        # missing degree + field
    })
    assert res.status_code == 422

    # Valid Student Signup
    res = client.post("/auth/register", json={
        "email": "stud1@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Student",
        "university": "Uni",
        "department": "CS",
        "degree": "MTech",
        "field_of_study": "AI"
    })
    assert res.status_code == 201

    # Test Teacher Signup (Missing designation)
    res = client.post("/auth/register", json={
        "email": "teacher1@uni.edu",
        "password": "pass",
        "role": "teacher",
        "name": "Teacher",
        "university": "Uni",
        "department": "CS"
        # missing designation
    })
    assert res.status_code == 422

    # Valid Teacher Signup
    res = client.post("/auth/register", json={
        "email": "teacher1@uni.edu",
        "password": "pass",
        "role": "teacher",
        "name": "Teacher",
        "university": "Uni",
        "department": "CS",
        "designation": "Professor"
    })
    assert res.status_code == 201

def test_advisor_linking_flow(client: TestClient, db: Session):
    # 1. Register student with advisor invite
    res_student = client.post("/auth/register", json={
        "email": "student_invite@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Invite Student",
        "university": "Uni",
        "department": "CS",
        "degree": "BTech",
        "field_of_study": "CS",
        "advisor_email": "advisor_invite@uni.edu"
    })
    assert res_student.status_code == 201
    student_id = res_student.json()["id"]

    # Verify AdvisorLink row is created with null teacher_id
    link = db.query(AdvisorLink).filter(AdvisorLink.student_id == student_id).first()
    assert link is not None
    assert link.teacher_email == "advisor_invite@uni.edu"
    assert link.teacher_id is None
    assert link.accepted is False

    # 2. Register advisor with matching email
    res_teacher = client.post("/auth/register", json={
        "email": "advisor_invite@uni.edu",
        "password": "pass",
        "role": "teacher",
        "name": "Invite Teacher",
        "university": "Uni",
        "department": "CS",
        "designation": "Assoc Prof"
    })
    assert res_teacher.status_code == 201
    teacher_id = res_teacher.json()["id"]

    # Verify link has teacher_id auto-attached
    db.refresh(link)
    assert link.teacher_id == teacher_id

    # 3. Log in as Teacher
    login_res = client.post("/auth/login", data={
        "username": "advisor_invite@uni.edu",
        "password": "pass"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Retrieve pending links
    pending_res = client.get("/links/pending", headers=headers)
    assert len(pending_res.json()) == 1
    link_id = pending_res.json()[0]["id"]

    # Create student's thesis metadata first
    login_student = client.post("/auth/login", data={
        "username": "student_invite@uni.edu",
        "password": "pass"
    })
    student_token = login_student.json()["access_token"]
    student_headers = {"Authorization": f"Bearer {student_token}"}
    
    thesis_res = client.post("/theses", json={
        "title": "Quantum AI Thesis",
        "field": "CS",
        "degree_level": "PhD"
    }, headers=student_headers)
    assert thesis_res.status_code == 201
    thesis_id = thesis_res.json()["id"]

    # Accept the link
    accept_res = client.post(f"/links/{link_id}/accept", headers=headers)
    assert accept_res.status_code == 200
    assert accept_res.json()["accepted"] is True

    # Verify the thesis advisor_id has been set to the teacher
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    assert thesis.advisor_id == teacher_id

def test_heading_based_synonyms_extraction():
    text = (
        "Some Title Page Info\n\n"
        "Abstract\n\n"
        "Introduction\n"
        "This is details about introduction chapter.\n\n"
        "Literature Review\n"
        "This details related works.\n\n"
        "Methods\n"
        "Research methodology steps.\n\n"
        "Findings\n"
        "Here are results and findings.\n\n"
        "Analysis\n"
        "Discussion and data analysis.\n\n"
        "Summary\n"
        "Conclusion summary of contributions."
    )
    
    sec_map = detect_sections_regex(text)
    assert "introduction" in sec_map
    assert "related_work" in sec_map
    assert "methodology" in sec_map
    assert "results" in sec_map
    assert "discussion" in sec_map
    assert "conclusion" in sec_map

def test_extract_candidate_headings():
    text = (
        "# 1. Introduction\n"
        "Short paragraph content here.\n\n"
        "METHODOLOGY CHAPTER\n"
        "Some details."
    )
    candidates = extract_candidate_headings(text)
    texts = [c["text"] for c in candidates]
    assert "# 1. Introduction" in texts
    assert "METHODOLOGY CHAPTER" in texts
    assert "Short paragraph content here." not in texts

def test_chunking_bounds():
    sec_text = "Paragraph 1 is here.\n\nParagraph 2 is here."
    chunks = chunk_section_text(sec_text, "introduction")
    assert len(chunks) == 1
    assert "Paragraph 1" in chunks[0]["content"]
    assert "Paragraph 2" in chunks[0]["content"]

def test_loud_embedding_key_failure():
    # Force settings ENV to "dev"
    old_env = settings.ENV
    settings.ENV = "dev"
    
    try:
        # Running get_embeddings outside of tests environment should raise ValueError
        with pytest.raises(ValueError) as exc:
            get_embeddings(["test"])
        assert "VOYAGE_API_KEY is missing/mock in a non-test environment" in str(exc.value)
    finally:
        # Restore old env
        settings.ENV = old_env

@pytest.mark.asyncio
async def test_quality_review_missing_evidence_guardrail(db: Session):
    student = User(email="std@uni.edu", password_hash="hash", role=RoleEnum.student, name="Stud")
    db.add(student)
    db.commit()
    
    thesis = Thesis(owner_id=student.id, title="Rubric Thesis")
    db.add(thesis)
    db.commit()
    
    # Text containing "generate_invalid_weakness" triggers mock generator to output empty evidence_excerpt
    raw_text = (
        "Introduction\n"
        "generate_invalid_weakness\n\n"
        "References\n"
        "[1] Citation."
    )
    
    version = ThesisVersion(
        thesis_id=thesis.id,
        version_number=1,
        raw_text=raw_text,
        section_map={"introduction": [0, len(raw_text)]}
    )
    db.add(version)
    db.commit()

    # Executing quality pipeline should raise ValueError since empty evidence_excerpt is generated
    with pytest.raises(ValueError) as exc:
        run_quality_review_pipeline(db, version)
    assert "Flag must carry a non-empty evidence excerpt" in str(exc.value)

@pytest.mark.asyncio
async def test_dashboard_visual_endpoint(client: TestClient, db: Session):
    # Setup student
    res = client.post("/auth/register", json={
        "email": "stud_dash@uni.edu",
        "password": "pass",
        "role": "student",
        "name": "Dash Student",
        "university": "Uni",
        "department": "CS",
        "degree": "BTech",
        "field_of_study": "CS"
    })
    student_id = res.json()["id"]

    # Log in
    login_res = client.post("/auth/login", data={
        "username": "stud_dash@uni.edu",
        "password": "pass"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create Thesis
    thesis_res = client.post("/theses", json={
        "title": "Dashboard Thesis",
        "field": "CS",
        "degree_level": "BTech"
    }, headers=headers)
    thesis_id = thesis_res.json()["id"]

    # Try dashboard when no version uploaded
    dash_res = client.get(f"/theses/{thesis_id}/dashboard", headers=headers)
    assert dash_res.status_code == 200
    assert dash_res.json()["overall_score"] == 0

    # Insert a dummy version with quality snapshot
    version = ThesisVersion(
        thesis_id=thesis_id,
        version_number=1,
        raw_text="Introduction\nThis is a weak intro.",
        section_map={"introduction": [0, 20]}
    )
    db.add(version)
    db.commit()

    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    thesis.current_version_id = version.id
    db.commit()

    # Trigger quality review
    trigger_res = client.post(f"/theses/{thesis_id}/quality-review/trigger", headers=headers)
    print("DEBUG TRIGGER RES:", trigger_res.status_code, trigger_res.text)
    assert trigger_res.status_code == 201

    # Fetch dashboard again
    dash_res = client.get(f"/theses/{thesis_id}/dashboard", headers=headers)
    assert dash_res.status_code == 200
    data = dash_res.json()
    assert data["overall_score"] > 0
    assert "introduction" in data["sections"]
    assert "score" in data["sections"]["introduction"]
    assert "flag_count" in data["sections"]["introduction"]
