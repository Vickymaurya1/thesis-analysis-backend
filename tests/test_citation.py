import pytest
from sqlalchemy.orm import Session
from app.models import User, Thesis, ThesisVersion, CitationRecord, Flag, AnalysisSnapshot, RoleEnum, SupportEnum, SeverityEnum, FlagTypeEnum
from app.services.citation import parse_bibliography, split_into_sentences, run_citation_verification_pipeline

def test_split_into_sentences():
    text = "This is sentence one. Sentence two? Sentence three! Sentence four."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "This is sentence one."
    assert sentences[1] == "Sentence two?"
    assert sentences[2] == "Sentence three!"
    assert sentences[3] == "Sentence four."

def test_parse_bibliography():
    text = """
    Some body text with citations.
    
    References
    [1] Smith, J. (2021). Neural Networks. DOI: 10.1000/xyz123
    2. Doe, A. (2022). Deep Learning. DOI: 10.1000/abc456
    Johnson (2023) Transformers in Action.
    """
    bib_map = parse_bibliography(text)
    assert "[1]" in bib_map
    assert "[2]" in bib_map
    assert "Johnson 2023" in bib_map
    assert "10.1000/xyz123" in bib_map["[1]"]
    assert "10.1000/abc456" in bib_map["[2]"]

@pytest.mark.asyncio
async def test_citation_pipeline_non_negotiable_snippet_rule(db: Session):
    # Setup student and thesis
    student = User(email="stud@univ.edu", password_hash="hash", role=RoleEnum.student, name="Student")
    db.add(student)
    db.commit()
    db.refresh(student)

    thesis = Thesis(owner_id=student.id, title="Test Thesis", field="CS")
    db.add(thesis)
    db.commit()
    db.refresh(thesis)

    # raw_text contains 2 citations:
    # [1] has no DOI and no snippet
    # [2] has a DOI, which normally fetches a snippet, but we will mock/test with empty snippet
    raw_text = (
        "We implement transformers [1] to improve translation accuracy.\n"
        "We also compare state-of-the-art results [2].\n\n"
        "References\n"
        "[1] Citation with no DOI.\n"
        "[2] Citation with DOI: 10.1000/nodata.\n"
    )

    version = ThesisVersion(thesis_id=thesis.id, version_number=1, raw_text=raw_text)
    db.add(version)
    db.commit()
    db.refresh(version)

    # Running verification pipeline
    # The pipeline will run the LLM service. If snippet is missing/empty, it MUST force supports_claim to "unverifiable"
    records = await run_citation_verification_pipeline(db, version)
    
    assert len(records) == 2
    for r in records:
        # Since no snippets were available for [1] or [2] (empty), they MUST be unverifiable
        assert r.supports_claim == SupportEnum.unverifiable

@pytest.mark.asyncio
async def test_flag_evidence_validation(db: Session):
    # Setup student, thesis, version, and snapshot
    student = User(email="flagtest@univ.edu", password_hash="hash", role=RoleEnum.student, name="Student")
    db.add(student)
    db.commit()

    thesis = Thesis(owner_id=student.id, title="Flag Test Thesis", field="CS")
    db.add(thesis)
    db.commit()

    version = ThesisVersion(thesis_id=thesis.id, version_number=1, raw_text="Plain text.")
    db.add(version)
    db.commit()

    snapshot = AnalysisSnapshot(version_id=version.id, tool_type="test", scores={})
    db.add(snapshot)
    db.commit()

    with pytest.raises(ValueError) as exc:
        flag = Flag(
            snapshot_id=snapshot.id,
            type=FlagTypeEnum.citation,
            severity=SeverityEnum.critical,
            message="No bibliography",
            evidence_excerpt=""  # invalid
        )
        db.add(flag)
        db.commit()
    assert "Flag must carry a non-empty evidence excerpt" in str(exc.value)

@pytest.mark.asyncio
async def test_incremental_pipeline_diff(db: Session):
    # Setup student, teacher, thesis
    student = User(email="stud2@univ.edu", password_hash="hash", role=RoleEnum.student, name="Student")
    db.add(student)
    db.commit()

    thesis = Thesis(owner_id=student.id, title="Incremental Thesis", field="CS")
    db.add(thesis)
    db.commit()

    # Upload Version 1
    raw_text1 = (
        "This is paragraph one containing a claim [1].\n\n"
        "References\n"
        "[1] Smith, J. (2021). Neural Networks. DOI: 10.1000/xyz123\n"
    )
    
    # We will insert a record to simulate that the citation has been verified in v1
    v1 = ThesisVersion(thesis_id=thesis.id, version_number=1, raw_text=raw_text1)
    db.add(v1)
    db.commit()

    # Let's write a verified v1 CitationRecord
    v1_record = CitationRecord(
        version_id=v1.id,
        citation_key="[1]",
        claim_text="This is paragraph one containing a claim [1].",
        claim_location="Para 1",
        exists_in_bib=True,
        doi="10.1000/xyz123",
        format_ok=True,
        supports_claim=SupportEnum.yes,
        confidence=0.95,
        source_snippet="Snippet from CrossRef"
    )
    db.add(v1_record)
    db.commit()

    # Upload Version 2 (Paragraph one is unchanged, new Paragraph two added)
    raw_text2 = (
        "This is paragraph one containing a claim [1].\n\n"
        "This is a new paragraph two with another claim [2].\n\n"
        "References\n"
        "[1] Smith, J. (2021). Neural Networks. DOI: 10.1000/xyz123\n"
        "[2] Doe, A. (2022). Deep Learning.\n"
    )
    
    # Simulate the diff compute by our router: changed_paragraphs has only Para 2
    diff = {"changed_paragraphs": ["This is a new paragraph two with another claim [2]."]}
    v2 = ThesisVersion(thesis_id=thesis.id, version_number=2, raw_text=raw_text2, diff_from_prev=diff)
    db.add(v2)
    db.commit()

    # Run pipeline on v2
    records_v2 = await run_citation_verification_pipeline(db, v2)
    
    assert len(records_v2) == 2
    
    # Check v2 records
    # [1] should be copied directly from v1 since paragraph was unchanged
    r1 = next(r for r in records_v2 if r.citation_key == "[1]")
    assert r1.supports_claim == SupportEnum.yes
    assert r1.confidence == 0.95
    assert r1.source_snippet == "Snippet from CrossRef"
    
    # [2] was newly analyzed and has no source snippet, so it must be unverifiable
    r2 = next(r for r in records_v2 if r.citation_key == "[2]")
    assert r2.supports_claim == SupportEnum.unverifiable
