import pytest
from fastapi import HTTPException
from app.permissions import can, AccessLevel
from app.dependencies import assert_owns_thesis, assert_advises_thesis, assert_can_view_thesis
from app.models import User, Thesis, RoleEnum

def test_static_can_permissions():
    # literature_review
    assert can("student", "literature_review", AccessLevel.VIEW) is True
    assert can("student", "literature_review", AccessLevel.FULL) is True
    assert can("teacher", "literature_review", AccessLevel.VIEW) is True
    assert can("teacher", "literature_review", AccessLevel.FULL) is False

    # milestone_approval
    assert can("student", "milestone_approval", AccessLevel.VIEW) is False
    assert can("student", "milestone_approval", AccessLevel.APPROVE) is False
    assert can("teacher", "milestone_approval", AccessLevel.APPROVE) is True

    # plagiarism_monitor
    assert can("student", "plagiarism_monitor", AccessLevel.VIEW) is True
    assert can("student", "plagiarism_monitor", AccessLevel.FULL) is False
    assert can("teacher", "plagiarism_monitor", AccessLevel.FULL) is True

def test_row_level_assertions():
    student1 = User(id="s1", email="s1@test.com", role=RoleEnum.student, name="Student 1")
    student2 = User(id="s2", email="s2@test.com", role=RoleEnum.student, name="Student 2")
    teacher1 = User(id="t1", email="t1@test.com", role=RoleEnum.teacher, name="Teacher 1")
    teacher2 = User(id="t2", email="t2@test.com", role=RoleEnum.teacher, name="Teacher 2")
    admin1 = User(id="a1", email="a1@test.com", role=RoleEnum.admin, name="Admin 1")

    thesis = Thesis(id="th1", owner_id="s1", advisor_id="t1", title="My Thesis")

    # Owner checks
    assert_owns_thesis(student1, thesis)  # should not raise
    with pytest.raises(HTTPException) as exc:
        assert_owns_thesis(student2, thesis)
    assert exc.value.status_code == 403

    # Advisor checks
    assert_advises_thesis(teacher1, thesis)  # should not raise
    with pytest.raises(HTTPException) as exc:
        assert_advises_thesis(teacher2, thesis)
    assert exc.value.status_code == 403

    # View checks
    assert_can_view_thesis(student1, thesis)  # should not raise
    assert_can_view_thesis(teacher1, thesis)  # should not raise
    assert_can_view_thesis(admin1, thesis)  # admin: should not raise
    
    with pytest.raises(HTTPException):
        assert_can_view_thesis(student2, thesis)
    with pytest.raises(HTTPException):
        assert_can_view_thesis(teacher2, thesis)
