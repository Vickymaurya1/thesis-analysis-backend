from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models import RoleEnum, ThesisStatusEnum, FlagTypeEnum, SeverityEnum, SupportEnum, MilestoneStatusEnum

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    role: RoleEnum
    name: str = Field(..., max_length=100)
    university: str = Field(..., max_length=100)
    department: str = Field(..., max_length=100)

    # student-only — required if role == student, ignored if teacher
    degree: Optional[str] = Field(None, max_length=100)
    field_of_study: Optional[str] = Field(None, max_length=100)
    advisor_email: Optional[str] = None

    # teacher-only — required if role == teacher, ignored if student
    designation: Optional[str] = Field(None, max_length=100)

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    role: RoleEnum
    name: str
    university: Optional[str] = None
    department: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    designation: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class ThesisCreate(BaseModel):
    title: str
    field: Optional[str] = None
    degree_level: Optional[str] = None
    advisor_id: Optional[str] = None

class ThesisResponse(BaseModel):
    id: str
    owner_id: str
    advisor_id: Optional[str] = None
    title: str
    field: Optional[str] = None
    degree_level: Optional[str] = None
    status: ThesisStatusEnum
    current_version_id: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ThesisVersionCreate(BaseModel):
    raw_text: str
    file_ref: Optional[str] = None

class ThesisVersionResponse(BaseModel):
    id: str
    thesis_id: str
    version_number: int
    file_ref: Optional[str] = None
    section_map: Optional[Dict[str, List[int]]] = None
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)

class CitationResponse(BaseModel):
    id: str
    version_id: str
    citation_key: str
    claim_text: str
    claim_location: Optional[str] = None
    exists_in_bib: bool
    doi: Optional[str] = None
    format_ok: bool
    supports_claim: SupportEnum
    confidence: float
    source_snippet: Optional[str] = None
    reasoning: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class FlagResponse(BaseModel):
    id: str
    snapshot_id: str
    type: FlagTypeEnum
    severity: SeverityEnum
    message: str
    evidence_excerpt: str
    page_ref: Optional[str] = None
    resolved: bool
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AdvisorLinkResponse(BaseModel):
    id: str
    student_id: str
    teacher_email: str
    teacher_id: Optional[str] = None
    accepted: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class SectionSummary(BaseModel):
    score: int
    flag_count: int
    status: str  # "ok", "warning", "critical"

class CitationSummary(BaseModel):
    total: int
    verified: int
    flagged: int

class IntegritySummary(BaseModel):
    plagiarism_flags: int
    novelty_flags: int

class RecentActivityItem(BaseModel):
    version: int
    uploaded_at: datetime
    new_flags_count: int

class MilestoneItem(BaseModel):
    title: str
    status: str
    due_date: Optional[str] = None

class DashboardResponse(BaseModel):
    overall_score: int
    overall_quality_score: int
    sections: Dict[str, SectionSummary]
    citation_summary: CitationSummary
    integrity_summary: IntegritySummary
    recent_activity: List[RecentActivityItem]
    milestones: List[MilestoneItem]

class NotificationResponse(BaseModel):
    id: str
    user_id: str
    type: str
    message: str
    related_flag_id: Optional[str] = None
    read: bool
    batched: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ReviewerSimSessionCreate(BaseModel):
    mode: str  # "student_practice" or "teacher_report"

class ReviewerSimMessageRequest(BaseModel):
    message: str

class LiteratureReviewRequest(BaseModel):
    topics: Optional[List[str]] = None

class ReviewerSimSessionResponse(BaseModel):
    id: str
    thesis_id: str
    user_id: str
    mode: str
    transcript: Optional[List[Dict[str, Any]]] = None
    report: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
