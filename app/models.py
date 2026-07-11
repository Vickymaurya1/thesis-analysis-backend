# models.py
import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, ForeignKey, DateTime,
    Enum, Text, JSON, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base, validates
from pgvector.sqlalchemy import Vector

Base = declarative_base()

def gen_uuid():
    return str(uuid.uuid4())

# ---------- ENUMS ----------

class RoleEnum(str, enum.Enum):
    student = "student"
    teacher = "teacher"
    admin = "admin"

class ThesisStatusEnum(str, enum.Enum):
    draft = "draft"
    in_review = "in_review"
    defended = "defended"
    archived = "archived"

class FlagTypeEnum(str, enum.Enum):
    citation = "citation"
    plagiarism = "plagiarism"
    novelty = "novelty"
    quality = "quality"

class SeverityEnum(str, enum.Enum):
    critical = "critical"
    moderate = "moderate"
    low = "low"

class SupportEnum(str, enum.Enum):
    yes = "yes"
    no = "no"
    partial = "partial"
    unverifiable = "unverifiable"

class MilestoneStatusEnum(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    late = "late"


# ---------- CORE ----------

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(Enum(RoleEnum), nullable=False)
    name = Column(String, nullable=False)
    university = Column(String)
    department = Column(String)

    # student-only fields (nullable for teachers)
    degree = Column(String)               # BTech / MTech / PhD
    field_of_study = Column(String)

    # teacher-only fields
    designation = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

    theses_owned = relationship("Thesis", back_populates="owner", foreign_keys="Thesis.owner_id")
    theses_advised = relationship("Thesis", back_populates="advisor", foreign_keys="Thesis.advisor_id")


class AdvisorLink(Base):
    """Pending/accepted student<->teacher links (invite by email flow)."""
    __tablename__ = "advisor_links"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    student_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    teacher_email = Column(String, nullable=False)   # invite target, may not have signed up yet
    teacher_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    accepted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Thesis(Base):
    __tablename__ = "theses"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    owner_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    advisor_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    title = Column(String, nullable=False)
    field = Column(String)                # drives field-specific rubric selection
    degree_level = Column(String)
    status = Column(Enum(ThesisStatusEnum), default=ThesisStatusEnum.draft)
    current_version_id = Column(UUID(as_uuid=False), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="theses_owned", foreign_keys=[owner_id])
    advisor = relationship("User", back_populates="theses_advised", foreign_keys=[advisor_id])
    versions = relationship("ThesisVersion", back_populates="thesis")

    @property
    def owner_name(self):
        return self.owner.name if self.owner else ""

    @property
    def owner_degree(self):
        return self.owner.degree if self.owner else ""

    @property
    def owner_department(self):
        return self.owner.department if self.owner else ""


class ThesisVersion(Base):
    __tablename__ = "thesis_versions"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    thesis_id = Column(UUID(as_uuid=False), ForeignKey("theses.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    file_ref = Column(String)             # storage path/url of uploaded PDF/DOCX
    raw_text = Column(Text)               # extracted full text
    diff_from_prev = Column(JSON)         # {"changed_paragraphs": [...], "changed_sections": [...]}
    section_map = Column(JSON, nullable=True) # {"introduction": [0, 1200], ...}
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    thesis = relationship("Thesis", back_populates="versions")
    chunks = relationship("Chunk", back_populates="version")

    __table_args__ = (UniqueConstraint("thesis_id", "version_number"),)


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    version_id = Column(UUID(as_uuid=False), ForeignKey("thesis_versions.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer)
    section_label = Column(String)        # e.g. "3.2 Methodology"
    embedding = Column(Vector(1024))      # voyage-3 dim; change here if swapping models

    version = relationship("ThesisVersion", back_populates="chunks")


class AnalysisSnapshot(Base):
    __tablename__ = "analysis_snapshots"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    version_id = Column(UUID(as_uuid=False), ForeignKey("thesis_versions.id"), nullable=False, index=True)
    tool_type = Column(String, nullable=False)   # "citation_verification", "quality_review", etc.
    scores = Column(JSON)                        # {"overall": 78, "sections": {...}}
    generated_at = Column(DateTime, default=datetime.utcnow)

    flags = relationship("Flag", back_populates="snapshot")


class Flag(Base):
    __tablename__ = "flags"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    snapshot_id = Column(UUID(as_uuid=False), ForeignKey("analysis_snapshots.id"), nullable=False, index=True)
    type = Column(Enum(FlagTypeEnum), nullable=False)
    severity = Column(Enum(SeverityEnum), nullable=False)
    message = Column(Text, nullable=False)
    evidence_excerpt = Column(Text, nullable=False)       # ENFORCED non-null at database & application layer
    page_ref = Column(String)
    resolved = Column(Boolean, default=False)
    resolved_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    snapshot = relationship("AnalysisSnapshot", back_populates="flags")

    @validates("evidence_excerpt")
    def validate_evidence_excerpt(self, key, value):
        if not value or not str(value).strip():
            raise ValueError("Flag must carry a non-empty evidence excerpt.")
        return value


class CitationRecord(Base):
    __tablename__ = "citation_records"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    version_id = Column(UUID(as_uuid=False), ForeignKey("thesis_versions.id"), nullable=False)
    citation_key = Column(String, nullable=False)     # e.g. "[12]" or "(Smith 2023)"
    claim_text = Column(Text, nullable=False)
    claim_location = Column(String)                   # "para 4.1"
    exists_in_bib = Column(Boolean, default=False)
    doi = Column(String, nullable=True)
    format_ok = Column(Boolean, default=False)
    supports_claim = Column(Enum(SupportEnum), default=SupportEnum.unverifiable)
    confidence = Column(Float, default=0.0)
    source_snippet = Column(Text, nullable=True)       # cited paper's relevant excerpt, if fetched
    reasoning = Column(Text, nullable=True)            # reasoning from LLM verification
    created_at = Column(DateTime, default=datetime.utcnow)


class Milestone(Base):
    __tablename__ = "milestones"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    thesis_id = Column(UUID(as_uuid=False), ForeignKey("theses.id"), nullable=False)
    title = Column(String, nullable=False)
    due_date = Column(DateTime)
    status = Column(Enum(MilestoneStatusEnum), default=MilestoneStatusEnum.pending)
    approved_by_advisor = Column(Boolean, default=False)


class AdvisorComment(Base):
    __tablename__ = "advisor_comments"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    thesis_id = Column(UUID(as_uuid=False), ForeignKey("theses.id"), nullable=False)
    advisor_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    tool_context = Column(String)         # which tool this comment relates to
    content = Column(Text, nullable=False)
    severity = Column(Enum(SeverityEnum), default=SeverityEnum.moderate)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    type = Column(String)                 # "critical", "digest"
    message = Column(Text, nullable=False)
    related_flag_id = Column(UUID(as_uuid=False), ForeignKey("flags.id"), nullable=True)
    read = Column(Boolean, default=False)
    batched = Column(Boolean, default=False)   # true = held for daily digest
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------- PHASE 2 INTEGRITY CORPS ----------

class CorpusSourceEnum(str, enum.Enum):
    internal_thesis = "internal_thesis"
    external_paper = "external_paper"

class CorpusDocument(Base):
    __tablename__ = "corpus_documents"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    source_type = Column(Enum(CorpusSourceEnum), nullable=False)
    source_thesis_id = Column(UUID(as_uuid=False), ForeignKey("theses.id"), nullable=True)  # if internal
    external_id = Column(String, nullable=True)     # DOI or Semantic Scholar paper ID, if external
    title = Column(String)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))
    indexed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_corpus_embedding", "embedding", postgresql_using="ivfflat",
              postgresql_with={"lists": 100}, postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

class SemanticScholarSearch(Base):
    __tablename__ = "semanticscholar_searches"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    query_text = Column(String, unique=True, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow)

class ReviewerSimSession(Base):
    __tablename__ = "reviewer_sim_sessions"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    thesis_id = Column(UUID(as_uuid=False), ForeignKey("theses.id"), nullable=False)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    mode = Column(String, nullable=False)   # "student_practice" or "teacher_report"
    transcript = Column(JSON, nullable=True)   # list of {role, message} turns for student mode
    report = Column(JSON, nullable=True)   # populated for teacher mode
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)   # "flag_resolved", "thesis_status_changed", "flag_created", etc.
    resource_type = Column(String)             # "Flag", "Thesis", "AnalysisSnapshot"
    resource_id = Column(UUID(as_uuid=False), nullable=True)
    details = Column(JSON)                     # before/after values where relevant
    created_at = Column(DateTime, default=datetime.utcnow)
