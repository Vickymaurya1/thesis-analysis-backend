from enum import Enum

class AccessLevel(str, Enum):
    NONE = "none"
    VIEW = "view"
    FULL = "full"
    FULL_PLUS_COMMENT = "full_plus_comment"
    APPROVE = "approve"

PERMISSION_MATRIX = {
    "literature_review":     {"student": AccessLevel.FULL, "teacher": AccessLevel.VIEW},
    "quality_review":        {"student": AccessLevel.FULL, "teacher": AccessLevel.FULL_PLUS_COMMENT},
    "citation_verification": {"student": AccessLevel.FULL, "teacher": AccessLevel.VIEW},
    "novelty_detection":     {"student": AccessLevel.FULL, "teacher": AccessLevel.FULL},
    "reviewer_simulation":   {"student": AccessLevel.FULL, "teacher": AccessLevel.FULL},
    "plagiarism_monitor":    {"student": AccessLevel.VIEW, "teacher": AccessLevel.FULL},
    "progress_tracker":      {"student": AccessLevel.VIEW, "teacher": AccessLevel.VIEW},
    "milestone_approval":    {"student": AccessLevel.NONE, "teacher": AccessLevel.APPROVE},
}

def can(role: str, feature: str, required: AccessLevel) -> bool:
    """Static check — does this role even have access at this level for this feature."""
    granted = PERMISSION_MATRIX.get(feature, {}).get(role, AccessLevel.NONE)
    rank = {
        AccessLevel.NONE: 0,
        AccessLevel.VIEW: 1,
        AccessLevel.FULL: 2,
        AccessLevel.FULL_PLUS_COMMENT: 2,
        AccessLevel.APPROVE: 2
    }
    return rank.get(granted, 0) >= rank.get(required, 0)
