from sqlalchemy.orm import Session
from typing import Optional, Any, Dict
from app.models import AuditLog

def log_audit_action(
    db: Session,
    user_id: str,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> AuditLog:
    """
    Logs an audit action to the database in the SAME transaction (caller is responsible for db.commit()).
    """
    log_entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details
    )
    db.add(log_entry)
    return log_entry
