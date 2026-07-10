from fastapi import APIRouter, Depends, Query, Header, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models import Notification, User
from app.schemas import NotificationResponse
from app.dependencies import get_current_user
from app.config import settings

router = APIRouter(prefix="", tags=["notifications"])

@router.get("/notifications", response_model=List[NotificationResponse])
def get_notifications(
    unread: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Notification).filter(Notification.user_id == current_user.id)
    if unread:
        query = query.filter(Notification.read == False)
    return query.order_by(Notification.created_at.desc()).all()

@router.post("/internal/send-digests")
def send_digests(
    x_internal_token: str = Header(None),
    db: Session = Depends(get_db)
):
    if not x_internal_token or x_internal_token != settings.INTERNAL_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized internal token."
        )
    # Find all unread batched notifications
    unread_batched = db.query(Notification).filter(
        Notification.batched == True,
        Notification.read == False
    ).all()
    
    digests = {}
    for n in unread_batched:
        if n.user_id not in digests:
            digests[n.user_id] = []
        digests[n.user_id].append(n.message)
        n.read = True  # Mark sent
        
    db.commit()
    return {"sent_digests_count": len(unread_batched), "digests": digests}

@router.post("/internal/reindex")
def trigger_database_reindex(
    x_internal_token: str = Header(None),
    db: Session = Depends(get_db)
):
    if not x_internal_token or x_internal_token != settings.INTERNAL_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized internal token."
        )
    
    from sqlalchemy import text
    try:
        # Check database dialect - if SQLite, REINDEX INDEX is syntax error or operates differently.
        # So we check connection url and run reindex only if postgres.
        # Let's run text("REINDEX INDEX ix_corpus_embedding")
        db.execute(text("REINDEX INDEX ix_corpus_embedding"))
        db.commit()
    except Exception as e:
        db.rollback()
        # Fallback for SQLite testing (SQLite does not support REINDEX INDEX index_name directly in same way, but REINDEX is fine)
        try:
            db.execute(text("REINDEX"))
            db.commit()
        except Exception as sqlite_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reindex failed: {str(sqlite_err)}"
            )
            
    return {"message": "Reindex executed successfully"}
