import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status, Request
from typing import Optional
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import bcrypt

from app.database import get_db
from app.config import settings
from app.models import User, Thesis
from app.permissions import can, AccessLevel

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def get_current_user(request: Request, token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    # Prioritize the Authorization header (token) to avoid TestClient cookie pollution in tests.
    actual_token = token if token else request.cookies.get("access_token")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not actual_token:
        raise credentials_exception

    try:
        payload = jwt.decode(actual_token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    request.state.user_id = user.id
    return user

def require_feature_access(feature: str, required: AccessLevel = AccessLevel.VIEW):
    def _dep(current_user: User = Depends(get_current_user)):
        if not can(current_user.role, feature, required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' cannot access '{feature}'"
            )
        return current_user
    return _dep

def assert_owns_thesis(user: User, thesis: Thesis):
    if thesis.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your thesis")

def assert_advises_thesis(user: User, thesis: Thesis):
    if thesis.advisor_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not supervise this student")

def assert_can_view_thesis(user: User, thesis: Thesis):
    if user.role == "student":
        assert_owns_thesis(user, thesis)
    elif user.role == "teacher":
        assert_advises_thesis(user, thesis)
    # admin has global view access, no row check needed
