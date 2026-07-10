from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.models import User, AdvisorLink, Thesis, RoleEnum
from app.schemas import RegisterRequest, UserResponse, Token, AdvisorLinkResponse
from app.dependencies import get_password_hash, verify_password, create_access_token, get_current_user
from app.limiter import limiter
from app.services.audit import log_audit_action

router = APIRouter(prefix="", tags=["auth"])

@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def register(request: Request, user_in: RegisterRequest, db: Session = Depends(get_db)):
    # 1. Email check
    db_user = db.query(User).filter(User.email == user_in.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
        
    # 2. Role-conditional checks (Reject with 422 for missing role-specific fields)
    if user_in.role == RoleEnum.student:
        if not user_in.degree or not user_in.degree.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="degree is required for students"
            )
        if not user_in.field_of_study or not user_in.field_of_study.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="field_of_study is required for students"
            )
    elif user_in.role == RoleEnum.teacher:
        if not user_in.designation or not user_in.designation.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="designation is required for teachers"
            )

    # 3. Create User
    hashed_pwd = get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email,
        password_hash=hashed_pwd,
        role=user_in.role,
        name=user_in.name,
        university=user_in.university,
        department=user_in.department,
        degree=user_in.degree,
        field_of_study=user_in.field_of_study,
        designation=user_in.designation
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
 
    # Log successful registration
    log_audit_action(
        db,
        user_id=new_user.id,
        action="register",
        resource_type="User",
        resource_id=new_user.id,
        details={"email": new_user.email}
    )
    db.commit()

    # 4. Advisor linking logic
    if user_in.role == RoleEnum.student and user_in.advisor_email:
        # Check if the teacher is already signed up
        teacher = db.query(User).filter(
            User.email == user_in.advisor_email,
            User.role == RoleEnum.teacher
        ).first()
        teacher_id = teacher.id if teacher else None

        new_link = AdvisorLink(
            student_id=new_user.id,
            teacher_email=user_in.advisor_email,
            teacher_id=teacher_id,
            accepted=False
        )
        db.add(new_link)
        db.commit()

    elif user_in.role == RoleEnum.teacher:
        # Check for any student links that invited this teacher by email
        links = db.query(AdvisorLink).filter(
            AdvisorLink.teacher_email == new_user.email
        ).all()
        for link in links:
            link.teacher_id = new_user.id
        if links:
            db.commit()

    return new_user

@router.post("/auth/login", response_model=Token)
@limiter.limit("5/minute")
def login(request: Request, response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        if user:
            # Log failed login attempt for existing user
            log_audit_action(
                db,
                user_id=user.id,
                action="login_failed",
                resource_type="User",
                resource_id=user.id,
                details={"email": form_data.username}
            )
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Log successful login
    log_audit_action(
        db,
        user_id=user.id,
        action="login_success",
        resource_type="User",
        resource_id=user.id,
        details={"email": user.email}
    )
    db.commit()
    
    access_token = create_access_token(data={"sub": user.id})
    
    # Set httpOnly cookie for frontend auth
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=3600 * 24,  # 1 day
        expires=3600 * 24,
        samesite="lax",
        secure=False,  # Set to True in production (with HTTPS)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/auth/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# ---------- ADVISOR LINKING ENDPOINTS ----------

@router.post("/links/{link_id}/accept", response_model=AdvisorLinkResponse)
def accept_link(
    link_id: str,
    thesis_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != RoleEnum.teacher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers can accept advisor link invitations."
        )

    link = db.query(AdvisorLink).filter(AdvisorLink.id == link_id).first()
    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Link invitation not found"
        )

    # Security check: ensure this teacher is indeed the recipient
    if link.teacher_email != current_user.email and link.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This invitation is not addressed to you."
        )

    link.teacher_id = current_user.id
    link.accepted = True
    db.commit()

    # Link the advisor to a single thesis
    if thesis_id:
        thesis = db.query(Thesis).filter(
            Thesis.id == thesis_id,
            Thesis.owner_id == link.student_id
        ).first()
        if not thesis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thesis not found or does not belong to the student."
            )
        thesis.advisor_id = current_user.id
        db.commit()
    else:
        # Default to the student's most recently created thesis
        thesis = db.query(Thesis).filter(
            Thesis.owner_id == link.student_id
        ).order_by(Thesis.created_at.desc()).first()
        if thesis:
            thesis.advisor_id = current_user.id
            db.commit()

    log_audit_action(
        db,
        user_id=current_user.id,
        action="advisor_link_accept",
        resource_type="AdvisorLink",
        resource_id=link.id,
        details={"student_id": link.student_id, "thesis_id": thesis.id if thesis else None}
    )
    db.commit()
    db.refresh(link)
    return link

@router.get("/links/pending", response_model=List[AdvisorLinkResponse])
def get_pending_links(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != RoleEnum.teacher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers have pending student links."
        )

    pending = db.query(AdvisorLink).filter(
        (AdvisorLink.teacher_id == current_user.id) | (AdvisorLink.teacher_email == current_user.email),
        AdvisorLink.accepted == False
    ).all()
    return pending
