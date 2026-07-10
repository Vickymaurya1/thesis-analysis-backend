from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import IntegrityError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.routers import auth, theses, notifications
from app.database import engine
from app.models import Base
from app.limiter import limiter

# Automatically create tables for quick local setup / sqlite testing
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Thesis Analysis RAG Agent Platform",
    description="Backend API for verifying academic theses.",
    version="1.0.0"
)

# CORS configuration allowing Vercel and localhost frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://thesis-rag-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup slowapi rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global SQLAlchemy IntegrityError handler
@app.exception_handler(IntegrityError)
def integrity_exception_handler(request: Request, exc: IntegrityError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Database integrity constraint violation (e.g. duplicate key)."}
    )

app.include_router(auth.router)
app.include_router(theses.router)
app.include_router(notifications.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Thesis Analysis RAG Agent Platform API"}
