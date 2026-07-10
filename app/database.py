from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

# For SQLite testing we might need check_same_thread: False, but since pgvector is used,
# PostgreSQL is the primary engine. We configure it here.
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
