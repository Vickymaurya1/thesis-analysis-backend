import pytest
import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Mock pgvector for SQLite tests
from sqlalchemy.types import UserDefinedType

class FakeVector(UserDefinedType):
    def __init__(self, dim=None):
        self.dim = dim
    def get_col_spec(self, **kw):
        return "TEXT"
    def bind_processor(self, dialect):
        return lambda value: json.dumps(list(value)) if value is not None else None
    def result_processor(self, dialect, coltype):
        return lambda value: json.loads(value) if value is not None else None

# Mock the Vector class in pgvector.sqlalchemy module before importing any app modules
import pgvector.sqlalchemy
pgvector.sqlalchemy.Vector = FakeVector

# Set environment variables for testing
os.environ["ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ANTHROPIC_API_KEY"] = "mock_api_key_for_testing"
os.environ["JWT_SECRET_KEY"] = "testsecretkeytestsecretkeytestsecretkey"

from app.models import Base
from app.main import app
from app.database import get_db
from app.dependencies import create_access_token

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db():
    # Make sure metadata tables are created for tests
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def get_auth_headers():
    def _headers(user_id: str) -> dict:
        token = create_access_token(data={"sub": user_id})
        return {"Authorization": f"Bearer {token}"}
    return _headers
