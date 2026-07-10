from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from app.config import settings

# IP-based rate limiting key for Auth endpoints
def get_ip_key(request: Request) -> str:
    return get_remote_address(request)

# User-based rate limiting key for LLM cost control
def get_user_key(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return user_id
    # Fallback to IP address if user not authenticated
    return get_remote_address(request)

limiter = Limiter(key_func=get_ip_key, enabled=(settings.ENV != "test"))
