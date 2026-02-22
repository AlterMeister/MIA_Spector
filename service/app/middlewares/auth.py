from fastapi import Request, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import get_settings

'''
    鉴权密钥验证
'''

settings = get_settings()

EXEMPT_PATHS = {"/healthz", "/readyz", "/metrics", "/favicon.ico","/docs","/openapi.json"}  # 不需要鉴权

def _parse_keys():
    keys = [k.strip() for k in settings.API_KEYS.split(",") if k.strip()]
    return set(keys)

ALLOWED_KEYS = _parse_keys()

print(ALLOWED_KEYS)

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 预检请求直接放行
        if request.method == "OPTIONS":
            return Response(status_code=200)

        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        
        if not settings.REQUIRE_AUTH:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            if token and (token in ALLOWED_KEYS):
                return await call_next(request)

        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
