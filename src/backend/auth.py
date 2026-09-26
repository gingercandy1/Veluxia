"""远程访问鉴权（ADR 0005）：配置了 token 时，除 /health 外的请求都要带 Bearer token。"""
import hmac
import ipaddress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# /health 只用于探活，不泄露任何数据；放行后前端才能区分"连不上"和"token 错"
PUBLIC_PATHS = frozenset({"/health"})


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def install_token_auth(app: FastAPI) -> None:
    """token 放在 app.state.api_token，启动参数解析后再赋值（app 在模块导入时就已创建）。"""
    app.state.api_token = ""

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        token = request.app.state.api_token
        # CORS 预检不带自定义头，拦下会让浏览器侧调用全部失败
        if not token or request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        scheme, _, provided = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(provided, token):
            return JSONResponse({"detail": "Invalid or missing API token"}, status_code=401)
        return await call_next(request)
