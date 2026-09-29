import sys
import os

_BACKEND = os.path.dirname(os.path.abspath(__file__))   # src/backend
_SRC = os.path.dirname(_BACKEND)
sys.path.insert(0, os.path.join(_BACKEND, 'router'))
sys.path.insert(0, os.path.join(_BACKEND, 'core'))
sys.path.insert(0, _BACKEND)
sys.path.insert(0, _SRC)



import argparse
import threading
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .auth import install_token_auth, is_loopback_host
from .core import generator_registry
from .core.model_base import GeneratorFactory
from .core.model_utils import apply_hf_token, get_media_root
from src.backend.router.image import ImageRouter
from src.backend.router.image_frame import ImageFrameRouter
from src.backend.router.speech import SpeechRouter
from src.backend.router.transcription import TranscriptionRouter
from src.backend.router.text import TextRouter
from src.backend.router.animation import AnimationRouter
from src.backend.router.translate import TranslateRouter
from src.backend.router.prompt import PromptRouter
from src.backend.router.library import LibraryRouter
from src.backend.router.storage import StorageRouter
from src.shared.settings import ConfigManager

VERSION = "1.0.0"
BACKEND_NAME = "Asset Generator Backend"
DESCRIPTION = "图像 / 动画 / 语音 / 文本生成 API"


def _warmup_generators():
    """后台预热重型依赖，不阻塞启动；名称列表已在 lifespan 中同步注册完毕。"""
    try:
        generator_registry.warmup()
        print("✅ 生成器预热完成，设备:", GeneratorFactory._device)
    except Exception as e:
        print(f"⚠️ 生成器预热失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    generator_registry.register_all()
    GeneratorFactory.apply_setting(setting=ConfigManager().get_backend_config())
    # token 只在后端进程内读取，不放进 get_backend_config()，避免随每个请求体传输
    apply_hf_token(ConfigManager().get("huggingface", "token", ""))
    GeneratorFactory.mark_ready()
    threading.Thread(target=_warmup_generators, daemon=True, name="warmup-generators").start()
    yield



def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan,
        title=BACKEND_NAME,
        version=VERSION,
        description=DESCRIPTION,
    )

    # 允许本地 UI 进程跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_token_auth(app)

    # 注册路由
    _routers = [
        ImageRouter(),
        ImageFrameRouter(),
        TextRouter(),
        AnimationRouter(),
        SpeechRouter(),
        TranscriptionRouter(),
        TranslateRouter(),
        PromptRouter(),
        LibraryRouter(),
        StorageRouter(),
    ]
    for r in _routers:
        app.include_router(r.router)

    # 生成素材的媒体服务：前端通过 /media/... 相对 URL 拉取文件，
    # 而不是依赖本地绝对路径（为将来 backend 独立部署/远程运行预留空间）
    app.mount("/media", StaticFiles(directory=str(get_media_root())), name="media")

    # import cProfile
    # import pstats
    # import io
    # import time
    #
    #
    # t_start = time.perf_counter()
    # pr = cProfile.Profile()
    # pr.enable()

    # pr.disable()
    # t_end = time.perf_counter()
    #
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s)
    # ps.sort_stats('cumulative')
    # ps.print_stats(40)
    # print(s.getvalue())

    # 健康检查：进程是否存活（uvicorn 起来就返回 ok，不等模型注册）
    @app.get("/health", tags=["System"])
    async def health() -> dict:
        return {"status": "ok"}

    # 就绪检查：模型清单是否已注册，UI 拉取 /models 前应先确认这个
    @app.get("/ready", tags=["System"])
    async def ready() -> dict:
        return {"ready": GeneratorFactory.is_ready()}
    return app

app = create_app()

if __name__ == "__main__":
    """
    # UI 进程内
    import subprocess, sys
    proc = subprocess.Popen([sys.executable, "-m", "api.server", "--port", "8765"])

    # 手动调试
    uvicorn api.server:app --port 8765 --reload
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("VELUXIA_TOKEN", ""),
                        help="远程访问令牌；也可用环境变量 VELUXIA_TOKEN，避免出现在进程命令行里")
    args = parser.parse_args()
    # 后端能读写生成结果、占用显卡，对外暴露时不允许裸奔
    if not args.token and not is_loopback_host(args.host):
        parser.error("监听非本机地址时必须提供 --token 或环境变量 VELUXIA_TOKEN")
    app.state.api_token = args.token
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")



