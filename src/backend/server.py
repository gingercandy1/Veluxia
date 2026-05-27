import sys
import os

from core.preloader import preloader

_BACKEND = os.path.dirname(os.path.abspath(__file__))   # src/backend
_SRC = os.path.dirname(_BACKEND)
sys.path.insert(0, os.path.join(_BACKEND, 'router'))
sys.path.insert(0, os.path.join(_BACKEND, 'core'))
sys.path.insert(0, _BACKEND)
sys.path.insert(0, _SRC)



import argparse
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.model_base import GeneratorFactory
from src.backend.router.image import ImageRouter
from src.backend.router.speech import SpeechRouter
from src.backend.router.text import TextRouter
from src.backend.router.animation import AnimationRouter
from src.backend.router.translate import TranslateRouter
from src.shared.settings import ConfigManager

VERSION = "1.0.0"
BACKEND_NAME = "Asset Generator Backend"
DESCRIPTION = "图像 / 动画 / 语音 / 文本生成 API"

@asynccontextmanager
async def lifespan(app: FastAPI):
    preloader.preload(
        "torch",
        lambda: __import__("torch")
    )

    preloader.preload(
        "sentence_transformers",
        lambda: __import__("sentence_transformers")
    )

    preloader.preload(
        "qdrant_client",
        lambda: __import__("qdrant_client")
    )


    from src.backend.core.text.llama_chat import LlamaGenerator
    from src.backend.core.image.flux_schnell import FluxSchnellGenerator
    from src.backend.core.image_frame.film_generator import FILMInterpolationGenerator
    from src.backend.core.animation.ltx_video import LTXVideoGenerator
    from src.backend.core.animation.wan2_2 import Wan2VideoGenerator
    from src.backend.core.speech.ace_step_music import AceStepMusicGenerator
    from src.backend.core.speech.qwen3_tts import Qwen3TTSGenerator

    # startup
    setting = ConfigManager().get_backend_config()
    GeneratorFactory.apply_setting(setting=setting)
    print("✅ 所有模型已注册，设备:", GeneratorFactory._device)
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

    # 注册路由
    _routers = [
        ImageRouter(),
        TextRouter(),
        AnimationRouter(),
        SpeechRouter(),
        TranslateRouter()
    ]
    for r in _routers:
        app.include_router(r.router)

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

    # 健康检查
    @app.get("/health", tags=["System"])
    async def health() -> dict:
        return {"status": "ok"}
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
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")



