"""生成器租约契约：全局同一时刻只驻留一个模型，且生成期间不允许被切换 / 卸载。

无权重、无 GPU 也可跑。
"""
import threading
import time

import pytest

from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.model_base import (
    BaseImageGenerator, BaseSpeechGenerator, GeneratorFactory, SingletonMeta,
)
from src.shared.enum_type import FactoryType


class _Base:
    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        pass


class _FakeImage(_Base, BaseImageGenerator):
    async def generate(self):
        pass


class _FakeSpeech(_Base, BaseSpeechGenerator):
    async def generate_music(self):
        pass


@pytest.fixture(autouse=True)
def _isolated_registry():
    fakes = {(FactoryType.Image, "img-a"): _FakeImage, (FactoryType.Image, "img-b"): _FakeImage,
             (FactoryType.Speech, "speech-a"): _FakeSpeech}
    for (ty, name), cls in fakes.items():
        GeneratorFactory.register_generator(ty, name, cls)
    yield
    for (ty, name), cls in fakes.items():
        GeneratorFactory._generators[ty].pop(name, None)
        SingletonMeta._instances.pop(cls, None)
    GeneratorFactory._holder = None


def _loaded(generator) -> bool:
    return generator.pipe is not None


def test_switching_to_another_class_unloads_the_previous_resident():
    with GeneratorFactory.acquire(FactoryType.Image, "img-a") as image:
        image.ensure_model_loaded()
    with GeneratorFactory.acquire(FactoryType.Speech, "speech-a") as speech:
        speech.ensure_model_loaded()
        assert not _loaded(image) and _loaded(speech)


def test_same_model_stays_loaded_between_requests():
    with GeneratorFactory.acquire(FactoryType.Image, "img-a") as image:
        image.ensure_model_loaded()
        loaded = image.pipe
    with GeneratorFactory.acquire(FactoryType.Image, "img-a") as again:
        assert again.pipe is loaded


def test_second_acquire_is_rejected_while_generating():
    with GeneratorFactory.acquire(FactoryType.Image, "img-a"):
        with pytest.raises(GeneratorBusyError):
            with GeneratorFactory.acquire(FactoryType.Speech, "speech-a"):
                pass


def test_rejected_switch_does_not_disturb_the_running_model():
    with GeneratorFactory.acquire(FactoryType.Image, "img-a") as image:
        image.ensure_model_loaded()
        # 同一个类换模型名会触发重绑定卸载，运行中必须被挡在门外
        with pytest.raises(GeneratorBusyError):
            with GeneratorFactory.acquire(FactoryType.Image, "img-b"):
                pass
        assert _loaded(image) and image.model_name == "img-a"


def test_lease_is_released_when_generation_fails():
    with pytest.raises(RuntimeError):
        with GeneratorFactory.acquire(FactoryType.Image, "img-a"):
            raise RuntimeError("boom")
    with GeneratorFactory.acquire(FactoryType.Image, "img-a"):
        pass


def test_exclusive_unloads_residents_and_blocks_generation():
    with GeneratorFactory.acquire(FactoryType.Image, "img-a") as image:
        image.ensure_model_loaded()
    with GeneratorFactory.exclusive():
        assert not _loaded(image)
        with pytest.raises(GeneratorBusyError):
            with GeneratorFactory.acquire(FactoryType.Image, "img-a"):
                pass
    with GeneratorFactory.acquire(FactoryType.Image, "img-a"):
        pass


def test_exclusive_is_rejected_while_generating():
    with GeneratorFactory.acquire(FactoryType.Image, "img-a"):
        with pytest.raises(GeneratorBusyError):
            with GeneratorFactory.exclusive():
                pass


class _SlowImage(_Base, BaseImageGenerator):
    """generate() 会一直阻塞到测试放行，用来模拟"正在生成"。"""
    started = threading.Event()
    release = threading.Event()

    async def generate(self):
        type(self).started.set()
        assert type(self).release.wait(10)
        return None


def _wait_job(client, job_id, terminal=("done", "error", "cancelled")):
    deadline = time.time() + 10
    while time.time() < deadline:
        status = client.get(f"/image/generate/status/{job_id}").json()
        if status["status"] in terminal:
            return status
        time.sleep(0.05)
    raise AssertionError("任务没有在预期时间内结束")


def test_router_rejects_new_job_while_another_is_generating():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.backend.router.image import ImageRouter

    GeneratorFactory.register_generator(FactoryType.Image, "slow-a", _SlowImage)
    GeneratorFactory.register_generator(FactoryType.Image, "slow-b", _SlowImage)
    _SlowImage.started.clear()
    _SlowImage.release.clear()
    app = FastAPI()
    app.include_router(ImageRouter().router)
    # 必须用 with：否则每个请求各起一个事件循环，请求结束关闭循环时会等后台线程跑完，
    # 第一个任务就被"同步等完"了，根本测不到并发
    with TestClient(app) as client:
        try:
            first = client.post("/image/generate/submit", json={"model_name": "slow-a", "extra": {}}).json()
            assert _SlowImage.started.wait(5)

            second = client.post("/image/generate/submit", json={"model_name": "slow-b", "extra": {}}).json()
            rejected = _wait_job(client, second["job_id"])
            assert rejected["status"] == "error" and "正在运行" in rejected["error"]

            # 被拒绝的切换不能动到正在运行的模型
            assert SingletonMeta._instances[_SlowImage].model_name == "slow-a"
        finally:
            _SlowImage.release.set()
        _wait_job(client, first["job_id"])
    for name in ("slow-a", "slow-b"):
        GeneratorFactory._generators[FactoryType.Image].pop(name, None)
    SingletonMeta._instances.pop(_SlowImage, None)
