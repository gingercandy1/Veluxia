"""生成器单例契约：同一个类只有一个实例，但换模型名时必须卸载旧权重并重新绑定新模型。

无权重、无 GPU 也可跑。
"""
import pytest

from src.backend.core.generator_registry import register_all
from src.backend.core.model_base import BaseImageGenerator, GeneratorFactory, SingletonMeta
from src.shared.enum_type import FactoryType


class _FakeGenerator(BaseImageGenerator):
    _model_attrs = ("pipe", "extra_pipe")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extra_pipe = None

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()
        self.extra_pipe = object()

    def parse_params(self, raw):
        pass

    async def generate(self):
        pass


@pytest.fixture(autouse=True)
def _reset_fake_singleton():
    SingletonMeta._instances.pop(_FakeGenerator, None)
    yield
    SingletonMeta._instances.pop(_FakeGenerator, None)


def test_same_class_keeps_single_instance_but_follows_latest_model_name():
    a = _FakeGenerator(model_name="A", device="cpu")
    b = _FakeGenerator(model_name="B", device="cpu")
    assert a is b
    assert b.model_name == "B"


def test_switching_model_releases_every_loaded_model():
    generator = _FakeGenerator(model_name="A", device="cpu")
    generator.ensure_model_loaded()
    assert generator.pipe is not None and generator.extra_pipe is not None

    _FakeGenerator(model_name="B", device="cpu")

    assert generator.pipe is None and generator.extra_pipe is None


def test_same_model_name_keeps_loaded_weights():
    generator = _FakeGenerator(model_name="A", device="cpu")
    generator.ensure_model_loaded()
    loaded = generator.pipe

    _FakeGenerator(model_name="A", device="cpu")

    assert generator.pipe is loaded


def test_shared_class_models_resolve_their_own_repo():
    register_all()
    old = GeneratorFactory.build_generator(FactoryType.Animation, "LTX-2.3")
    assert old.model_id == "Lightricks/LTX-2.3-Diffusers"
    new = GeneratorFactory.build_generator(FactoryType.Animation, "LTX-2.5")
    assert new.model_name == "LTX-2.5"
    assert new.model_id == "Lightricks/LTX-2.5-Diffusers"


def test_frame_interpolation_backend_follows_model_name():
    register_all()
    film = GeneratorFactory.build_generator(FactoryType.ImageFrame, "FILM")
    assert film.backend == "film"
    rife = GeneratorFactory.build_generator(FactoryType.ImageFrame, "Rife")
    assert rife.backend == "rife"
