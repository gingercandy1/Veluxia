"""模型文件占用与清理：列出、按大小排序、只删列出来的条目、有任务在跑时拒绝。"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.core import storage
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.model_base import GeneratorFactory, SingletonMeta
from src.backend.router.storage import StorageRouter


def _file(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    models, ace, rembg = tmp_path / "models", tmp_path / "ace", tmp_path / "u2net"
    _file(models / "image" / "sdxl-base" / "unet" / "model.safetensors", 300)
    _file(models / "image" / "sdxl-base" / ".cache" / "meta", 5)
    _file(models / "text" / "Qwen2.5-7B" / "qwen.gguf", 200)
    _file(models / "upscale" / "RealESRGAN_x4plus.pth", 50)
    _file(models / "image_frame" / "film_net" / "film.pt", 80)
    _file(models / "bge-small-zh" / "model.bin", 20)   # 直接放在 models/ 下的整个算一条
    _file(models / ".locks" / "x.lock", 1)             # 隐藏目录不列
    _file(ace / "acestep-v15" / "model.bin", 100)
    _file(rembg / "u2net.onnx", 10)
    monkeypatch.setattr(storage, "_roots", lambda: [
        ("models", models), ("ace_step", ace), ("rembg", rembg), ("missing", tmp_path / "nope")])
    yield models
    GeneratorFactory._holder = None


def test_lists_one_entry_per_model_sorted_by_size(roots):
    sized = [(entry.id, size) for entry, size in storage.list_storage()]
    assert sized == [
        ("models/image/sdxl-base", 305),   # .cache 跟着模型目录一起算
        ("models/text/Qwen2.5-7B", 200),
        ("ace_step/acestep-v15", 100),
        ("models/image_frame/film_net", 80),
        ("models/upscale/RealESRGAN_x4plus.pth", 50),
        ("models/bge-small-zh", 20),
        ("rembg/u2net.onnx", 10),
    ]
    manual = {e.id for e in storage.list_entries() if e.manual}
    assert manual == {"models/image_frame/film_net"}  # 插帧权重不会自动重新下载


def test_delete_removes_listed_entry_and_unloads_models(roots):
    unloaded = []

    class _Loaded:
        uses_vram = False  # 抠图这类 CPU 模型：exclusive() 不管它，也要卸掉才能删文件

        def unload(self):
            unloaded.append(True)
    SingletonMeta._instances[_Loaded] = _Loaded()
    try:
        storage.delete_entry("models/image/sdxl-base")
        storage.delete_entry("models/upscale/RealESRGAN_x4plus.pth")
    finally:
        SingletonMeta._instances.pop(_Loaded, None)
    assert not (roots / "image" / "sdxl-base").exists()
    assert not (roots / "upscale" / "RealESRGAN_x4plus.pth").exists()
    assert (roots / "image").is_dir()  # 只删那一条，分类目录留着
    assert unloaded == [True, True]
    assert GeneratorFactory._holder is None


@pytest.mark.parametrize("entry_id", ["models/image", "models/../models", "models/image/nope", ""])
def test_delete_only_accepts_listed_ids(roots, entry_id):
    with pytest.raises(FileNotFoundError):
        storage.delete_entry(entry_id)
    assert (roots / "image" / "sdxl-base").is_dir()


def test_delete_is_refused_while_a_model_is_running(roots):
    GeneratorFactory._holder = "SDXL"
    with pytest.raises(GeneratorBusyError):
        storage.delete_entry("models/image/sdxl-base")
    assert (roots / "image" / "sdxl-base").is_dir()


def test_routes(roots):
    app = FastAPI()
    app.include_router(StorageRouter().router)
    client = TestClient(app)
    listed = client.get("/storage/models").json()
    assert listed["total"] == 765 and listed["entries"][0]["id"] == "models/image/sdxl-base"
    assert listed["entries"][3]["manual"] is True

    after = client.post("/storage/models/delete", json={"id": "models/text/Qwen2.5-7B"}).json()
    assert after["total"] == 565
    assert "models/text/Qwen2.5-7B" not in [e["id"] for e in after["entries"]]
    assert client.post("/storage/models/delete", json={"id": "models/text/Qwen2.5-7B"}
                       ).status_code == 404
    GeneratorFactory._holder = "SDXL"
    assert client.post("/storage/models/delete", json={"id": "rembg/u2net.onnx"}
                       ).status_code == 409
