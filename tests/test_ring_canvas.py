"""循环画布：高度补齐 / 外扩 / 文字改图 / 版本切换 / 分段规则。生成器全用假的，不加载权重。"""
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.backend.core.collection import library, ring
from src.backend.core.collection.manifest import load_manifest, save_manifest
from src.backend.core.collection.ring_extend import (
    extend_canvas,
    fit_height,
    model_dimensions,
    restore_middle,
)
from src.backend.core.model_base import GeneratorFactory
from src.backend.router.library import LibraryRouter
from src.shared.schemas import (
    CollectionItem,
    ImageVersion,
    Manifest,
    RingExtendRequest,
    SegmentLayers,
    StepState,
    VersionHistory,
)


# ---------- 纯图像 ----------

def _gradient(width=40, height=20):
    """每行颜色不同的图，方便从像素判断哪一行去了哪里。"""
    image = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            image.putpixel((x, y), (y * 10, 50, 200))
    return image


def test_fit_height_centers_and_stretches_edge_rows():
    image = _gradient()
    fitted = fit_height(image, 30)
    assert fitted.size == (40, 30)
    # 上下各补 5 行：补的是最上 / 最下一行的颜色，原图居中
    assert fitted.getpixel((3, 0)) == image.getpixel((3, 0))
    assert fitted.getpixel((3, 4)) == image.getpixel((3, 0))
    assert fitted.getpixel((3, 5)) == image.getpixel((3, 0))
    assert fitted.getpixel((3, 24)) == image.getpixel((3, 19))
    assert fitted.getpixel((3, 29)) == image.getpixel((3, 19))
    assert np.array_equal(np.asarray(fitted.crop((0, 5, 40, 25))), np.asarray(image))


def test_fit_height_odd_gap_and_already_tall_enough():
    assert fit_height(_gradient(), 31).size == (40, 31)
    same = fit_height(Image.new("RGBA", (40, 20), (1, 2, 3, 4)), 20)
    assert same.mode == "RGB" and same.size == (40, 20)
    assert fit_height(_gradient(), 10).size == (40, 20)  # 比目标高：原样返回


def test_extend_canvas_adds_pad_on_both_sides():
    image = _gradient()
    canvas = extend_canvas(image, 8)
    assert canvas.size == (40, 36)
    assert canvas.getpixel((0, 7)) == image.getpixel((0, 0))
    assert canvas.getpixel((0, 8)) == image.getpixel((0, 0))
    assert canvas.getpixel((0, 27)) == image.getpixel((0, 19))
    assert canvas.getpixel((0, 35)) == image.getpixel((0, 19))


def test_restore_middle_brings_back_original_and_keeps_new_edges():
    original = _gradient()
    result = Image.new("RGB", (40, 36), (255, 0, 0))
    restored = restore_middle(result, original, pad=8, feather=4)
    assert restored.size == (40, 36)
    # 新补的上下两条是模型的结果
    assert restored.getpixel((5, 0)) == (255, 0, 0)
    assert restored.getpixel((5, 7)) == (255, 0, 0)
    assert restored.getpixel((5, 35)) == (255, 0, 0)
    # 离交界 feather 行以外完全是原图
    for y in range(4, 16):
        assert restored.getpixel((5, 8 + y)) == original.getpixel((5, y))
    # 交界处线性过渡：第一行原图占的权重小于全部，且比模型结果更接近原图的内部
    edge = restored.getpixel((5, 8))
    assert edge != (255, 0, 0) and edge != original.getpixel((5, 0))
    with pytest.raises(ValueError, match="尺寸"):
        restore_middle(Image.new("RGB", (40, 30)), original, pad=8)


def test_model_dimensions_are_legal_for_awkward_canvases():
    for size in ((700, 320), (1000, 100), (4000, 4000), (64, 64)):
        width, height = model_dimensions(*size, target_area=768 * 768)
        assert width % 32 == 0 and height % 32 == 0
        assert 256 <= width <= 2048 and 256 <= height <= 2048
    # 面积接近目标、比例大致保持
    width, height = model_dimensions(1000, 500, target_area=768 * 768)
    assert abs(width / height - 2) < 0.15


# ---------- 资源包 ----------

def _make_pack(tmp_path, monkeypatch, ids=("a", "b", "c"), size=(700, 64)):
    monkeypatch.setattr(library, "library_root", lambda: tmp_path)
    directory = tmp_path / "pack"
    items = []
    for index, item_id in enumerate(ids):
        (directory / item_id).mkdir(parents=True)
        Image.new("RGB", size, (index * 70, 20, 100)).save(directory / item_id / "generate.png")
        items.append(CollectionItem(
            id=item_id, prompt=f"景物 {item_id}", fields={"view": "side"},
            steps={"generate": StepState(status="done", outputs=[f"{item_id}/generate.png"])}))
    save_manifest(directory, Manifest(
        id="pack", type="scene", template="scene_background", items=items))
    ring.set_order("pack", list(ids))
    ring.render_ring("pack")
    return directory


class _FakeEditor:
    """假图生图 + 假接缝重绘：记录每次调用，输出按请求尺寸生成一张纯色图。"""

    def __init__(self, tmp_path, fail_on_call=None, alpha=False):
        self.tmp_path = tmp_path
        self.params = []
        self.fail_on_call = fail_on_call
        self.alpha = alpha
        self.cancel_event = None
        self.progress_callback = None
        self.callbacks_seen = []

    def ensure_model_loaded(self):
        pass

    def check_cancelled(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise RuntimeError("cancelled")

    def parse_params(self, raw):
        self.current = dict(raw)
        with Image.open(raw["reference_image"]) as reference:
            self.current["reference_size"] = reference.size
        self.params.append(self.current)

    async def generate(self):
        self.callbacks_seen.append(self.progress_callback)
        if self.fail_on_call == len(self.params):
            raise RuntimeError("fake edit failed")
        size = (self.current["width"], self.current["height"])
        mode = "RGBA" if self.alpha else "RGB"
        path = self.tmp_path / f"out_{len(self.params)}.png"
        Image.new(mode, size, (0, 255, 0, 255) if self.alpha else (0, 255, 0)).save(path)
        return path

    def repaint_seam(self, window, prompt, seed, harmonize):
        return Image.new("RGB", window.size, (240, 10, 20)), None


def _lease(monkeypatch, fake, models=None):
    @contextmanager
    def acquire(cls, factory_type, model_name):
        if models is not None:
            models.append(model_name)
        yield fake

    monkeypatch.setattr(GeneratorFactory, "acquire", classmethod(acquire))


def _slot_heights(directory):
    return [Image.open(directory / path).height for path in ring.ring_status("pack")["rendered"]]


def test_extend_only_selected_screens_keeps_slot_heights_equal_and_ring_stitchable(
        tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch)
    fake = _FakeEditor(tmp_path)
    models = []
    _lease(monkeypatch, fake, models)
    events = []

    manifest = ring.extend_screens("pack", ["a"], pad=16,
                                   on_progress=lambda key, fraction: events.append((key, fraction)))

    assert models == [ring.EDIT_MODEL]   # 一次租约包住整个循环
    assert len(fake.params) == 1         # 只扩了选中的 a
    assert events[0] == ("a", 0.0) and events[-1] == ("a", 1.0)
    assert fake.progress_callback is None and fake.cancel_event is None
    assert fake.callbacks_seen[0] is not None     # 推理期间回调是挂着的
    # 给模型的尺寸合法（32 的倍数、256～2048），内容里有外扩指令
    sent = fake.params[0]
    assert sent["width"] % 32 == 0 and sent["height"] % 32 == 0
    assert "Extend" in sent["content"]
    assert sent["reference_size"] == (700, 64 + 32)

    status = ring.ring_status("pack")
    assert status["ring_height"] == 96
    assert status["screen_padding"] == [0, 32, 32]
    assert status["screen_files"][0] == "ring/versions/screen_a/1.png"
    assert status["screen_files"][1] == "b/generate.png"
    history = status["versions"]["screen:a"]
    assert history.current == 0 and history.versions[0].kind == "extend"
    # 所有位置的合成图高度一致，长图拼得起来
    assert _slot_heights(directory) == [96, 96, 96]
    paths = ring._rendered_sources(directory, manifest.ring)
    assert ring._ring_long_image(paths).size == (2100, 96)
    # 原图没动，输出里中间 64 行是原图像素
    with Image.open(directory / "ring/versions/screen_a/1.png") as extended:
        assert extended.size == (700, 96)
        assert extended.getpixel((10, 48)) == (0, 20, 100)
    assert load_manifest(directory).ring.versions["screen:a"].current == 0


def test_extend_defaults_to_all_screens_and_accumulates_on_the_current_image(
        tmp_path, monkeypatch):
    _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    fake = _FakeEditor(tmp_path)
    _lease(monkeypatch, fake)
    ring.extend_screens("pack", pad=16)
    assert len(fake.params) == 2
    ring.extend_screens("pack", ["a"], pad=16)
    # 第二次是在第一次外扩后的图上继续扩
    assert fake.params[2]["reference_size"] == (700, 96 + 32)
    status = ring.ring_status("pack")
    assert len(status["versions"]["screen:a"].versions) == 2
    assert status["ring_height"] == 128 and status["screen_padding"] == [0, 32]


def test_extend_refuses_after_segmenting_and_validates_arguments(tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch)
    fake = _FakeEditor(tmp_path)
    _lease(monkeypatch, fake)
    with pytest.raises(ValueError, match="环上没有这些屏：x"):
        ring.extend_screens("pack", ["x"])
    with pytest.raises(ValueError, match="16～256"):
        ring.extend_screens("pack", pad=8)
    ring.confirm_cuts("pack", [0, 1000])
    with pytest.raises(ValueError, match="请在分段之前外扩"):
        ring.extend_screens("pack", ["a"])
    assert fake.params == []
    assert load_manifest(directory).ring.versions == {}


def test_failed_extend_keeps_screens_already_done(tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch)
    fake = _FakeEditor(tmp_path, fail_on_call=2)
    _lease(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="fake edit failed"):
        ring.extend_screens("pack", ["a", "b"], pad=16)
    saved = load_manifest(directory).ring.versions
    assert list(saved) == ["screen:a"]
    assert fake.cancel_event is None and fake.progress_callback is None


def test_screen_source_follows_current_version_and_base_sig(tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    manifest = load_manifest(directory)
    item = manifest.items[0]
    original = directory / "a" / "generate.png"
    assert ring._screen_source(directory, manifest.ring, item) == original.resolve()

    version = directory / "ring" / "versions" / "screen_a" / "1.png"
    version.parent.mkdir(parents=True)
    Image.new("RGB", (700, 90), "red").save(version)
    sig = ring.file_signature(original)
    manifest.ring.versions["screen:a"] = VersionHistory(
        base_sig=sig, versions=[ImageVersion(file="ring/versions/screen_a/1.png", kind="edit")],
        current=0)
    assert ring._screen_source(directory, manifest.ring, item) == version.resolve()

    manifest.ring.versions["screen:a"].current = -1     # 选了原图
    assert ring._screen_source(directory, manifest.ring, item) == original.resolve()
    manifest.ring.versions["screen:a"].current = 0
    version.unlink()                                      # 文件丢了：退回原图而不是报错
    assert ring._screen_source(directory, manifest.ring, item) == original.resolve()

    Image.new("RGB", (700, 90), "red").save(version)
    manifest.ring.versions["screen:a"].base_sig = "old-sig"   # 原图被重新生成了
    assert ring._screen_source(directory, manifest.ring, item) == original.resolve()
    save_manifest(directory, manifest)
    assert "screen:a" not in ring.ring_status("pack")["versions"]


def test_regenerated_original_resets_history_on_next_edit(tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    fake = _FakeEditor(tmp_path)
    _lease(monkeypatch, fake)
    ring.extend_screens("pack", ["a"], pad=16)
    ring.edit_target("pack", "screen:a", "make it night")
    assert len(ring.ring_status("pack")["versions"]["screen:a"].versions) == 2

    Image.new("RGB", (700, 64), "yellow").save(directory / "a" / "generate.png")
    status = ring.ring_status("pack")
    assert "screen:a" not in status["versions"] and status["ring_height"] == 64

    ring.edit_target("pack", "screen:a", "make it night")
    history = ring.ring_status("pack")["versions"]["screen:a"]
    assert len(history.versions) == 1 and history.versions[0].file.endswith("/1.png")


def test_extending_makes_seams_stale_and_old_patches_are_skipped(tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    fake = _FakeEditor(tmp_path)
    _lease(monkeypatch, fake)
    ring.generate_seams("pack", ["a>b"], cancel_event=threading.Event())
    assert ring.ring_status("pack")["seams"]["a>b"].stale is False

    ring.extend_screens("pack", ["a"], pad=16)      # 旧补丁高度 64，对不上新的 96
    status = ring.ring_status("pack")
    assert status["seams"]["a>b"].stale is True
    assert _slot_heights(directory) == [96, 96]

    ring.generate_seams("pack", cancel_event=threading.Event())   # 只补过期的接缝
    # 旧的采用候选仍是 c1（已过期）；用户采用新候选后接缝才恢复，再合成
    assert ring.ring_status("pack")["seams"]["a>b"].stale is True
    ring.adopt_seam("pack", "a>b", "c2")
    ring.render_ring("pack")
    status = ring.ring_status("pack")
    assert status["seams"]["a>b"].stale is False
    assert _slot_heights(directory) == [96, 96]
    assert Image.open(directory / status["rendered"][1]).size == (700, 96)


def test_edit_target_creates_a_version_and_select_version_goes_back(tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    fake = _FakeEditor(tmp_path)
    _lease(monkeypatch, fake)
    events = []

    manifest = ring.edit_target("pack", "screen:a", "  add snow  ",
                                on_progress=lambda key, fraction: events.append((key, fraction)))

    assert fake.params[0]["content"] == "add snow"
    assert events[0] == ("screen:a", 0.0) and events[-1] == ("screen:a", 1.0)
    history = manifest.ring.versions["screen:a"]
    assert history.current == 0
    assert history.versions[0].kind == "edit" and history.versions[0].prompt == "add snow"
    status = ring.ring_status("pack")
    assert status["screen_files"][0] == "ring/versions/screen_a/1.png"
    # 改图结果会合成进 slot：这屏的合成图是模型输出的绿色
    assert Image.open(directory / status["rendered"][0]).getpixel((300, 30)) == (0, 255, 0)

    ring.select_version("pack", "screen:a", -1)
    status = ring.ring_status("pack")
    assert status["screen_files"][0] == "a/generate.png"
    assert Image.open(directory / status["rendered"][0]).getpixel((300, 30)) == (0, 20, 100)
    assert len(status["versions"]["screen:a"].versions) == 1   # 旧版本保留

    ring.select_version("pack", "screen:a", 0)
    assert ring.ring_status("pack")["screen_files"][0] == "ring/versions/screen_a/1.png"
    with pytest.raises(ValueError, match="没有第 5 个版本"):
        ring.select_version("pack", "screen:a", 5)
    with pytest.raises(ValueError, match="没有历史版本"):
        ring.select_version("pack", "screen:b", 0)
    assert ring.select_version("pack", "screen:b", -1) is not None
    with pytest.raises(ValueError, match="请输入修改描述"):
        ring.edit_target("pack", "screen:a", "   ")
    with pytest.raises(ValueError, match="无效的目标"):
        ring.edit_target("pack", "bogus:a", "x")


class _NoLayerTemplate:
    def step_ids(self):
        return ["generate"]


def _segmented_pack(tmp_path, monkeypatch):
    directory = _make_pack(tmp_path, monkeypatch, ids=("a", "b", "c"), size=(1200, 64))
    ring.confirm_cuts("pack", [0, 1200, 2400])
    return directory


def _fake_layers(directory, key):
    """给一段塞四层假分层，返回层文件的相对路径。"""
    manifest = load_manifest(directory)
    layer_dir = directory / ring._layer_dir(key)
    layer_dir.mkdir(parents=True)
    relatives = []
    for number in range(1, 5):
        relative = (ring._layer_dir(key) / f"layer_{number}.png").as_posix()
        Image.new("RGBA", (40, 8), (number * 40, 0, 0, 255)).save(directory / relative)
        relatives.append(relative)
    manifest.ring.segment_layers[key] = SegmentLayers(
        source_sig=manifest.ring.cuts_sig, layers=relatives)
    save_manifest(directory, manifest)
    return relatives


def test_segment_and_layer_edits_use_the_current_version(tmp_path, monkeypatch):
    directory = _segmented_pack(tmp_path, monkeypatch)
    key = ring.segment_key(0, 1200)
    _fake_layers(directory, key)
    fake = _FakeEditor(tmp_path, alpha=False)
    _lease(monkeypatch, fake)

    ring.edit_target("pack", f"segment:{key}", "darker")
    status = ring.ring_status("pack")
    assert status["segments"][0] == "ring/versions/segment_0_1200/1.png"
    assert status["segments"][1] == "ring/segments/02.png"
    assert ring._confirmed_segments(directory, load_manifest(directory))[0][1] == \
        (directory / status["segments"][0]).resolve()

    # 层是透明 RGBA：模型输出不带透明时沿用原层的 alpha（这里原层 alpha 全 255，改成一半验证）
    layer_path = directory / "ring/layers/0_1200/layer_2.png"
    Image.new("RGBA", (40, 8), (80, 0, 0, 128)).save(layer_path)
    ring.edit_target("pack", f"layer:{key}:L2", "make it green")
    status = ring.ring_status("pack")
    edited = status["segment_layers"][0][1]
    assert edited == "ring/versions/layer_0_1200_L2/1.png"
    with Image.open(directory / edited) as image:
        assert image.mode == "RGBA" and image.getpixel((5, 5)) == (0, 255, 0, 128)
    assert status["segment_layers"][0][0] == "ring/layers/0_1200/layer_1.png"

    ring.select_version("pack", f"layer:{key}:L2", -1)
    assert ring.ring_status("pack")["segment_layers"][0][1] == "ring/layers/0_1200/layer_2.png"
    with pytest.raises(ValueError, match="没有层 L9"):
        ring.edit_target("pack", f"layer:{key}:L9", "x")


def test_merge_layers_uses_the_current_layer_version(tmp_path, monkeypatch):
    directory = _segmented_pack(tmp_path, monkeypatch)
    key = ring.segment_key(0, 1200)
    _fake_layers(directory, key)
    _lease(monkeypatch, _FakeEditor(tmp_path, alpha=True))
    ring.edit_target("pack", f"layer:{key}:L1", "green")      # L1 变成不透明绿色
    Image.new("RGBA", (40, 8), (0, 0, 0, 0)).save(directory / "ring/layers/0_1200/layer_2.png")

    manifest = ring.merge_layers("pack", 0, ["L1", "L2"])

    merged = manifest.ring.segment_layers[key].merged[0]
    with Image.open(directory / merged.path) as image:
        # L2 全透明，合并结果就是 L1：是改后的绿色而不是原来的红色
        assert image.getpixel((5, 5)) == (0, 255, 0, 255)


def test_confirm_cuts_clears_layers_and_segment_layer_versions_but_keeps_screens(
        tmp_path, monkeypatch):
    directory = _segmented_pack(tmp_path, monkeypatch)
    key = ring.segment_key(0, 1200)
    _fake_layers(directory, key)
    _lease(monkeypatch, _FakeEditor(tmp_path))
    ring.edit_target("pack", f"segment:{key}", "darker")
    ring.edit_target("pack", f"layer:{key}:L1", "greener")
    ring.edit_target("pack", "screen:b", "snow")
    versions_dir = directory / "ring" / "versions"
    assert (versions_dir / "segment_0_1200").is_dir() and (versions_dir / "layer_0_1200_L1").is_dir()

    manifest = ring.confirm_cuts("pack", [0, 1200, 2400])

    assert manifest.ring.segment_layers == {}
    assert set(manifest.ring.versions) == {"screen:b"}
    assert not (versions_dir / "segment_0_1200").exists()
    assert not (versions_dir / "layer_0_1200_L1").exists()
    assert (versions_dir / "screen_b").is_dir()      # 屏的历史不受影响
    assert not (directory / "ring" / "layers" / "0_1200").exists()


def test_auto_cuts_only_suggests_and_leaves_confirmed_state_untouched(tmp_path, monkeypatch):
    directory = _segmented_pack(tmp_path, monkeypatch)
    key = ring.segment_key(0, 1200)
    relatives = _fake_layers(directory, key)
    before = load_manifest(directory).ring
    monkeypatch.setattr(ring, "estimate_ring_depth",
                        lambda image, cancel_event=None: np.zeros(
                            (image.height, image.width), np.float32))
    monkeypatch.setattr(ring, "column_cost", lambda depth: np.where(
        np.isin(np.arange(depth.shape[1]), [504, 1704, 2904]), 0.0, 1.0))

    manifest = ring.auto_cuts("pack")

    assert manifest.ring.suggested_cuts == [504, 1704, 2904]
    assert manifest.ring.cuts == before.cuts == [0, 1200, 2400]
    assert manifest.ring.cuts_sig == before.cuts_sig
    assert manifest.ring.segments == before.segments and len(manifest.ring.segments) == 3
    assert list(manifest.ring.segment_layers) == [key]
    assert all((directory / path).is_file() for path in relatives)


def test_replanning_keeps_screen_versions(tmp_path, monkeypatch):
    _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    _lease(monkeypatch, _FakeEditor(tmp_path))
    ring.edit_target("pack", "screen:a", "snow")
    manifest = ring.set_order("pack", ["b", "a", "b"])
    assert "screen:a" in manifest.ring.versions
    assert ring.plan_ring("pack").ring.versions["screen:a"].current == 0


# ---------- 路由 ----------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "library_root", lambda: tmp_path)
    app = FastAPI()
    app.include_router(LibraryRouter().router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("pad", [0, 15, 257, 1000])
def test_extend_route_rejects_pad_out_of_range(client, pad):
    response = client.post("/library/ring/extend/submit", json={"pack_id": "pack", "pad": pad})
    assert response.status_code == 422


def test_extend_request_defaults_and_bounds():
    assert RingExtendRequest(pack_id="p").pad == 128
    assert RingExtendRequest(pack_id="p", pad=16).pad == 16
    assert RingExtendRequest(pack_id="p", pad=256).pad == 256
    with pytest.raises(ValueError):
        RingExtendRequest(pack_id="p", pad=300)


def test_extend_route_accepts_valid_pad_and_reports_progress(tmp_path, monkeypatch, client):
    _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    _lease(monkeypatch, _FakeEditor(tmp_path))
    response = client.post("/library/ring/extend/submit",
                           json={"pack_id": "pack", "item_ids": ["a"], "pad": 16})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    for _ in range(100):
        status = client.get(f"/library/ring/extend/status/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        threading.Event().wait(0.05)
    assert status["status"] == "done" and not status["error"], status
    assert "screen:a" in load_manifest(Path(tmp_path) / "pack").ring.versions


def test_version_route_switches_version(tmp_path, monkeypatch, client):
    _make_pack(tmp_path, monkeypatch, ids=("a", "b"))
    _lease(monkeypatch, _FakeEditor(tmp_path))
    ring.edit_target("pack", "screen:a", "snow")
    response = client.post("/library/packs/pack/ring/version",
                           json={"target": "screen:a", "index": -1})
    assert response.status_code == 200
    assert load_manifest(Path(tmp_path) / "pack").ring.versions["screen:a"].current == -1
    bad = client.post("/library/packs/pack/ring/version",
                      json={"target": "screen:a", "index": 9})
    assert bad.status_code == 400
    status = client.get("/library/packs/pack/ring").json()
    assert status["screen_files"][0] == "a/generate.png"
    assert status["ring_height"] == 64 and status["screen_padding"] == [0, 0]
    assert "screen:a" in status["versions"]
