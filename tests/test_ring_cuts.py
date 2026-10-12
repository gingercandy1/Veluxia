"""循环拼接切段和分段分层：纯计算 + 资源包流程契约，深度和分层模型都用假的代替，不加载权重。"""
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from src.backend.core.collection import library, ring
from src.backend.core.collection.manifest import save_manifest
from src.backend.core.collection.ring_cuts import (
    MAX_SEGMENT,
    MIN_SEGMENT,
    column_cost,
    export_segments,
    plan_cuts,
    segment_widths,
    validate_cuts,
)
from src.backend.core.model_base import GeneratorFactory
from src.shared.schemas import CollectionItem, Manifest, StepState


def test_column_cost_marks_near_objects_not_the_ground():
    depth = np.full((10, 200), 0.2, np.float32)
    depth[7:, :] = 0.9            # 地面整行都近
    depth[:, 100:110] = 0.8       # 一根树干从上到下
    cost = column_cost(depth)
    assert cost[105] > 10 * cost[30]
    # 地面那几行每列一样近，减掉行中位数后对代价没有贡献
    assert cost[30] == pytest.approx(0.0, abs=1e-6)


def test_plan_cuts_avoids_objects_and_respects_widths():
    width = 3600  # 三个空隙间距都是 1200，包括从最后一个绕回第一个
    cost = np.full(width, 1.0)
    for gap in (300, 1500, 2700):
        cost[gap - 20:gap + 20] = 0.0
    cuts = plan_cuts(cost)
    # 空隙里每个位置代价都是 0，落在哪都对；关键是没有切在物体上
    assert len(cuts) == 3
    assert all(any(abs(cut - gap) <= 20 for gap in (300, 1500, 2700)) for cut in cuts)
    assert all(MIN_SEGMENT <= w <= MAX_SEGMENT for w in segment_widths(cuts, width))


def test_plan_cuts_short_ring_is_one_segment_and_impossible_width_errors():
    cost = np.ones(1200)
    cost[400] = 0.0
    assert plan_cuts(cost) == [400]
    # 1600 宽：一段太宽、两段太窄
    with pytest.raises(ValueError, match="没法切成"):
        plan_cuts(np.ones(1600))


def test_validate_cuts_sorts_and_accepts_any_segment_width():
    assert validate_cuts([1800, 200, 1000, 200], 2400) == [200, 1000, 1800]
    # 段宽不限制：窄到 100 像素、宽到整条环，只要切点在范围内就接受
    assert validate_cuts([200, 300, 1300], 2400) == [200, 300, 1300]
    assert validate_cuts([0], 2400) == [0]
    with pytest.raises(ValueError, match="超出长图范围"):
        validate_cuts([200, 2400], 2400)
    with pytest.raises(ValueError, match="至少需要一个切点"):
        validate_cuts([], 2400)


def test_export_segments_wraps_around_the_ring(tmp_path):
    image = Image.new("RGB", (1000, 4))
    for x in range(1000):
        image.putpixel((x, 0), (x % 256, x // 256, 0))
    (tmp_path / "stale.png").write_bytes(b"old")
    paths = export_segments(image, [100, 600], tmp_path)
    assert [p.name for p in paths] == ["01.png", "02.png"]
    assert not (tmp_path / "stale.png").exists()
    first, second = (Image.open(p) for p in paths)
    assert first.width == 500 and second.width == 500
    # 第二段从 600 走到末尾 999，再接上开头 0～99
    assert second.getpixel((0, 0)) == (600 % 256, 600 // 256, 0)
    assert second.getpixel((399, 0)) == (999 % 256, 999 // 256, 0)
    assert second.getpixel((400, 0)) == (0, 0, 0)


def _rendered_pack(tmp_path, monkeypatch, widths=(1200, 1200)):
    monkeypatch.setattr(library, "library_root", lambda: tmp_path)
    directory = tmp_path / "pack"
    items = []
    for index, width in enumerate(widths):
        item_id = f"s{index}"
        (directory / item_id).mkdir(parents=True)
        Image.new("RGB", (width, 64), (index * 60, 30, 90)).save(directory / item_id / "generate.png")
        items.append(CollectionItem(id=item_id, prompt=item_id, steps={
            "generate": StepState(status="done", outputs=[f"{item_id}/generate.png"])}))
    save_manifest(directory, Manifest(id="pack", type="scene", template="scene_background",
                                      items=items))
    ring.plan_ring("pack")
    return directory


def _fake_depth(monkeypatch, gaps):
    """假深度模型；代价直接钉成"只有 gaps 处为 0"，代价函数本身另有单测。"""
    calls = []

    def estimate(image, cancel_event=None):
        calls.append(image.size)
        return np.zeros((image.height, image.width), np.float32)

    def cost(depth):
        values = np.ones(depth.shape[1])
        values[list(gaps)] = 0.0
        return values

    monkeypatch.setattr(ring, "estimate_ring_depth", estimate)
    monkeypatch.setattr(ring, "column_cost", cost)
    return calls


def test_auto_cuts_needs_rendered_screens(tmp_path, monkeypatch):
    _rendered_pack(tmp_path, monkeypatch)
    _fake_depth(monkeypatch, [])
    with pytest.raises(ValueError, match="还没合成"):
        ring.auto_cuts("pack")


def test_auto_cut_confirm_export_and_stale(tmp_path, monkeypatch):
    directory = _rendered_pack(tmp_path, monkeypatch)
    ring.render_ring("pack")
    # 两个空隙相距 1200：候选间隔是 8，从第一个空隙出发正好能落到第二个
    calls = _fake_depth(monkeypatch, [504, 1704])

    manifest = ring.auto_cuts("pack")
    assert calls == [(2400, 64)]
    # 自动切割只给建议，确认之前不碰已确认的切点
    assert manifest.ring.suggested_cuts == [504, 1704]
    assert manifest.ring.cuts == [] and manifest.ring.segments == []
    status = ring.ring_status("pack")
    assert status["suggested_cuts"] == [504, 1704] and status["cuts"] == []
    assert status["cuts_stale"] is False and status["segments"] == []

    confirmed = ring.confirm_cuts("pack", [1700, 500])
    assert confirmed.ring.cuts == [500, 1700]
    assert confirmed.ring.suggested_cuts == []
    assert confirmed.ring.segments == ["ring/segments/01.png", "ring/segments/02.png"]
    assert [Image.open(directory / p).width for p in confirmed.ring.segments] == [1200, 1200]
    assert ring.ring_status("pack")["segments"] == confirmed.ring.segments

    # 顺序没变的重新规划保留切点
    assert ring.plan_ring("pack").ring.cuts == [500, 1700]

    # 合成图变了：切点失效
    Image.new("RGB", (1200, 64), "yellow").save(directory / ring.ring_status("pack")["rendered"][0])
    assert ring.ring_status("pack")["cuts_stale"] is True

    # 段宽不再限制：窄段也能确认，由用户自己判断是否合适
    confirmed = ring.confirm_cuts("pack", [500, 600])
    assert confirmed.ring.cuts == [500, 600]


class _FakeLayered:
    """假 Qwen-Layered：记录编码和去噪的顺序，产物写到临时目录让 ring 搬进资源包。"""

    def __init__(self, tmp_path, fail_on_denoise=None):
        self.tmp_path = tmp_path
        self.events = []
        self.fail_on_denoise = fail_on_denoise
        self.cancel_event = None
        self.params = {}

    def ensure_model_loaded(self):
        pass

    def parse_params(self, raw):
        self.params = raw

    def save_prompt_embeds(self):
        self.events.append(("encode", self.params["content"]))
        path = self.tmp_path / f"embeds_{len(self.events)}.pt"
        path.write_bytes(b"embeds")
        return path

    async def generate(self):
        width = Image.open(self.params["reference_image"]).width
        self.events.append(("denoise", width))
        if self.fail_on_denoise == len([e for e in self.events if e[0] == "denoise"]):
            raise RuntimeError("fake denoise failed")
        paths = []
        for index in range(1, self.params["layers"] + 1):
            path = self.tmp_path / f"layer_{len(self.events)}_{index}.png"
            Image.new("RGBA", (width // 2, 32), (index * 50, 0, 0, 255)).save(path)
            paths.append(path)
        return paths


def _layered_pack(tmp_path, monkeypatch, fake):
    directory = _rendered_pack(tmp_path, monkeypatch)
    ring.render_ring("pack")
    _fake_depth(monkeypatch, [504, 1704])
    ring.auto_cuts("pack")
    ring.confirm_cuts("pack", [504, 1704])
    monkeypatch.setattr(ring, "load_template", lambda _name: _NoLayerTemplate())
    monkeypatch.setattr(GeneratorFactory, "acquire",
                        classmethod(lambda cls, factory_type, model_name: _lease(fake)))
    return directory


class _NoLayerTemplate:
    def step_ids(self):
        return ["generate"]


@contextmanager
def _lease(fake):
    yield fake


def test_layer_segments_encodes_all_first_then_denoises_and_resumes(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    fake = _FakeLayered(work, fail_on_denoise=2)
    directory = _layered_pack(tmp_path, monkeypatch, fake)

    # 第二段去噪失败：第一段的图层和两段的编码都已落盘
    with pytest.raises(RuntimeError, match="fake denoise failed"):
        ring.layer_segments("pack")
    assert [e[0] for e in fake.events] == ["encode", "encode", "denoise", "denoise"]
    # 段 1（504→1704）只盖住 s0 的后半和 s1 的前半，描述里两屏都有
    assert fake.events[0][1].startswith("s0, s1")
    status = ring.ring_status("pack")
    assert len(status["segment_layers"][0]) == ring.LAYER_COUNT
    assert status["segment_layers"][1] == []

    # 再点一次：只做剩下的第二段，而且复用已有编码，不再加载编码器
    fake.events.clear()
    fake.fail_on_denoise = None
    manifest = ring.layer_segments("pack")
    assert [e[0] for e in fake.events] == ["denoise"]
    status = ring.ring_status("pack")
    assert all(len(layers) == ring.LAYER_COUNT for layers in status["segment_layers"])
    first = manifest.ring.segment_layers["504+1200"]
    assert first.layers[0] == "ring/layers/504_1200/layer_1.png"
    assert (directory / first.layers[3]).is_file()

    # 全部完成后再点：什么都不做
    fake.events.clear()
    ring.layer_segments("pack")
    assert fake.events == []

    # 指定重做第 1 段：重新编码 + 去噪
    ring.layer_segments("pack", [0])
    assert [e[0] for e in fake.events] == ["encode", "denoise"]
    with pytest.raises(ValueError, match="没有这些段：3"):
        ring.layer_segments("pack", [2])


def test_layer_segments_reports_stage_current_segment_and_remaining(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    fake = _FakeLayered(work, fail_on_denoise=2)
    _layered_pack(tmp_path, monkeypatch, fake)
    reports = []

    def report(stage, index, remaining):
        reports.append((stage, index, remaining))

    # 先把所有段编码，再逐段去噪；剩余段包含当前段，做完一段就少一段
    with pytest.raises(RuntimeError, match="fake denoise failed"):
        ring.layer_segments("pack", on_progress=report)
    assert reports == [
        ("encode", 0, [0, 1]), ("encode", 1, [1]),
        ("denoise", 0, [0, 1]), ("denoise", 1, [1])]

    # 续跑只报剩下的段，而且已有编码不再重编，所以没有 encode
    reports.clear()
    fake.fail_on_denoise = None
    ring.layer_segments("pack", on_progress=report)
    assert reports == [("denoise", 1, [1])]

    # 指定重做第 1 段：序号是它在全部段里的位置，不是在待做列表里的位置
    reports.clear()
    ring.layer_segments("pack", [1], on_progress=report)
    assert reports == [("encode", 1, [1]), ("denoise", 1, [1])]


def test_confirming_cuts_clears_all_layers_even_for_untouched_segments(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    directory = _rendered_pack(tmp_path, monkeypatch, widths=(1200, 1200, 1200))
    ring.render_ring("pack")
    _fake_depth(monkeypatch, [0, 1200, 2400])
    ring.auto_cuts("pack")
    ring.confirm_cuts("pack", [0, 1200, 2400])
    fake = _FakeLayered(work)
    monkeypatch.setattr(ring, "load_template", lambda _name: _NoLayerTemplate())
    monkeypatch.setattr(GeneratorFactory, "acquire",
                        classmethod(lambda cls, factory_type, model_name: _lease(fake)))
    ring.layer_segments("pack")

    # 只挪第三个切点：新规则下第一段 0+1200 的分层也一并清空，目录全部删除
    manifest = ring.confirm_cuts("pack", [0, 1200, 2300])
    assert manifest.ring.segment_layers == {}
    layers = ring.ring_status("pack")["segment_layers"]
    assert layers == [[], [], []]
    assert not (directory / "ring" / "layers" / "0_1200").exists()
    assert not (directory / "ring" / "layers" / "1200_1200").exists()

    fake.events.clear()
    ring.layer_segments("pack")
    assert [e[0] for e in fake.events] == ["encode"] * 3 + ["denoise"] * 3


def test_layer_segments_needs_confirmed_fresh_cuts(tmp_path, monkeypatch):
    directory = _rendered_pack(tmp_path, monkeypatch)
    ring.render_ring("pack")
    with pytest.raises(ValueError, match="请先确认切段"):
        ring.layer_segments("pack")
    _fake_depth(monkeypatch, [504, 1704])
    ring.auto_cuts("pack")
    ring.confirm_cuts("pack", [504, 1704])
    Image.new("RGB", (1200, 64), "yellow").save(directory / ring.ring_status("pack")["rendered"][0])
    with pytest.raises(ValueError, match="合成图变了"):
        ring.layer_segments("pack")
