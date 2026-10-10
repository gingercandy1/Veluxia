"""循环拼接纯图像及资源包流程契约，不加载权重或 GPU。"""
import json
import threading
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from src.backend.core.collection import library, ring
from src.backend.core.collection.manifest import load_manifest, save_manifest
from src.backend.core.collection.seamless_loop import (
    BAND,
    FEATHER,
    HALF_WINDOW,
    _tour_cost,
    apply_patch,
    band_alpha,
    build_window,
    edge_cost_matrix,
    edge_strips,
    extract_patch,
    narrow_merge,
    order_loop,
)
from src.backend.core.model_base import GeneratorFactory
from src.shared.schemas import (
    CollectionItem,
    Manifest,
    RingGenerateRequest,
    SeamState,
    StepState,
)


def test_order_loop_reduces_cost_and_rejects_single_item():
    strips = []
    for right_color, left_color in ((0, 200), (200, 100), (100, 0)):
        right = np.full((2, 3), right_color, dtype=np.float32)
        left = np.full((2, 3), left_color, dtype=np.float32)
        strips.append((right, left))
    costs = edge_cost_matrix(strips)
    order = order_loop(costs)
    assert _tour_cost(costs, order) < _tour_cost(costs, [0, 1, 2])
    with pytest.raises(ValueError, match="至少要两张"):
        order_loop(np.zeros((1, 1)))


def test_edge_strips_read_rgb_edges_and_reject_mismatched_heights(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (20, 8), (10, 30, 50)).save(first)
    Image.new("RGB", (20, 9), (10, 30, 50)).save(second)
    right, left = edge_strips(first)
    assert right.shape == left.shape == (8, 3)
    assert np.all(right == (10, 30, 50))
    with pytest.raises(ValueError, match="高度必须一致"):
        edge_cost_matrix([edge_strips(first), edge_strips(second)])


def test_window_patch_alpha_and_pixel_mapping():
    left = Image.new("RGB", (700, 40), "red")
    right = Image.new("RGB", (650, 40), "blue")
    window = build_window(left, right)
    assert window.size == (HALF_WINDOW * 2, 40)
    assert window.getpixel((0, 0)) == (255, 0, 0)
    assert window.getpixel((HALF_WINDOW, 0)) == (0, 0, 255)

    alpha = band_alpha(40)
    assert alpha.shape == (40, BAND + 2 * FEATHER, 1)
    assert alpha[0, 0, 0] == 0
    assert alpha[0, FEATHER - 1, 0] == 1
    assert np.all(alpha[:, FEATHER:FEATHER + BAND] == 1)
    assert alpha[0, -1, 0] == 0

    extracted = extract_patch(window)
    assert extracted.mode == "RGBA"
    assert extracted.size == (BAND + 2 * FEATHER, 40)
    assert extracted.getpixel((0, 0))[3] == 0
    assert extracted.getpixel((FEATHER, 0))[3] == 255
    patch_pixels = np.zeros((40, BAND + 2 * FEATHER, 4), dtype=np.uint8)
    patch_pixels[:, :152, :3] = (255, 0, 0)
    patch_pixels[:, 152:, :3] = (0, 255, 0)
    patch_pixels[:, :, 3] = np.round(alpha[:, :, 0] * 255).astype(np.uint8)
    patch = Image.fromarray(patch_pixels, "RGBA")

    source = Image.new("RGB", (700, 40), "blue")
    right_result = apply_patch(source, patch, "right")
    assert np.array_equal(
        np.asarray(right_result.crop((0, 0, 548, 40))),
        np.asarray(source.crop((0, 0, 548, 40))),
    )
    assert right_result.getpixel((699, 20)) == (255, 0, 0)

    left_result = apply_patch(source, patch, "left")
    assert np.array_equal(
        np.asarray(left_result.crop((152, 0, 700, 40))),
        np.asarray(source.crop((152, 0, 700, 40))),
    )
    assert left_result.getpixel((0, 20)) == (0, 255, 0)
    assert left_result.getpixel((151, 20)) == (0, 0, 255)


def test_window_rejects_too_narrow_images():
    with pytest.raises(ValueError, match="至少为 512"):
        build_window(Image.new("RGB", (511, 20)), Image.new("RGB", (512, 20)))


def test_narrow_merge_changes_only_transition_band():
    original = Image.new("RGB", (1024, 20), "black")
    harmonized = Image.new("RGB", (1024, 20), "white")
    merged = narrow_merge(original, harmonized)
    assert merged.getpixel((319, 10)) == (0, 0, 0)
    assert merged.getpixel((320, 10)) == (0, 0, 0)
    assert merged.getpixel((512, 10)) == (255, 255, 255)
    assert merged.getpixel((704, 10)) == (0, 0, 0)


def _make_pack(tmp_path, monkeypatch, item_ids=("a", "b", "c")):
    monkeypatch.setattr(library, "library_root", lambda: tmp_path)
    directory = tmp_path / "pack"
    items = []
    for index, item_id in enumerate(item_ids):
        item_dir = directory / item_id
        item_dir.mkdir(parents=True)
        source = item_dir / "generate.png"
        Image.new("RGB", (700, 64), (index * 70, 20, 100)).save(source)
        items.append(CollectionItem(
            id=item_id,
            prompt=f"景物 {item_id}",
            fields={"view": "side view"},
            steps={"generate": StepState(status="done", outputs=[f"{item_id}/generate.png"])},
        ))
    manifest = Manifest(
        id="pack",
        name="test ring",
        type="scene",
        template="scene_background",
        style={"prompt": "painterly style"},
        items=items,
    )
    save_manifest(directory, manifest)
    return directory, manifest


def test_plan_ring_checks_completion_and_preserves_only_adjacent_seams(tmp_path, monkeypatch):
    directory, manifest = _make_pack(tmp_path, monkeypatch)
    first = ring.plan_ring("pack")
    order = first.ring.order
    valid = f"{order[0]}>{order[1]}"
    invalid = f"{order[0]}>{order[2]}"
    preserved = SeamState(candidates=[{
        "id": "c1", "file": "ring/a_b/c1.png", "seed": 1, "prompt": "prompt",
        "left_sig": "left", "right_sig": "right", "harmonize": 0,
    }], adopted="c1")
    first.ring.seams[valid] = preserved
    first.ring.seams[invalid] = SeamState(adopted="stale")
    save_manifest(directory, first)
    planned = ring.plan_ring("pack")
    assert planned.ring.order == order
    assert planned.ring.seams[valid] == preserved
    assert invalid not in planned.ring.seams

    manifest.items[1].steps["generate"].status = "pending"
    save_manifest(directory, manifest)
    with pytest.raises(ValueError, match="条目 b"):
        ring.plan_ring("pack")


def test_plan_ring_uses_only_chosen_items_and_keeps_them_on_replan(tmp_path, monkeypatch):
    directory, manifest = _make_pack(tmp_path, monkeypatch, item_ids=("a", "b", "c", "d"))
    # 没选上的条目 generate 没完成也不影响规划
    manifest.items[3].steps["generate"].status = "pending"
    save_manifest(directory, manifest)

    planned = ring.plan_ring("pack", ["c", "a", "a"])
    assert sorted(planned.ring.order) == ["a", "c"]
    assert set(planned.ring.seams) == {"a>c", "c>a"}

    # 不传成员时沿用上次的，不会把 b、d 拉回来
    replanned = ring.plan_ring("pack")
    assert sorted(replanned.ring.order) == ["a", "c"]


def test_plan_ring_rejects_unknown_or_too_few_items(tmp_path, monkeypatch):
    _make_pack(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="没有这些条目：x"):
        ring.plan_ring("pack", ["a", "x"])
    with pytest.raises(ValueError, match="至少需要两个条目"):
        ring.plan_ring("pack", ["a"])


def test_set_order_allows_repeats_and_keeps_seams_between_unchanged_neighbours(
        tmp_path, monkeypatch):
    directory, manifest = _make_pack(tmp_path, monkeypatch)
    ordered = ring.set_order("pack", ["a", "b", "a", "c"])
    assert ordered.ring.order == ["a", "b", "a", "c"]
    # 相邻关系决定接缝；同样的 a>b 出现一次就够了
    assert set(ordered.ring.seams) == {"a>b", "b>a", "a>c", "c>a"}

    kept = SeamState(adopted="c1")
    ordered.ring.seams["a>b"] = kept
    ordered.ring.cuts, ordered.ring.cuts_sig = [10], "sig"
    save_manifest(directory, ordered)
    # 挪动 c：a>b 还在，顺序变了所以切点作废
    moved = ring.set_order("pack", ["a", "b", "c", "a"])
    assert moved.ring.seams["a>b"] == kept
    assert moved.ring.cuts == []
    # 顺序没变，切点保留
    ordered.ring.cuts = [10]
    save_manifest(directory, ordered)
    same = ring.set_order("pack", ["a", "b", "a", "c"])
    assert same.ring.cuts == [10]


def test_set_order_rejects_unknown_unfinished_and_too_short(tmp_path, monkeypatch):
    directory, manifest = _make_pack(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="没有这些条目：x"):
        ring.set_order("pack", ["a", "x"])
    with pytest.raises(ValueError, match="至少需要两个条目"):
        ring.set_order("pack", ["a"])
    manifest.items[1].steps["generate"].status = "pending"
    save_manifest(directory, manifest)
    with pytest.raises(ValueError, match="条目 b"):
        ring.set_order("pack", ["a", "b"])


def test_render_keeps_a_separate_image_for_each_repeat_of_a_screen(tmp_path, monkeypatch):
    directory, _ = _make_pack(tmp_path, monkeypatch, item_ids=("a", "b"))
    manifest = ring.set_order("pack", ["a", "b", "a", "b"])
    fake = _FakeGenerator(directory)
    _mock_generator(monkeypatch, fake)
    generated = ring.generate_seams("pack", ["a>b"], cancel_event=threading.Event())
    ring.adopt_seam("pack", "a>b", generated.ring.seams["a>b"].candidates[0].id)
    ring.render_ring("pack")

    rendered = ring.ring_status("pack")["rendered"]
    # 四个位置都有合成图；邻居相同的位置共用一个文件（a、b 各出现两次，邻居都一样）
    assert len(rendered) == 4 and all(rendered)
    assert rendered[0] == rendered[2] and rendered[1] == rendered[3]
    assert manifest.ring.order == ["a", "b", "a", "b"]

    # 位置变了、邻居也变了而还没重新合成：长图的素材缺失要明确报错
    ring.set_order("pack", ["a", "a", "b", "b"])
    assert ring.ring_status("pack")["rendered"].count("") >= 1
    with pytest.raises(ValueError, match="还没合成"):
        ring.auto_cuts("pack")


class _FakeGenerator:
    def __init__(self, directory, fail_after=None):
        self.directory = directory
        self.calls = []
        self.counts_before_call = []
        self.fail_after = fail_after
        self.cancel_event = None
        self.progress_callback = None

    def ensure_model_loaded(self):
        pass

    def check_cancelled(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise RuntimeError("cancelled")

    def repaint_seam(self, window, prompt, seed, harmonize):
        self.calls.append((prompt, seed, harmonize))
        if self.progress_callback is not None:
            self.progress_callback(0.5)
        manifest = load_manifest(self.directory)
        self.counts_before_call.append(
            sum(len(state.candidates) for state in manifest.ring.seams.values()))
        if self.fail_after and len(self.calls) > self.fail_after:
            raise RuntimeError("fake repaint failed")
        return Image.new("RGB", window.size, (240, 10, 20)), None


@contextmanager
def _fake_lease(fake):
    yield fake


def _mock_generator(monkeypatch, fake):
    monkeypatch.setattr(
        GeneratorFactory,
        "acquire",
        classmethod(lambda cls, factory_type, model_name: _fake_lease(fake)),
    )


def test_generate_candidates_prunes_old_files_and_saves_each_seam(tmp_path, monkeypatch):
    directory, _ = _make_pack(tmp_path, monkeypatch)
    manifest = ring.plan_ring("pack")
    key = f"{manifest.ring.order[0]}>{manifest.ring.order[1]}"
    fake = _FakeGenerator(directory)
    _mock_generator(monkeypatch, fake)

    for _ in range(5):
        result = ring.generate_seams("pack", [key], cancel_event=threading.Event())

    state = result.ring.seams[key]
    assert [candidate.id for candidate in state.candidates] == ["c1", "c3", "c4", "c5"]
    assert state.adopted == "c1"
    assert fake.counts_before_call == [0, 1, 2, 3, 4]
    assert fake.calls[0][0].startswith(f"景物 {manifest.ring.order[0]}")
    assert fake.calls[0][2] == 0.3
    assert not (directory / "ring" / key.replace(">", "_") / "c2.png").exists()
    assert all((directory / candidate.file).is_file() for candidate in state.candidates)
    assert load_manifest(directory).ring.seams[key].adopted == "c1"


def test_generate_reports_progress_per_seam_and_detaches_the_callback(tmp_path, monkeypatch):
    directory, _ = _make_pack(tmp_path, monkeypatch)
    manifest = ring.plan_ring("pack")
    keys = [f"{manifest.ring.order[index]}>{manifest.ring.order[(index + 1) % 3]}"
            for index in range(2)]
    fake = _FakeGenerator(directory)
    _mock_generator(monkeypatch, fake)
    events = []
    ring.generate_seams("pack", keys, cancel_event=threading.Event(),
                        on_progress=lambda key, fraction: events.append((key, fraction)))
    assert events == [(keys[0], 0.0), (keys[0], 0.5), (keys[1], 0.0), (keys[1], 0.5)]
    # 单驻留的生成器会被下一个任务复用：回调不能留着指向已结束的任务
    assert fake.progress_callback is None


def test_failed_generation_keeps_completed_seam_progress(tmp_path, monkeypatch):
    directory, _ = _make_pack(tmp_path, monkeypatch)
    manifest = ring.plan_ring("pack")
    keys = [f"{manifest.ring.order[index]}>{manifest.ring.order[(index + 1) % 3]}"
            for index in range(2)]
    fake = _FakeGenerator(directory, fail_after=1)
    _mock_generator(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="fake repaint failed"):
        ring.generate_seams("pack", keys, cancel_event=threading.Event())
    saved = load_manifest(directory)
    assert len(saved.ring.seams[keys[0]].candidates) == 1
    assert saved.ring.seams[keys[1]].candidates == []


def test_adopt_render_and_stale_detection(tmp_path, monkeypatch):
    directory, _ = _make_pack(tmp_path, monkeypatch, item_ids=("left", "right"))
    manifest = ring.plan_ring("pack")
    key = f"{manifest.ring.order[0]}>{manifest.ring.order[1]}"
    fake = _FakeGenerator(directory)
    _mock_generator(monkeypatch, fake)
    generated = ring.generate_seams("pack", [key], cancel_event=threading.Event())
    candidate = generated.ring.seams[key].candidates[0]
    # 生成接缝时已经把采用的候选贴回了相邻两屏，不必另外合成
    generated_slots = ring.ring_status("pack")["rendered"]
    assert len(generated_slots) == 2 and all(generated_slots)
    assert ring.ring_status("pack")["seams"][key].stale is False
    adopted = ring.adopt_seam("pack", key, candidate.id)
    assert adopted.ring.seams[key].adopted == candidate.id
    with pytest.raises(ValueError, match="没有候选"):
        ring.adopt_seam("pack", key, "missing")

    ring.render_ring("pack")
    rendered = ring.ring_status("pack")["rendered"]
    assert rendered == generated_slots
    left_id, right_id = key.split(">")
    left_original = np.asarray(Image.open(directory / left_id / "generate.png"))
    left_rendered = np.asarray(Image.open(directory / rendered[0]))
    right_original = np.asarray(Image.open(directory / right_id / "generate.png"))
    right_rendered = np.asarray(Image.open(directory / rendered[1]))
    assert np.array_equal(left_rendered[:, :-152], left_original[:, :-152])
    assert np.array_equal(right_rendered[:, 152:], right_original[:, 152:])
    assert not np.array_equal(left_rendered[:, -1], left_original[:, -1])
    assert not np.array_equal(right_rendered[:, 0], right_original[:, 0])

    source = directory / left_id / "generate.png"
    Image.new("RGB", (700, 64), "yellow").save(source)
    assert ring.ring_status("pack")["seams"][key].stale is True


def test_old_manifest_without_ring_remains_readable(tmp_path):
    directory = tmp_path / "old_pack"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "id": "old",
        "template": "scene_background",
        "items": [],
    }), encoding="utf-8")
    assert load_manifest(directory).ring is None
    assert RingGenerateRequest(pack_id="pack", keys=None).keys is None


def test_extra_prompt_is_appended_after_the_style_before_the_constraints(tmp_path, monkeypatch):
    directory, _ = _make_pack(tmp_path, monkeypatch, item_ids=("left", "right"))
    manifest = ring.plan_ring("pack")
    key = f"{manifest.ring.order[0]}>{manifest.ring.order[1]}"
    fake = _FakeGenerator(directory)
    _mock_generator(monkeypatch, fake)
    ring.generate_seams("pack", [key], cancel_event=threading.Event(), extra_prompt="  a small stream  ")
    prompt = fake.calls[0][0]
    assert prompt.endswith("a small stream, no characters, no text")
    ring.generate_seams("pack", [key], cancel_event=threading.Event())
    assert "a small stream" not in fake.calls[1][0]
