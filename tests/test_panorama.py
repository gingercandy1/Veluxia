"""宽幅分窗生成的契约测试（无权重、无 GPU 也可跑）。"""
from itertools import pairwise

import pytest
import torch

from src.backend.core.image.panorama import (
    panorama_transformer,
    parse_segment_prompts,
    segment_of,
    validate_segments,
    window_starts,
)
from src.backend.core.image.tileable import circular_transformer


def _covered(starts: list[int], window: int, total: int) -> set[int]:
    return {(start + i) % total for start in starts for i in range(window)}


def test_window_starts_span_canvas_edge_to_edge():
    starts = window_starts(total=504, window=168, overlap=0.25, wrap=False)
    assert starts[0] == 0
    assert starts[-1] == 504 - 168
    assert all(start % 2 == 0 for start in starts)
    assert all(b - a < 168 for a, b in pairwise(starts))
    assert _covered(starts, 168, 504) == set(range(504))


def test_window_starts_wrap_around_the_seam():
    starts = window_starts(total=504, window=168, overlap=0.25, wrap=True)
    assert all(start % 2 == 0 for start in starts)
    # 要有窗口越过右端接回左端，首尾才会在生成时对齐
    assert any(start + 168 > 504 for start in starts)
    assert _covered(starts, 168, 504) == set(range(504))


@pytest.mark.parametrize("wrap", [False, True])
def test_window_starts_balanced_across_segments(wrap):
    # 3 段时每段分到的窗口数一样多，否则某一段在画面里会更宽
    starts = window_starts(total=504, window=168, overlap=0.25, wrap=wrap, multiple=3)
    assert len(starts) % 3 == 0
    counts = [0, 0, 0]
    for start in starts:
        counts[segment_of(start, 168, 504, 3)] += 1
    assert counts[0] == counts[1] == counts[2]
    assert _covered(starts, 168, 504) == set(range(504))


class _PointwiseDiT(torch.nn.Module):
    """逐像素的假 transformer：输出只取决于该像素、该样本的 timestep 和提示词嵌入。
    这样分窗再合回的结果应和整图一次前向完全一致，可以检验切窗、复制条件、加权合并都没错位。"""

    def forward(self, latents, timestep, cap_feats):
        return ([latent * t + feat.mean() for latent, t, feat in zip(latents, timestep, cap_feats, strict=True)],)


@pytest.mark.parametrize("wrap", [False, True])
def test_panorama_matches_full_forward(wrap):
    model = _PointwiseDiT()
    latents = [torch.rand(4, 1, 6, 40), torch.rand(4, 1, 6, 40)]
    timestep = torch.tensor([0.3, 0.7])
    feats = [torch.rand(5, 8), torch.rand(3, 8)]
    expected = model(latents, timestep, feats)[0]
    with panorama_transformer(model, window=12, wrap=wrap):
        merged = model(latents, timestep, feats)[0]
    assert len(merged) == 2
    for out, ref in zip(merged, expected, strict=True):
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)


def test_panorama_composes_with_vertical_circular():
    model = _PointwiseDiT()
    latents = [torch.rand(4, 1, 8, 40)]
    timestep, feats = torch.tensor([0.5]), [torch.rand(5, 8)]
    expected = model(latents, timestep, feats)[0][0]
    with circular_transformer(model, "vertical", context=2), \
            panorama_transformer(model, window=12, wrap=True):
        out = model(latents, timestep, feats)[0][0]
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5)


class _CountingDiT(_PointwiseDiT):
    def __init__(self):
        super().__init__()
        self.widths = []

    def forward(self, latents, timestep, cap_feats):
        self.widths.append(latents[0].shape[-1])
        return super().forward(latents, timestep, cap_feats)


def test_panorama_runs_one_window_per_forward():
    # 8GB 下窗口不能拼成一批：每次前向只能是一个窗口宽
    model = _CountingDiT()
    with panorama_transformer(model, window=12):
        model([torch.rand(4, 1, 6, 40)], torch.tensor([0.5]), [torch.rand(5, 8)])
    assert len(model.widths) > 1
    assert set(model.widths) == {12}


class _FeatRecordingDiT(_PointwiseDiT):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, latents, timestep, cap_feats):
        self.calls.append(cap_feats[0])
        return super().forward(latents, timestep, cap_feats)


def test_panorama_uses_each_segments_prompt():
    model = _FeatRecordingDiT()
    segment_feats = [[torch.full((2, 8), float(i))] for i in range(3)]
    with panorama_transformer(model, window=12, segment_feats=segment_feats):
        model([torch.rand(4, 1, 6, 36)], torch.tensor([0.5]), [torch.rand(5, 8)])
    used = [int(feat[0, 0]) for feat in model.calls]
    # 从左到右依次经过三段，且每段都至少有一个窗口
    assert used == sorted(used)
    assert set(used) == {0, 1, 2}


def test_segment_of_wraps_last_window_to_its_center():
    # 绕圈窗口的中心越过右端时，归到开头那一段
    assert segment_of(start=30, window=12, total=36, segments=3) == 0
    assert segment_of(start=0, window=12, total=36, segments=3) == 0
    assert segment_of(start=24, window=12, total=36, segments=3) == 2


@pytest.mark.parametrize(("raw", "expected"), [
    ("森林 | 村庄 |城堡", ["森林", "村庄", "城堡"]),
    ("森林｜村庄", ["森林", "村庄"]),
    ("", []),
    (["a", " "], ["a"]),
])
def test_parse_segment_prompts(raw, expected):
    assert parse_segment_prompts(raw) == expected


def test_panorama_restores_forward_on_exit():
    model = _PointwiseDiT()
    with panorama_transformer(model, window=12):
        pass
    assert "forward" not in model.__dict__


@pytest.mark.parametrize("value", ["0", "5", 0])
def test_segments_out_of_range_rejected(value):
    with pytest.raises(ValueError):
        validate_segments(value)


def test_segments_accept_template_string():
    assert validate_segments("3") == 3
