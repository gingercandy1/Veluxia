"""无缝平铺的契约测试（无权重、无 GPU 也可跑）。"""
import pytest
import torch

from src.backend.core.image.tileable import (
    apply_tile_mode,
    circular_transformer,
    validate_tile_mode,
)


def _conv_stack() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.Conv2d(4, 3, 3, padding=1),
    ).eval()


def test_both_axes_conv_is_shift_equivariant():
    # 循环填充下，输入平移后再卷积 == 卷积后再平移；零填充则边缘会不一致
    model = _conv_stack()
    x = torch.rand(1, 3, 16, 16)
    apply_tile_mode([model], "both")
    with torch.no_grad():
        shifted_first = model(torch.roll(x, shifts=(5, 7), dims=(2, 3)))
        shifted_after = torch.roll(model(x), shifts=(5, 7), dims=(2, 3))
    assert torch.allclose(shifted_first, shifted_after, atol=1e-5)


def test_horizontal_only_wraps_width_not_height():
    model = _conv_stack()
    x = torch.rand(1, 3, 16, 16)
    apply_tile_mode([model], "horizontal")
    with torch.no_grad():
        width_shift = model(torch.roll(x, 7, dims=3)) - torch.roll(model(x), 7, dims=3)
        height_shift = model(torch.roll(x, 5, dims=2)) - torch.roll(model(x), 5, dims=2)
    assert width_shift.abs().max() < 1e-5
    assert height_shift.abs().max() > 1e-3


def test_off_restores_original_output():
    model = _conv_stack()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        baseline = model(x)
        apply_tile_mode([model], "both")
        apply_tile_mode([model], "off")
        restored = model(x)
    assert torch.equal(baseline, restored)


class _FakeDiT(torch.nn.Module):
    """模拟 Z-Image transformer 的调用约定：输入 latent 列表 (C, F, H, W)，输出同形状列表。
    卷积核能看到左右邻居，所以不补边时首尾两列互相看不到。"""

    def __init__(self):
        super().__init__()
        self.conv = _conv_stack()

    def forward(self, latents, timestep, prompt_embeds):
        return ([self.conv(latent.squeeze(1).unsqueeze(0)).squeeze(0).unsqueeze(1)
                 for latent in latents],)


class _PointwiseDiT(torch.nn.Module):
    """逐像素的假 transformer：平移前向再移回应和一次前向完全一致，用来检验平移、加权合并没错位。"""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, latents, timestep, prompt_embeds):
        self.shapes.append(latents[0].shape)
        return ([latent * 2 + 1 for latent in latents],)


@pytest.mark.parametrize(("mode", "passes"), [("horizontal", 2), ("vertical", 2), ("both", 4)])
def test_circular_transformer_merges_without_duplicating_tokens(mode, passes):
    # 每次前向都是原尺寸：不补边，首尾 token 不会在序列里出现两次
    model = _PointwiseDiT()
    latents = [torch.rand(3, 1, 16, 20), torch.rand(3, 1, 16, 20)]
    expected = [latent * 2 + 1 for latent in latents]
    with circular_transformer(model, mode):
        merged = model(latents, None, None)[0]
    assert model.shapes == [latents[0].shape] * passes
    for out, ref in zip(merged, expected, strict=True):
        assert torch.allclose(out, ref, atol=1e-5)


def test_circular_transformer_heals_the_seam():
    # 零填充卷积在首尾各错一圈；平移合并后应接近真正首尾相接（循环卷积）的结果
    model = _FakeDiT().eval()
    x = torch.rand(3, 1, 16, 20)
    torus = _FakeDiT().eval()
    apply_tile_mode([torus.conv], "horizontal")
    with torch.no_grad():
        reference = torus([x], None, None)[0][0]
        baseline = model([x], None, None)[0][0]
        with circular_transformer(model, "horizontal"):
            out = model([x], None, None)[0][0]
    seam = [0, 1, -2, -1]
    assert (out - reference)[..., seam].abs().max() < 0.5 * (baseline - reference)[..., seam].abs().max()


def test_circular_transformer_restores_forward_on_exit():
    model = _FakeDiT().eval()
    x = torch.rand(3, 1, 16, 20)
    with torch.no_grad():
        baseline = model([x], None, None)[0][0]
        with circular_transformer(model, "both"):
            model([x], None, None)
        restored = model([x], None, None)[0][0]
    assert "forward" not in model.__dict__
    assert torch.equal(baseline, restored)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        validate_tile_mode("diagonal")
