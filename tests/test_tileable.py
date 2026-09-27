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


def test_circular_transformer_is_shift_equivariant_and_restores_shape():
    model = _FakeDiT().eval()
    x = torch.rand(3, 1, 16, 20)
    with torch.no_grad(), circular_transformer(model, "horizontal", context=4):
        out = model([x], None, None)[0][0]
        shifted = model([torch.roll(x, 7, dims=-1)], None, None)[0][0]
    assert out.shape == x.shape
    assert torch.allclose(shifted, torch.roll(out, 7, dims=-1), atol=1e-5)


def test_circular_transformer_removes_hooks_on_exit():
    model = _FakeDiT().eval()
    x = torch.rand(3, 1, 16, 20)
    with torch.no_grad():
        baseline = model([x], None, None)[0][0]
        with circular_transformer(model, "both", context=4):
            model([x], None, None)
        restored = model([x], None, None)[0][0]
    assert torch.equal(baseline, restored)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        validate_tile_mode("diagonal")
