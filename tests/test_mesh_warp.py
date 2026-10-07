"""mesh_warp 与 frames.sway 步骤的契约测试（ADR 0008）。

无权重、无 GPU 也可跑。覆盖：多块分离元素不丢、根部不动、首尾帧一致、帧数/尺寸、
alpha 无黑边、渲染耗时、模板注册、runner 参数校验与取消。
"""
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from src.backend.core.collection.frame_steps import FramesSwayRunner
from src.backend.core.collection.steps import StepContext, registered_runners
from src.backend.core.collection.template import load_template
from src.backend.core.image_frame.mesh_warp import (
    SwayParams,
    build_mesh,
    render_sway_frames,
    sway_offsets,
)
from src.shared.schemas import CollectionItem


def _ctx(tmp_path, params, inputs=(), step_id="sway"):
    out_dir = tmp_path / "pack" / "plant"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="plant", prompt="a red flower"), prompt="a red flower, pixel art",
        negative_prompt="", inputs=list(inputs), params=params, out_dir=out_dir,
        step_id=step_id, values={"prompt": "a red flower"},
    )


def _transparent_plant(width=64, height=64, margin_bottom=0):
    """透明底植物：茎一直画到不透明区域最低一行，顶部是红色花。

    margin_bottom 是图底到植物根部之间的透明留白，用来验证根部按植物自己的底边算。
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = img.load()
    root = height - 1 - margin_bottom
    for y in range(height):
        for x in range(width):
            if height * 0.4 <= y <= root and width * 0.45 < x < width * 0.55:
                pixels[x, y] = (0, 180, 0, 255)
            elif y < height * 0.4 and abs(x - width * 0.5) < width * 0.2:
                pixels[x, y] = (255, 50, 50, 255)
    return img


def _grass(width=300, height=400, blades=5, margin_bottom=0):
    """多根互相分离的草叶，高矮不一。返回图片和每根草叶中部所在的列。"""
    canvas = np.zeros((height, width, 4), np.uint8)
    root = height - 1 - margin_bottom
    gap = width // blades
    centers = []
    for i in range(blades):
        x = gap // 2 + i * gap
        tip_y = int(height * (0.1 + 0.12 * (i % 3)))
        pts = np.array([[x - 8, root], [x + 8, root], [x, tip_y]], np.int32)
        cv2.fillConvexPoly(canvas, pts, (40, 170, 60, 255))
        centers.append(x)
    return Image.fromarray(canvas, "RGBA"), centers


def _alpha_rows(img):
    rows = np.nonzero(np.asarray(img)[..., 3].any(axis=1))[0]
    return int(rows.min()), int(rows.max())


def test_mesh_root_vertices_have_zero_height():
    img = _transparent_plant()
    mesh = build_mesh(img, density=8)
    _, bottom = _alpha_rows(img)
    root = mesh.vertices[:, 1] == bottom
    assert root.any(), "最底一行没有网格顶点"
    assert np.all(mesh.height[root] == 0)
    top = mesh.vertices[:, 1] == _alpha_rows(img)[0]
    assert top.any()
    assert np.all(mesh.height[top] == 1)


def test_mesh_height_uses_plant_bottom_not_image_bottom():
    img = _transparent_plant(height=80, margin_bottom=20)
    mesh = build_mesh(img, density=8)
    _, bottom = _alpha_rows(img)
    assert bottom < img.height - 1 - 10
    assert np.all(mesh.height[mesh.vertices[:, 1] == bottom] == 0)


def test_mesh_triangles_are_valid_indices():
    mesh = build_mesh(_transparent_plant(), density=8)
    assert mesh.triangles.shape[1] == 3
    assert mesh.triangles.dtype == np.int32
    assert mesh.triangles.min() >= 0
    assert mesh.triangles.max() < len(mesh.vertices)


def test_mesh_keeps_every_separate_element():
    img, centers = _grass()
    mesh = build_mesh(img, density=12)
    for x in centers:
        # 每根草叶的根部附近都要有顶点，否则这根草叶不会被变形，也不会被渲染
        near = np.abs(mesh.vertices[:, 0] - x) <= 8
        assert near.any(), f"x={x} 的草叶没有网格顶点"


def test_sway_keeps_all_blades_when_barely_moving():
    """摆幅趋近 0 时输出应和原图基本一致：不透明面积不丢，每根草叶都还在。"""
    img, centers = _grass()
    frames = render_sway_frames(img, SwayParams(frames=4, amplitude=0.0005, mesh_density=12))
    margin = _margin(img, frames[0])
    source = np.asarray(img)[..., 3] > 0
    out = (np.asarray(frames[0])[..., 3] > 0)[:, margin:margin + img.width]
    assert (source & out).sum() >= source.sum() * 0.97
    mid = img.height * 3 // 4
    for x in centers:
        assert out[mid, x - 2:x + 3].any(), f"x={x} 的草叶在输出里消失了"


def _margin(img, frame):
    """输出画布左右各加宽的像素数。"""
    assert frame.height == img.height
    assert (frame.width - img.width) % 2 == 0
    return (frame.width - img.width) // 2


def test_sway_frames_count_and_size():
    img = _transparent_plant()
    params = SwayParams(frames=12, amplitude=0.1, power=1.5, lag=0.6, mesh_density=8)
    frames = render_sway_frames(img, params)
    assert len(frames) == 12
    for f in frames:
        # 画布左右加宽，容纳向两侧摆出原图边界的部分
        assert _margin(img, f) >= params.amplitude * img.width
        assert f.mode == "RGBA"


def test_sway_does_not_clip_content_at_image_edge():
    """内容贴着原图左右边缘时，摆出去的部分不能被画布边界切掉。"""
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (100, 80), (40, 170, 60, 255)), (0, 20))
    # 纯斜切（无行波、无颤动）面积守恒，少了就是被画布边界切掉了
    plain = SwayParams(frames=12, amplitude=0.2, mesh_density=8, wave=0, flutter=0)
    for f in render_sway_frames(img, plain):
        assert (np.asarray(f)[..., 3] > 0).sum() >= 100 * 80 * 0.95, "摆动时内容被切掉了"
    # 带行波和颤动时内容会被局部压缩，面积不守恒；但最左、最右一列必须是空的，
    # 说明画布够宽，内容没有顶到边界被切
    for f in render_sway_frames(img, SwayParams(frames=12, amplitude=0.2, mesh_density=8,
                                                wave=3.0, flutter=1.0)):
        alpha = np.asarray(f)[..., 3] > 0
        assert not alpha[:, 0].any() and not alpha[:, -1].any()


def test_sway_actually_moves_the_tip():
    img = _transparent_plant(width=128, height=128)
    frames = render_sway_frames(img, SwayParams(frames=16, amplitude=0.1, mesh_density=8))
    centroids = []
    for f in frames:
        top = np.asarray(f)[:20, :, 3] > 0
        centroids.append(np.nonzero(top)[1].mean())
    assert max(centroids) - min(centroids) > 3


def test_sway_first_and_last_frame_match():
    """整周期 = N 帧，帧 N-1 到帧 0 的跳变应和正常相邻帧同量级（无缝循环）。"""
    img = _transparent_plant()
    params = SwayParams(frames=16, amplitude=0.1, power=1.5, lag=0.6, mesh_density=8)
    frames = render_sway_frames(img, params)
    arrs = [np.asarray(f).astype(float) for f in frames]
    adjacent = np.mean([np.abs(arrs[i] - arrs[i - 1]).mean() for i in range(1, len(arrs))])
    seam = np.abs(arrs[-1] - arrs[0]).mean()
    assert seam < adjacent * 2.0


def test_sway_no_black_edges_in_transparent_area():
    """透明像素的 RGB 必须为 0，不能出黑边。"""
    img, _ = _grass()
    frames = render_sway_frames(img, SwayParams(frames=4, amplitude=0.15, mesh_density=8))
    for frame in frames:
        arr = np.asarray(frame)
        transparent = arr[..., 3] == 0
        assert transparent.any()
        assert np.all(arr[..., :3][transparent] == 0), "透明区域 RGB 不为 0，有黑边"


@pytest.mark.parametrize("margin_bottom", [0, 25])
def test_sway_root_row_stays_fixed(margin_bottom):
    """植物自己的最低一行在帧间不能动；图底有留白时也一样。"""
    img, _ = _grass(margin_bottom=margin_bottom)
    frames = render_sway_frames(img, SwayParams(frames=8, amplitude=0.1, mesh_density=12))
    _, bottom = _alpha_rows(img)
    margin = _margin(img, frames[0])
    source = np.asarray(img)[bottom]
    opaque = source[..., 3] > 0
    assert opaque.any(), "根部行没有内容，测试没有意义"
    for frame in frames:
        row = np.asarray(frame)[bottom][margin:margin + img.width]
        assert np.array_equal(row[opaque], source[opaque]), "根部像素在帧间移动了"
        # 草叶边缘的双线性插值允许在根部行外侧留下几乎看不见的残影，但不能是实打实的像素
        assert row[~opaque][..., 3].max() <= 8


def _offsets(img, params, t):
    mesh = build_mesh(img, params.mesh_density)
    return mesh, sway_offsets(mesh, params, params.amplitude * img.width, img.width, t)


def test_sway_offsets_loop_exactly_after_one_period():
    """颤动在周期里是整数圈，第 N 帧必须和第 0 帧完全相同，循环才无缝。"""
    img, _ = _grass()
    params = SwayParams(frames=24, wave=2.0, flutter=0.5, mesh_density=12)
    _, first = _offsets(img, params, 0)
    _, wrapped = _offsets(img, params, params.frames)
    assert np.allclose(first, wrapped, atol=1e-4)


def test_sway_offsets_never_exceed_canvas_margin():
    """加了行波和颤动后位移仍不能超过 amplitude * 宽度，否则画布余量不够又会切边。"""
    img, _ = _grass()
    params = SwayParams(frames=32, amplitude=0.1, wave=3.0, flutter=1.0, mesh_density=12)
    limit = params.amplitude * img.width
    for t in range(params.frames):
        _, dx = _offsets(img, params, t)
        assert np.abs(dx).max() <= limit + 1e-3


def test_sway_wave_makes_left_and_right_move_out_of_phase():
    img = Image.new("RGBA", (200, 100), (40, 170, 60, 255))
    mesh = build_mesh(img, density=8)
    top = mesh.height > 0.9
    left = top & (mesh.vertices[:, 0] < 20)
    right = top & (mesh.vertices[:, 0] > 180)
    assert left.any() and right.any()

    def series(params):
        values = [sway_offsets(mesh, params, 16.0, img.width, t) for t in range(params.frames)]
        return (np.array([v[left].mean() for v in values]),
                np.array([v[right].mean() for v in values]))

    sync_left, sync_right = series(SwayParams(frames=24, wave=0, flutter=0))
    assert np.allclose(sync_left, sync_right, atol=1e-3)  # 没有行波：整丛同步
    wave_left, wave_right = series(SwayParams(frames=24, wave=2.0, flutter=0))
    assert np.abs(wave_left - wave_right).max() > 4  # 有行波：左右错开


def test_sway_without_wave_and_flutter_is_the_plain_swing():
    """wave 和 flutter 都为 0 时回到最初的单一摆动：高度^power * sin(角度 + lag * 高度)。"""
    img, _ = _grass()
    params = SwayParams(frames=12, amplitude=0.05, power=1.5, lag=0.6, wave=0, flutter=0)
    mesh, dx = _offsets(img, params, 3)
    angle = 2 * np.pi * 3 / 12
    expected = 0.05 * img.width * mesh.height ** 1.5 * np.sin(angle + mesh.height * 0.6)
    assert np.allclose(dx, expected, atol=1e-4)


def test_sway_rejects_invalid_params():
    with pytest.raises(ValueError, match="wave"):
        SwayParams(wave=-1)
    with pytest.raises(ValueError, match="flutter"):
        SwayParams(flutter=1.5)
    with pytest.raises(ValueError, match="frames"):
        SwayParams(frames=1)
    with pytest.raises(ValueError, match="amplitude"):
        SwayParams(amplitude=0)
    with pytest.raises(ValueError, match="power"):
        SwayParams(power=0)
    with pytest.raises(ValueError, match="lag"):
        SwayParams(lag=-1)
    with pytest.raises(ValueError, match="mesh_density"):
        SwayParams(mesh_density=2)


def test_sway_rejects_fully_transparent_image():
    with pytest.raises(ValueError, match="完全透明"):
        render_sway_frames(Image.new("RGBA", (32, 32)), SwayParams())


def test_sway_renders_real_size_image_in_reasonable_time():
    """旧实现每个三角形都处理整图，600x800 一帧要 13 秒；现在只处理三角形的包围盒。"""
    img, _ = _grass(width=600, height=800, blades=9)
    start = time.perf_counter()
    render_sway_frames(img, SwayParams(frames=4, mesh_density=12))
    assert time.perf_counter() - start < 8


def test_sway_runner_outputs_numbered_frames(tmp_path):
    source = tmp_path / "input.png"
    _transparent_plant().save(source)
    params = {"frames": 6, "amplitude": 0.1, "power": 1.5, "lag": 0.6, "mesh_density": 8}
    ctx = _ctx(tmp_path, params, [source])
    outputs = FramesSwayRunner().run(ctx)
    assert [p.name for p in outputs] == [f"{i:04d}.png" for i in range(6)]
    assert all(p.parent == ctx.out_dir / "sway" for p in outputs)
    assert ctx.meta["frames"] == 6


def test_sway_runner_rejects_bad_params(tmp_path):
    source = tmp_path / "input.png"
    _transparent_plant().save(source)
    with pytest.raises(ValueError, match="frames"):
        FramesSwayRunner().run(_ctx(tmp_path, {"frames": 1}, [source]))
    with pytest.raises(ValueError, match="amplitude"):
        FramesSwayRunner().run(_ctx(tmp_path, {"amplitude": 0}, [source]))


def test_sway_runner_stops_when_cancelled(tmp_path):
    source = tmp_path / "input.png"
    _transparent_plant().save(source)
    ctx = _ctx(tmp_path, {"frames": 4}, [source])
    ctx.cancel_event.set()
    with pytest.raises(RuntimeError, match="已取消"):
        FramesSwayRunner().run(ctx)


def test_sway_runner_clears_stale_frames_on_rerun(tmp_path):
    source = tmp_path / "input.png"
    _transparent_plant().save(source)
    stale = tmp_path / "pack" / "plant" / "sway" / "9999.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    FramesSwayRunner().run(_ctx(tmp_path, {"frames": 4}, [source]))
    assert not stale.exists()


def test_scene_foliage_template_registered():
    template = load_template("scene_foliage")
    assert template.type == "scene"
    assert template.step("sway").type == "frames.sway"
    runners = registered_runners()
    assert all(step.type in runners for step in template.steps)


def test_scene_foliage_sway_step_params():
    sway = load_template("scene_foliage").step("sway")
    assert sway.params["frames"] == 24
    assert sway.params["amplitude"] == 0.08
    assert sway.params["power"] == 1.5
    assert sway.params["lag"] == 0.6
    assert sway.params["mesh_density"] == 12
    assert sway.params["wave"] == 1.5
    assert sway.params["flutter"] == 0.3
