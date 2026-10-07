"""程序化网格变形：把透明底静态图变成可无缝循环的摇摆序列帧（ADR 0008）。

纯 CPU，不加载模型、不占显存租约。核心思路：
- 用 alpha 轮廓 + 不透明区域的网格点自动取网格（approxPolyDP + Delaunay 三角剖分）；
- 根部（不透明区域的最低一行）不动，越往上偏移越大，叶尖比叶根慢半拍（相位随高度滞后）；
- 整周期 = N 帧，首尾天然衔接，不需要叠化。
"""
import math
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# 颤动在一个摆动周期里的圈数：必须是整数，否则首尾帧对不上，循环会跳
FLUTTER_CYCLES = 3


@dataclass
class Mesh:
    """自动生成的变形网格：顶点坐标、三角形索引、每顶点归一化高度。"""
    vertices: np.ndarray  # (V, 2) float32，像素坐标
    triangles: np.ndarray  # (T, 3) int32，顶点索引
    # (V,) float32：0=不透明区域最低一行（根部，不动），1=最高处（叶尖）。
    # 曲线（power）留到渲染时再取，这样同一张网格可以用不同的软硬程度重复渲染
    height: np.ndarray


@dataclass
class SwayParams:
    """摇摆参数。frames=周期帧数，amplitude=最大位移（占图宽比例），
    power=高度权重曲线，lag=叶尖相位滞后（弧度），mesh_density=网格密度，
    wave=从左到右的相位差（弧度，风像一道波扫过草丛，0 则整丛同步摆），
    flutter=叶尖高频颤动占主摆动的比例（0~1，0 则只有主摆动）。"""
    frames: int = 24
    amplitude: float = 0.08
    power: float = 1.5
    lag: float = 0.6
    mesh_density: int = 12
    wave: float = 1.5
    flutter: float = 0.3

    def __post_init__(self) -> None:
        if self.wave < 0:
            raise ValueError(f"wave 必须 >= 0：{self.wave}")
        if not 0 <= self.flutter <= 1:
            raise ValueError(f"flutter 必须在 0 到 1 之间：{self.flutter}")
        if self.frames < 2:
            raise ValueError(f"frames 必须 >= 2：{self.frames}")
        if self.amplitude <= 0:
            raise ValueError(f"amplitude 必须 > 0：{self.amplitude}")
        if self.power <= 0:
            raise ValueError(f"power 必须 > 0：{self.power}")
        if self.lag < 0:
            raise ValueError(f"lag 必须 >= 0：{self.lag}")
        if self.mesh_density < 4:
            raise ValueError(f"mesh_density 必须 >= 4：{self.mesh_density}")


def build_mesh(image: Image.Image, density: int = 12) -> Mesh:
    """从 alpha 通道自动生成 Delaunay 三角网格。

    草丛这类由多块分离元素组成的图，每一块的轮廓都要进网格，所以取全部外轮廓；
    内部补点则直接按 alpha 判断是否落在不透明区域，不依赖某一个轮廓。
    顶点高度按不透明区域的上下边界归一化，图底有留白时根部也不会跟着晃。
    """
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        raise ValueError("图片完全透明，无法生成网格")
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())

    points: set[tuple[int, int]] = set()
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        for x, y in cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2):
            points.add((int(x), int(y)))

    # 补点网格包含最底一行和最右一列，保证根部一定有顶点落在 h=0 上
    step = max((right - left + 1) // density, 4)
    grid_y = np.unique(np.append(np.arange(top, bottom + 1, step), bottom))
    grid_x = np.unique(np.append(np.arange(left, right + 1, step), right))
    for y in grid_y:
        for x in grid_x:
            if alpha[y, x] > 0:
                points.add((int(x), int(y)))

    height, width = alpha.shape
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for point in points:
        subdiv.insert((float(point[0]), float(point[1])))
    triangle_list = subdiv.getTriangleList()
    if len(triangle_list) == 0:
        raise ValueError("三角剖分失败，网格点太少")

    vertex_index: dict[tuple[int, int], int] = {}
    vertices: list[tuple[int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    for tri in triangle_list:
        corners = [(round(float(tri[i])), round(float(tri[i + 1]))) for i in range(0, 6, 2)]
        if any(c not in points for c in corners):
            continue  # Subdiv2D 的外框虚拟顶点
        for corner in corners:
            if corner not in vertex_index:
                vertex_index[corner] = len(vertices)
                vertices.append(corner)
        triangles.append(tuple(vertex_index[c] for c in corners))
    if not triangles:
        raise ValueError("三角剖分失败，网格点太少")

    verts = np.array(vertices, dtype=np.float32)
    span = max(bottom - top, 1)
    heights = np.clip((bottom - verts[:, 1]) / span, 0, 1).astype(np.float32)
    return Mesh(verts, np.array(triangles, dtype=np.int32), heights)


def iter_sway_frames(image: Image.Image, params: SwayParams) -> Iterator[Image.Image]:
    """逐帧渲染摇摆序列。整周期 = N 帧，首尾天然衔接。

    做成迭代器是为了让调用方在两帧之间检查取消。
    输出画布左右各比原图宽一个最大位移：草往一侧摆时贴着原图边缘的部分会越过原来的边界，
    画布不加宽就被切掉。
    """
    image = image.convert("RGBA")
    max_dx = params.amplitude * image.width
    margin = math.ceil(max_dx) + 2
    padded = Image.new("RGBA", (image.width + 2 * margin, image.height))
    padded.paste(image, (margin, 0))
    mesh = build_mesh(padded, params.mesh_density)
    return _render(np.asarray(padded, dtype=np.float32), mesh, params, max_dx)


def render_sway_frames(image: Image.Image, params: SwayParams) -> list[Image.Image]:
    return list(iter_sway_frames(image, params))


def _render(rgba: np.ndarray, mesh: Mesh, params: SwayParams,
            max_dx: float) -> Iterator[Image.Image]:
    img_h, img_w = rgba.shape[:2]
    # 预乘 alpha，避免插值时透明像素的颜色渗进边缘出黑边
    premult = rgba.copy()
    premult[..., :3] *= rgba[..., 3:4] / 255.0

    covered = _covered_triangles(rgba[..., 3] > 0, mesh)

    for t in range(params.frames):
        dx = sway_offsets(mesh, params, max_dx, img_w, t)
        canvas = np.zeros_like(premult)
        for tri in covered:
            _paste_triangle(canvas, premult, mesh.vertices[tri], dx[tri], img_w, img_h)
        yield Image.fromarray(_unpremultiply(canvas), "RGBA")


def sway_offsets(mesh: Mesh, params: SwayParams, max_dx: float, img_w: int, t: int) -> np.ndarray:
    """第 t 帧每个顶点的水平位移，绝对值不超过 max_dx。

    单靠"高度决定位移"整丛草会同相位地像一块布在晃，所以在主摆动上叠两层变化：
    - 行波：相位随横坐标推移，风像一道波从一侧扫到另一侧，草叶依次而不是同时摆；
    - 颤动：叶尖上频率更高的小幅摆动，让末梢比根部更灵动。
    颤动在一个周期里整数圈，t = frames 时和 t = 0 完全相同，循环依然无缝。
    """
    angle = 2 * math.pi * t / params.frames
    across = mesh.vertices[:, 0] / max(img_w - 1, 1)
    phase = mesh.height * params.lag + across * params.wave
    swing = np.sin(angle + phase) + params.flutter * np.sin(FLUTTER_CYCLES * (angle + phase))
    return max_dx * np.power(mesh.height, params.power) * swing / (1 + params.flutter)


def _covered_triangles(opaque: np.ndarray, mesh: Mesh) -> list[np.ndarray]:
    """去掉原图里完全没有内容的三角形（如草叶之间的空隙），渲染时可省掉。"""
    kept = []
    img_h, img_w = opaque.shape
    for tri in mesh.triangles:
        pts = mesh.vertices[tri]
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
        x1, y1 = min(x1, img_w), min(y1, img_h)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(pts - (x0, y0)).astype(np.int32), 255)
        if (opaque[y0:y1, x0:x1] & (mask > 0)).any():
            kept.append(tri)
    return kept


def _paste_triangle(canvas: np.ndarray, premult: np.ndarray, src: np.ndarray,
                    dx: np.ndarray, img_w: int, img_h: int) -> None:
    """把源图中的一个三角形仿射变换到位移后的位置，只处理它的包围盒。

    重叠处按"alpha 大者胜"合成：相邻三角形共享的边界像素不会被重复叠加，
    也不会被另一个三角形里的透明部分盖掉。
    """
    dst = src.copy()
    dst[:, 0] += dx
    x0, y0 = np.floor(dst.min(axis=0)).astype(int)
    x1, y1 = np.ceil(dst.max(axis=0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img_w), min(y1, img_h)
    if x1 <= x0 or y1 <= y0:
        return
    local = dst - (x0, y0)
    matrix = cv2.getAffineTransform(src.astype(np.float32), local.astype(np.float32))
    patch = cv2.warpAffine(premult, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
    # 亚像素精度的三角形掩码，边界像素两侧都会覆盖到，不会留缝
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(local * 16).astype(np.int32), 255,
                       lineType=cv2.LINE_8, shift=4)
    target = canvas[y0:y1, x0:x1]
    win = (mask > 0) & (patch[..., 3] >= target[..., 3])
    target[win] = patch[win]


def _unpremultiply(canvas: np.ndarray) -> np.ndarray:
    alpha = canvas[..., 3:4]
    out = np.zeros_like(canvas)
    np.divide(canvas[..., :3] * 255.0, alpha, out=out[..., :3], where=alpha > 0)
    out[..., 3:4] = alpha
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    # 插值留下的极小 alpha 取整后为 0，但 RGB 可能非零，清掉以免透明区域出杂色
    out[out[..., 3] == 0, :3] = 0
    return out
