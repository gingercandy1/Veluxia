"""宽幅分窗生成（MultiDiffusion）：一次生成几屏宽的长背景，相邻段在生成过程中自动衔接。

分开生成几张再拼，接缝两边的地平线、光照、透视都对不上，事后融合只会糊出一条过渡带。
这里改为开一张超宽 latent，每步去噪把它切成互相重叠、模型原生尺寸的窗口，重叠处按权重平均：
各段的噪声不同所以内容各异，而每一步都在重叠区对齐，所以衔接是生成出来的而不是拼出来的。

Z-Image 是流匹配模型，调度器一步是 x + dt·v，对 v 线性，平均模型输出等价于平均去噪结果。
"""
import math
from contextlib import contextmanager

import torch

MAX_SEGMENTS = 4
# 相邻窗口的重叠比例：越大接缝越自然，但窗口数和耗时也越多
PANORAMA_OVERLAP = 0.25


def validate_segments(segments) -> int:
    # 模板占位符替换后是字符串，这里统一转成整数
    value = int(segments)
    if not 1 <= value <= MAX_SEGMENTS:
        raise ValueError(f"背景段数必须在 1 到 {MAX_SEGMENTS} 之间：{segments}")
    return value


def window_starts(total: int, window: int, overlap: float, wrap: bool, align: int = 2,
                  multiple: int = 1) -> list[int]:
    """各窗口的起点（latent 列）。起点对齐到 align，保持和 2×2 patch 对齐。

    wrap 时画布首尾相接，窗口均匀绕一圈，有窗口越过右端接回左端，整条长卷也能无缝平铺；
    窗口中心落在各自等分区间的正中，这样分段时每个窗口都整段落在某一段里。
    否则第一个贴左端、最后一个贴右端，中间均匀分布。
    multiple：窗口数向上取到它的倍数。分段描述时传段数，否则 3 段只有 4 个窗口，
    某一段会多分到一个窗口、在画面里占得更宽。
    """
    stride = max(int(window * (1 - overlap)), align)
    if wrap:
        count = math.ceil(math.ceil(total / stride) / multiple) * multiple
        return [round(((i + 0.5) * total / count - window / 2) / align) * align % total
                for i in range(count)]
    if window >= total:
        return [0]
    count = math.ceil((math.ceil((total - window) / stride) + 1) / multiple) * multiple
    return [round(i * (total - window) / (count - 1) / align) * align for i in range(count)]


def window_weight(window: int, overlap: float) -> torch.Tensor:
    """窗口内的权重：两侧重叠带线性渐升，中间为 1。均匀平均会在窗口边界留下一道硬边，渐变过渡才看不出。"""
    ramp = max(round(window * overlap), 1)
    position = torch.arange(window, dtype=torch.float32)
    distance = torch.minimum(position, window - 1 - position)
    return torch.clamp((distance + 1) / ramp, max=1.0)


def parse_segment_prompts(value) -> list[str]:
    """分段描述：模板表格里是单行文本，用 | 分隔（全角｜也认）；接口调用也可以直接传列表。"""
    if isinstance(value, str):
        value = value.replace("｜", "|").split("|")
    return [part.strip() for part in value if part.strip()]


def segment_of(start: int, window: int, total: int, segments: int) -> int:
    """窗口归哪一段：看窗口中心落在哪一屏。跨段的窗口各用各的提示词，重叠区加权平均就是过渡带。"""
    center = (start + window // 2) % total
    return center * segments // total


@contextmanager
def panorama_transformer(transformer: torch.nn.Module, window: int,
                         overlap: float = PANORAMA_OVERLAP, wrap: bool = False,
                         segment_feats: list | None = None):
    """把超宽 latent 沿宽度切成窗口，逐个窗口前向，再按权重合回原宽度。

    window 是窗口宽度（latent 列）。输入是 Z-Image 的格式：latent 列表 (C, F, H, W)，
    第三个位置参数是提示词嵌入列表。segment_feats 给出时，每段一份嵌入列表（形状同该参数），
    窗口改用它所属那段的嵌入，实现"森林 → 村庄 → 城堡"这样逐段变化的长卷。
    必须逐窗前向而不是拼成一批：一批 4 个窗口的激活会撑爆 8GB，溢出到共享内存后慢到不可用。
    钩子做不到"一次调用拆成多次前向"，所以临时包一层实例 forward；包的是 cpu offload
    已经包过的那个 forward，逐窗调用它仍会走 offload，退出时原样还原。
    circular_transformer 的钩子在 forward 外层，两者可叠用：先循环补高、再切窗。
    """
    weight = window_weight(window, overlap)
    had_own_forward = "forward" in transformer.__dict__
    original = transformer.forward

    def forward(latents, timestep, cap_feats, *args, **kwargs):
        width = latents[0].shape[-1]
        total = [latent.new_zeros(latent.shape, dtype=torch.float32) for latent in latents]
        weight_sum = torch.zeros(width, device=latents[0].device)
        part_weight = weight.to(latents[0].device)
        multiple = len(segment_feats) if segment_feats else 1
        for start in window_starts(width, window, overlap, wrap, multiple=multiple):
            index = ((torch.arange(window) + start) % width).to(latents[0].device)
            feats = (segment_feats[segment_of(start, window, width, len(segment_feats))]
                     if segment_feats else cap_feats)
            parts = original([latent.index_select(-1, index) for latent in latents],
                             timestep, feats, *args, **kwargs)[0]
            for acc, part in zip(total, parts, strict=True):
                acc.index_add_(acc.dim() - 1, index, part.float() * part_weight)
            weight_sum.index_add_(0, index, part_weight)
        return ([(acc / weight_sum).to(latent.dtype) for acc, latent in zip(total, latents, strict=True)],)

    transformer.forward = forward
    try:
        yield
    finally:
        if had_own_forward:
            transformer.forward = original
        else:
            del transformer.forward
