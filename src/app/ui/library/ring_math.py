"""环形切点的纯计算：不依赖 Qt，控件和页面共用同一份，免得各算各的。"""


def segment_widths(cuts: list[int], total: int) -> list[int]:
    """第 i 段从第 i 个切点到下一个，最后一段绕回第一个；只有一个切点时整条环是一段。"""
    if len(cuts) == 1:
        return [total]
    return [(cuts[(i + 1) % len(cuts)] - cuts[i]) % total for i in range(len(cuts))]
