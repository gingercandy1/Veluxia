"""一段的图层叠放顺序：原层 L1…L4 从底到顶，合并层 M1… 按被合并层里最上面那层的位置插入。
前后端都按这个规则排，预览叠出来的才和后端合并出来的一致。"""


def stack_key(layer_id: str, z: int) -> tuple[int, int, int]:
    """排序键：先按位置，同一位置合并层压在原层上面，再按编号。"""
    kind, number = layer_id[:1], layer_id[1:]
    if kind not in ("L", "M") or not number.isdigit():
        raise ValueError(f"不认识的图层编号：{layer_id}")
    return z, 1 if kind == "M" else 0, int(number)
