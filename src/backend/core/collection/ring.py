"""资源包级循环接缝规划、候选生成、采用和成品回贴。"""
import functools
import random
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.backend.core.collection import library
from src.backend.core.collection.executor import file_signature
from src.backend.core.collection.generator_runner import call_blocking, move_into
from src.backend.core.collection.manifest import load_manifest, save_manifest
from src.backend.core.collection.ring_cuts import (
    column_cost,
    export_segments,
    long_image,
    plan_cuts,
    segment_widths,
    validate_cuts,
)
from src.backend.core.collection.ring_extend import (
    EXTEND_PAD,
    extend_canvas,
    fit_height,
    model_dimensions,
    restore_middle,
)
from src.backend.core.collection.seamless_loop import (
    EDGE_COLUMNS,
    apply_patch,
    build_window,
    edge_cost_matrix,
    edge_strips,
    extract_patch,
    narrow_merge,
    order_loop,
)
from src.backend.core.collection.template import load_template
from src.backend.core.image.depth import estimate_ring_depth
from src.backend.core.model_base import GeneratorFactory
from src.shared.enum_type import FactoryType
from src.shared.layer_stack import stack_key
from src.shared.schemas import (
    CollectionItem,
    ImageVersion,
    Manifest,
    MergedLayer,
    RingSeamStatus,
    RingState,
    SeamCandidate,
    SeamState,
    SegmentLayers,
    VersionHistory,
)

MAX_CANDIDATES = 4
HARMONIZE_STRENGTH = 0.3
GENERATE_STEP = "generate"
SEGMENT_DIR = Path("ring") / "segments"
SLOT_DIR = Path("ring") / "slots"
# 每段的分层结果放在各自目录：按段身份命名，切点变了旧目录不会被新段误用
LAYER_DIR = Path("ring") / "layers"
LAYER_STEP = "layer"
LAYER_MODEL = "Qwen-Image-Layered"
# 和用户定的一致：每段固定 4 层，之后在"检查与合并"里按需合并
LAYER_COUNT = 4
# 外扩和文字改图用的图生图模型（8GB 用量化版）
EDIT_MODEL = "Qwen-Image-2.1-Turbo-Q4"
# 给编辑模型的画面面积 = 边长²。1024 下编辑会忽略指令（diffusers #14824），和 qwen_image_21 的默认一致
EDIT_SIDE = 768
# 外扩 / 改图的历史版本：每个目标一个目录，按序号存
VERSION_DIR = Path("ring") / "versions"
# 外扩时给模型的指令：只许续画上下两条新补的区域，不许动中间
EXTEND_PROMPT = (
    "Extend the image upward and downward. Continue the existing scene seamlessly into the "
    "blank stretched areas at the top and bottom, matching the art style, lighting and "
    "perspective. Keep the center of the image unchanged. No characters, no text."
)


def plan_ring(pack_id: str, item_ids: list[str] | None = None) -> Manifest:
    """检查场景图产物并按边缘相似度规划闭环，保留仍有效的旧接缝状态。

    item_ids 指定参与循环的条目；不给时沿用上次规划的成员，从没规划过就用全部条目。
    """
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        if manifest.type != "scene":
            raise ValueError("循环拼接只支持场景资源包")
        members = _ring_members(manifest, item_ids)
        if len(members) < 2:
            raise ValueError("循环拼接至少需要两个条目")

        sources = [_screen_source(directory, manifest.ring, item) for item in members]
        # 部分外扩后各屏高度不同，边缘代价要在同一高度上比较
        height = _max_height(sources)
        strips = [_fitted_edge_strips(path, height) for path in sources]
        order_ids = [members[index].id
                     for index in order_loop(edge_cost_matrix(strips))]
        _commit_order(directory, manifest, order_ids)
        return manifest


def set_order(pack_id: str, order: list[str]) -> Manifest:
    """用户手动排的顺序：同一屏可以出现多次，位置就是它在环上的位置。

    和 plan_ring 的区别是不重新排序；接缝按相邻关系保留，所以挪动一张图只影响它前后的接缝。
    """
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        if manifest.type != "scene":
            raise ValueError("循环拼接只支持场景资源包")
        if len(order) < 2:
            raise ValueError("循环拼接至少需要两个条目")
        items = {item.id: item for item in manifest.items}
        unknown = sorted(set(order) - set(items))
        if unknown:
            raise ValueError(f"资源包里没有这些条目：{', '.join(unknown)}")
        for item_id in dict.fromkeys(order):
            _screen_source(directory, manifest.ring, items[item_id])
        _commit_order(directory, manifest, list(order))
        return manifest


def _commit_order(directory: Path, manifest: Manifest, order_ids: list[str]) -> None:
    old_seams = manifest.ring.seams if manifest.ring else {}
    ordered_keys = [
        _seam_key(order_ids[index], order_ids[(index + 1) % len(order_ids)])
        for index in range(len(order_ids))
    ]
    seams = {key: old_seams[key] for key in ordered_keys if key in old_seams}
    for key in ordered_keys:
        seams.setdefault(key, SeamState())
    previous = manifest.ring
    new_ring = RingState(order=order_ids, seams=seams)
    if previous is not None:
        # 每屏自己的外扩 / 改图历史跟着条目走，和排在哪无关
        new_ring.versions = {key: history for key, history in previous.versions.items()
                             if key.startswith("screen:")}
    if previous is not None and previous.order == order_ids:
        # 顺序没变，长图坐标也没变：切点和已导出的段继续有效（图变了由签名判断）
        new_ring.cuts, new_ring.cuts_sig = previous.cuts, previous.cuts_sig
        new_ring.segments = previous.segments
        new_ring.suggested_cuts = previous.suggested_cuts
        new_ring.versions.update({key: history for key, history in previous.versions.items()
                                  if key.startswith("segment:")})
    if previous is not None:
        # 没带过来的段 / 层历史对应的图不会再被引用，文件一并清掉
        for key, history in previous.versions.items():
            if key not in new_ring.versions:
                _drop_history(directory, key, history)
    manifest.ring = new_ring
    save_manifest(directory, manifest)


def generate_seams(pack_id: str, keys: list[str] | None = None,
                   prompt_override: str | None = None,
                   cancel_event: threading.Event | None = None,
                   extra_prompt: str | None = None,
                   on_progress: Callable[[str, float], None] | None = None) -> Manifest:
    """按需生成接缝候选；一次持有模型租约，逐缝保存进度以便失败后续跑。
    on_progress(接缝 key, 0~1)：当前这条接缝的推理进度，界面据此在对应卡片上画进度条。"""
    directory = library.pack_dir(pack_id)
    cancel_event = cancel_event or threading.Event()
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        template = load_template(manifest.template)
        model_name = template.step(GENERATE_STEP).params.get("model_name")
        if not model_name:
            raise ValueError(f"模板 {manifest.template} 的 generate 步骤缺少 model_name")

        # 接缝基于每屏“当前图”（外扩 / 改图后的版本）：版本一换签名就变，旧接缝自动变成 stale
        source_paths = {
            item.id: _screen_source(directory, ring, item)
            for item in manifest.items
        }
        signatures = {item_id: file_signature(path) for item_id, path in source_paths.items()}
        height = _ring_height(directory, manifest, ring)
        statuses = _statuses(ring, signatures)
        if keys:
            unknown = sorted(set(keys) - set(ring.seams))
            if unknown:
                raise ValueError(f"循环拼接中没有这些接缝：{', '.join(unknown)}")
            selected = list(dict.fromkeys(keys))
        else:
            selected = [key for key, status in statuses.items()
                        if not status.adopted or status.stale]
        if not selected:
            return manifest

        try:
            with GeneratorFactory.acquire(FactoryType.Image, model_name) as generator:
                generator.cancel_event = cancel_event
                try:
                    generator.ensure_model_loaded()
                    for key in selected:
                        if cancel_event.is_set():
                            generator.check_cancelled()
                        left_id, right_id = _split_key(key)
                        left_item = _find_item(manifest, left_id)
                        right_item = _find_item(manifest, right_id)
                        prompt = _build_prompt(
                            left_item.prompt,
                            right_item.prompt,
                            manifest.style.prompt,
                            left_item.fields.get("view", ""),
                            (extra_prompt or "").strip(),
                        )
                        if prompt_override and prompt_override.strip():
                            prompt = prompt_override.strip()
                        seed = random.randint(0, 2**31 - 1)
                        if on_progress is not None:
                            on_progress(key, 0.0)
                            generator.progress_callback = functools.partial(on_progress, key)
                        with Image.open(source_paths[left_id]) as left_image, \
                                Image.open(source_paths[right_id]) as right_image:
                            # 窗口两侧高度必须一致：较矮的屏先补齐到环高度
                            window = build_window(fit_height(left_image, height),
                                                  fit_height(right_image, height))
                        inpainted, harmonized = call_blocking(generator.repaint_seam(
                            window, prompt, seed, HARMONIZE_STRENGTH))
                        result = (narrow_merge(inpainted, harmonized)
                                  if harmonized is not None else inpainted)
                        patch = extract_patch(result)

                        seam = ring.seams.setdefault(key, SeamState())
                        candidate_id = _next_candidate_id(seam)
                        relative_path = Path("ring") / f"{left_id}_{right_id}" / \
                            f"{candidate_id}.png"
                        patch_path = _safe_pack_path(directory, relative_path)
                        patch_path.parent.mkdir(parents=True, exist_ok=True)
                        patch.save(patch_path, format="PNG")
                        candidate = SeamCandidate(
                            id=candidate_id,
                            file=relative_path.as_posix(),
                            seed=seed,
                            prompt=prompt,
                            left_sig=signatures[left_id],
                            right_sig=signatures[right_id],
                            harmonize=HARMONIZE_STRENGTH if harmonized is not None else 0.0,
                        )
                        seam.candidates.append(candidate)
                        if not seam.adopted:
                            seam.adopted = candidate_id
                        _prune_candidates(directory, seam)
                        # 做完一条就合成：界面刷新时预览里已经带着这条接缝，不必等全部做完
                        _render_seam_slots(directory, manifest, ring, key)
                        save_manifest(directory, manifest)
                finally:
                    generator.cancel_event = None
                    generator.progress_callback = None
        except Exception:
            # 每个成功接缝已单独落盘；此处再写一次，确保失败不会丢失已经推进的状态。
            save_manifest(directory, manifest)
            raise
        return manifest


def adopt_seam(pack_id: str, key: str, candidate_id: str) -> Manifest:
    """把指定接缝的一个现有候选设为采用版本。"""
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        seam = ring.seams.get(key)
        if seam is None or not any(candidate.id == candidate_id for candidate in seam.candidates):
            raise ValueError(f"接缝 {key} 没有候选 {candidate_id}")
        seam.adopted = candidate_id
        save_manifest(directory, manifest)
        return manifest


def render_ring(pack_id: str) -> Manifest:
    """纯 CPU 将采用的补丁分别贴回每屏，原始 generate 图始终保留。"""
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        if len(ring.order) < 2:
            raise ValueError("循环拼接至少需要两个条目，请先规划循环顺序")
        for index in range(len(ring.order)):
            _render_slot(directory, manifest, ring, index)
        return manifest


def _render_slot(directory: Path, manifest: Manifest, ring: RingState, index: int) -> None:
    """合成环上第 index 个位置：原图贴上左右两条接缝已采用的补丁。"""
    item_id = ring.order[index]
    item = next((candidate for candidate in manifest.items if candidate.id == item_id), None)
    if item is None:
        raise ValueError(f"循环顺序引用了不存在的条目：{item_id}")
    ring_height = _ring_height(directory, manifest, ring)
    with Image.open(_screen_source(directory, ring, item)) as source_image:
        # 所有位置合成成同一高度，长图才拼得起来
        rendered = fit_height(source_image, ring_height)
    previous_id = ring.order[index - 1]
    next_id = ring.order[(index + 1) % len(ring.order)]
    previous = _adopted_candidate(directory, ring, _seam_key(previous_id, item_id))
    following = _adopted_candidate(directory, ring, _seam_key(item_id, next_id))
    # 补丁高度和当前画面高度不一致说明外扩后环高度变了，旧补丁对不上：跳过它。
    # 这条接缝的签名已经变了，会显示为 stale，等用户重新生成
    if previous:
        with Image.open(_safe_pack_path(directory, previous.file)) as patch:
            if patch.height == rendered.height:
                rendered = apply_patch(rendered, patch, side="left")
    if following:
        with Image.open(_safe_pack_path(directory, following.file)) as patch:
            if patch.height == rendered.height:
                rendered = apply_patch(rendered, patch, side="right")
    target = _safe_pack_path(directory, _slot_file(item_id, previous_id, next_id))
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(target, format="PNG")


def _render_seam_slots(directory: Path, manifest: Manifest, ring: RingState, key: str) -> None:
    """只重合成受这条接缝影响的位置（它左右两屏）。同一对相邻屏在环里出现多次时都要做。"""
    left_id, right_id = _split_key(key)
    count = len(ring.order)
    if count < 2:
        return
    for index, item_id in enumerate(ring.order):
        if (item_id == left_id and ring.order[(index + 1) % count] == right_id) \
                or (item_id == right_id and ring.order[index - 1] == left_id):
            _render_slot(directory, manifest, ring, index)


def auto_cuts(pack_id: str, cancel_event: threading.Event | None = None) -> Manifest:
    """估整条环的深度，自动选切段位置：切线落在物体之间的空隙，一棵树不会分进两段。"""
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        paths = _rendered_sources(directory, ring)
        image = _ring_long_image(paths)
        depth = estimate_ring_depth(image, cancel_event)
        # 只写建议：用户可能在对话框里取消，取消后已确认的切点、段图和分层都不能变
        ring.suggested_cuts = plan_cuts(column_cost(depth))
        save_manifest(directory, manifest)
        return manifest


def confirm_cuts(pack_id: str, cuts: list[int]) -> Manifest:
    """保存用户确认（可能拖动过）的切点，并按切点导出每段图供分层。纯 CPU。"""
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        paths = _rendered_sources(directory, ring)
        image = _ring_long_image(paths)
        ring.cuts = validate_cuts(cuts, image.width)
        ring.cuts_sig = _cuts_signature(paths)
        segment_dir = _safe_pack_path(directory, SEGMENT_DIR)
        exported = export_segments(image, ring.cuts, segment_dir)
        ring.segments = [(SEGMENT_DIR / path.name).as_posix() for path in exported]
        # 确认切段就清空全部分层：切点一变，段内容和用户对分层的预期都不再可靠，
        # 与其保留半截结果让人猜哪些有效，不如统一重来（分层是显式的单独操作）
        for key in list(ring.segment_layers):
            del ring.segment_layers[key]
            shutil.rmtree(_safe_pack_path(directory, _layer_dir(key)), ignore_errors=True)
        # 段 / 层的改图历史依附于旧的段和层，一并删除（含文件）
        for key in [key for key in ring.versions if key.startswith(("segment:", "layer:"))]:
            _drop_history(directory, key, ring.versions.pop(key))
        ring.suggested_cuts = []
        save_manifest(directory, manifest)
        return manifest


def layer_segments(pack_id: str, indices: list[int] | None = None,
                   cancel_event: threading.Event | None = None,
                     on_progress: Callable[[str, int, list[int]], None] | None = None) -> Manifest:
    """每段交给 Qwen-Layered 分成 LAYER_COUNT 层。
    on_progress(阶段, 当前段序号, 还没做完的段序号)：阶段是 encode / denoise，序号是在
    ring.segments 里的位置；界面据此给每段上色，不必自己推断哪些段已经做完。

    一次持有模型租约：先把要做的段集中编码，再集中去噪，编码器和 transformer 各只加载一次。
    每做完一段就存盘，中途停止或失败后再点，只做剩下的段。
    indices 给了就强制重做这些段（重新编码和去噪）。
    """
    directory = library.pack_dir(pack_id)
    cancel_event = cancel_event or threading.Event()
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        segments = _confirmed_segments(directory, manifest)
        if indices is not None:
            bad = sorted(set(indices) - set(range(len(segments))))
            if bad:
                raise ValueError(f"没有这些段：{', '.join(str(i + 1) for i in bad)}")
            todo = [segments[i] for i in dict.fromkeys(indices)]
            for key, *_ in todo:
                old = ring.segment_layers.pop(key, None)
                if old is not None:
                    _drop_merged(directory, old)
                _drop_layer_versions(directory, ring, key)
        else:
            todo = [segment for segment in segments
                    if not _segment_done(directory, ring, segment[0])]
        if not todo:
            return manifest

        position = {key: index for index, (key, *_) in enumerate(segments)}

        def report(stage: str, current: int):
            if on_progress is not None:
                on_progress(stage, position[todo[current][0]],
                            [position[key] for key, *_ in todo[current:]])

        model_name = _layer_model(manifest)
        try:
            with GeneratorFactory.acquire(FactoryType.Image, model_name) as generator:
                generator.cancel_event = cancel_event
                try:
                    generator.ensure_model_loaded()
                    for current, (key, path, prompt) in enumerate(todo):
                        state = ring.segment_layers.get(key)
                        # 只重做去噪时（上次停在去噪阶段）编码可以复用：描述没变就不用再加载编码器
                        if state is not None and state.prompt == prompt and state.embeds and \
                                _safe_pack_path(directory, state.embeds).is_file():
                            continue
                        if state is not None:
                            # 描述变了，图层要重做，旧的合并层跟着作废
                            _drop_merged(directory, state)
                        report("encode", current)
                        generator.parse_params({"content": prompt})
                        embeds = generator.save_prompt_embeds()
                        relative = _layer_dir(key) / "encode.pt"
                        move_into(Path(embeds), _safe_pack_path(directory, relative))
                        ring.segment_layers[key] = SegmentLayers(
                            source_sig=ring.cuts_sig, prompt=prompt,
                            embeds=relative.as_posix())
                        save_manifest(directory, manifest)
                    for current, (key, path, prompt) in enumerate(todo):
                        state = ring.segment_layers[key]
                        report("denoise", current)
                        generator.parse_params({
                            "content": prompt, "layers": LAYER_COUNT,
                            "reference_image": str(path),
                            "prompt_embeds_path": str(_safe_pack_path(directory, state.embeds)),
                        })
                        outputs = call_blocking(generator.generate())
                        if not outputs or not all(Path(p).is_file() for p in outputs):
                            raise RuntimeError(f"段 {key} 没有生成图层，请查看后端日志")
                        _drop_merged(directory, state)
                        # 层重做了：旧层上的改图历史对不上新图，base_sig 又相同，必须显式清掉
                        _drop_layer_versions(directory, ring, key)
                        state.layers = []
                        state.source_sig = ring.cuts_sig
                        for index, output in enumerate(outputs, start=1):
                            relative = _layer_dir(key) / f"layer_{index}.png"
                            move_into(Path(output), _safe_pack_path(directory, relative))
                            state.layers.append(relative.as_posix())
                        save_manifest(directory, manifest)
                finally:
                    generator.cancel_event = None
        except Exception:
            save_manifest(directory, manifest)
            raise
        return manifest


def merge_layers(pack_id: str, segment: int, layer_ids: list[str]) -> Manifest:
    """把一段里选中的几层按叠放顺序合成一张新图层，原层不动。纯 CPU，不碰模型租约。

    来源可以是原层 L1…L4，也可以是已有的合并层；新层的位置取来源里最上面那层的位置。
    """
    ids = list(dict.fromkeys(layer_ids))
    if len(ids) < 2:
        raise ValueError("至少选两层才能合并")
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        key, state = _layered_segment(directory, manifest, ring, segment)
        stack = _layer_stack(state)
        unknown = [layer_id for layer_id in ids if layer_id not in stack]
        if unknown:
            raise ValueError(f"这一段没有这些层：{', '.join(unknown)}")
        chosen = sorted(ids, key=lambda layer_id: stack_key(layer_id, stack[layer_id][1]))
        canvas = None
        for layer_id in chosen:
            # 层被文字改过图就合并改后的版本，和画布上看到的一致
            current = _current_version(directory, ring, _layer_target(key, layer_id),
                                       ring.cuts_sig)
            layer_path = current or _safe_pack_path(directory, stack[layer_id][0])
            with Image.open(layer_path) as opened:
                layer = opened.convert("RGBA")
            if canvas is None:
                canvas = Image.new("RGBA", layer.size)
            elif layer.size != canvas.size:
                raise ValueError(f"{layer_id} 和其他层尺寸不同，无法合并")
            canvas = Image.alpha_composite(canvas, layer)
        number = max((int(merged.id[1:]) for merged in state.merged), default=0) + 1
        relative = _layer_dir(key) / f"merged_{number}.png"
        canvas.save(_safe_pack_path(directory, relative))
        state.merged.append(MergedLayer(
            id=f"M{number}", path=relative.as_posix(), sources=chosen,
            z=max(stack[layer_id][1] for layer_id in chosen)))
        save_manifest(directory, manifest)
        return manifest


def delete_merged(pack_id: str, segment: int, layer_id: str) -> Manifest:
    """删掉一张合并层（只删它自己的文件，原层和别的合并层不受影响）。"""
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        key, state = _layered_segment(directory, manifest, ring, segment)
        target = next((merged for merged in state.merged if merged.id == layer_id), None)
        if target is None:
            raise ValueError(f"这一段没有合并层 {layer_id}")
        _safe_pack_path(directory, target.path).unlink(missing_ok=True)
        state.merged.remove(target)
        # 合并层编号会被下一次合并复用，留着它的改图历史会错挂到新层上
        history = ring.versions.pop(_layer_target(key, layer_id), None)
        if history is not None:
            _drop_history(directory, _layer_target(key, layer_id), history)
        save_manifest(directory, manifest)
        return manifest


def _layered_segment(directory: Path, manifest: Manifest, ring: RingState,
                     segment: int) -> tuple[str, SegmentLayers]:
    """第 segment 段的 (段身份, 分层结果)；段不存在或还没分完层时报错。"""
    segments = _confirmed_segments(directory, manifest)
    if not 0 <= segment < len(segments):
        raise ValueError(f"没有第 {segment + 1} 段")
    key = segments[segment][0]
    if not _segment_done(directory, ring, key):
        raise ValueError(f"第 {segment + 1} 段还没分层")
    return key, ring.segment_layers[key]


def _layer_stack(state: SegmentLayers) -> dict[str, tuple[str, int]]:
    """层编号 -> (文件, 叠放位置)：原层 L1…L4 的位置就是 1…4。"""
    stack = {f"L{index}": (path, index) for index, path in enumerate(state.layers, start=1)}
    stack.update({merged.id: (merged.path, merged.z) for merged in state.merged})
    return stack


def _drop_merged(directory: Path, state: SegmentLayers) -> None:
    """这一段的图层要重做了：合并层是从旧图层合成的，一并作废。"""
    for merged in state.merged:
        _safe_pack_path(directory, merged.path).unlink(missing_ok=True)
    state.merged = []


def ring_status(pack_id: str) -> dict[str, Any]:
    """返回接缝候选及采用版本是否已因 generate 产物改变而失效。"""
    directory = library.pack_dir(pack_id)
    manifest = load_manifest(directory)
    if manifest.ring is None:
        return {"order": [], "seams": {}, "rendered": []}
    ring = manifest.ring
    # 有些条目的 generate 可能还没做完：状态查询不能因此报错，缺的屏按“没有图”处理
    screens: dict[str, Path] = {}
    originals: dict[str, str] = {}
    for item in manifest.items:
        try:
            screens[item.id] = _screen_source(directory, ring, item)
            originals[item.id] = file_signature(_generate_source(directory, item))
        except ValueError:
            continue
    signatures = {item_id: file_signature(path) for item_id, path in screens.items()}
    heights = {item_id: _image_height(path) for item_id, path in screens.items()}
    ring_height = max((heights[item_id] for item_id in ring.order if item_id in heights),
                      default=0)
    entries = _segment_entries(directory, ring)
    return {
        "order": list(ring.order),
        "seams": _statuses(ring, signatures),
        "rendered": _rendered_files(directory, ring),
        "cuts": list(ring.cuts),
        "cuts_stale": bool(ring.cuts) and _current_cuts_signature(directory, ring) != ring.cuts_sig,
        "segments": [_relative(directory, _segment_current(directory, ring, key, path))
                     for key, path in _segment_files(directory, ring)],
        "segment_layers": [
            [_relative(directory, _layer_current(directory, ring, key, f"L{number}", path))
             for number, path in enumerate(state.layers, start=1)] if state else []
            for key, state in entries],
        "segment_merged": [
            [merged.model_copy(update={"path": _relative(directory, _layer_current(
                directory, ring, key, merged.id, merged.path))})
             for merged in state.merged
             if _safe_pack_path(directory, merged.path).is_file()] if state else []
            for key, state in entries],
        "screen_files": [_relative(directory, screens[item_id]) if item_id in screens else ""
                         for item_id in ring.order],
        "screen_padding": [ring_height - heights[item_id] if item_id in heights else 0
                           for item_id in ring.order],
        "ring_height": ring_height,
        "suggested_cuts": list(ring.suggested_cuts),
        "versions": {key: history for key, history in ring.versions.items()
                     if _history_valid(ring, key, history, originals)},
    }


def _segment_entries(directory: Path, ring: RingState) -> list[tuple[str, SegmentLayers | None]]:
    """与 ring.segments 一一对应的 (段身份, 已完成分层)；没分层或切点失效时分层是 None。"""
    try:
        paths = _rendered_sources(directory, ring)
    except ValueError:
        return [("", None) for _ in ring.segments]
    total = 0
    for path in paths:
        with Image.open(path) as image:
            total += image.width
    if not ring.cuts or _cuts_signature(paths) != ring.cuts_sig:
        return [("", None) for _ in ring.segments]
    keys = [segment_key(start, width)
            for start, width in zip(ring.cuts, segment_widths(ring.cuts, total))]
    return [(key, ring.segment_layers[key] if _segment_done(directory, ring, key) else None)
            for key in keys[:len(ring.segments)]]


def _segment_files(directory: Path, ring: RingState) -> list[tuple[str, Path]]:
    """已导出且文件还在的段图：(段身份, 原段图)。切点失效时段身份未知，用空串。"""
    keys = [key for key, _ in _segment_entries(directory, ring)]
    result = []
    for key, relative in zip(keys, ring.segments):
        path = _safe_pack_path(directory, relative)
        if path.is_file():
            result.append((key, path))
    return result


def _relative(directory: Path, path: Path) -> str:
    return path.relative_to(directory.resolve()).as_posix()


def _image_height(path: Path) -> int:
    with Image.open(path) as image:
        return image.height


def _history_valid(ring: RingState, key: str, history: VersionHistory,
                   originals: dict[str, str]) -> bool:
    """历史的 base_sig 对得上当前“原图”才有效；屏看 generate.png，段 / 层看 cuts_sig。"""
    kind, _, rest = key.partition(":")
    current = originals.get(rest) if kind == "screen" else ring.cuts_sig
    return bool(current) and history.base_sig == current


def segment_key(start: int, width: int) -> str:
    return f"{start}+{width}"


def _layer_dir(key: str) -> Path:
    return LAYER_DIR / key.replace("+", "_")


def _layer_model(manifest: Manifest) -> str:
    # 和模板里整图分层用同一个模型；模板没有分层步骤时用默认的
    template = load_template(manifest.template)
    if LAYER_STEP in template.step_ids():
        return template.step(LAYER_STEP).params.get("model_name") or LAYER_MODEL
    return LAYER_MODEL


def _confirmed_segments(directory: Path, manifest: Manifest) -> list[tuple[str, Path, str]]:
    """已确认切段的每段：(段身份, 段图路径, 分层用的描述)，按切点顺序。"""
    ring = _require_ring(manifest)
    if not ring.segments:
        raise ValueError("请先确认切段")
    paths = _rendered_sources(directory, ring)
    if _cuts_signature(paths) != ring.cuts_sig:
        raise ValueError("切段后合成图变了，请重新自动切段并确认")
    widths = []
    for path in paths:
        with Image.open(path) as image:
            widths.append(image.width)
    total = sum(widths)
    spans = segment_widths(ring.cuts, total)
    if len(spans) != len(ring.segments):
        raise ValueError("段图和切点对不上，请重新确认切段")
    result = []
    for start, width, relative in zip(ring.cuts, spans, ring.segments):
        path = _safe_pack_path(directory, relative)
        if not path.is_file():
            raise ValueError(f"段图不存在：{relative}，请重新确认切段")
        key = segment_key(start, width)
        # 段被文字改过图就拿改后的版本去分层，和画布上看到的一致
        result.append((key, _segment_current(directory, ring, key, path),
                       _segment_prompt(manifest, widths, start, width)))
    return result


def _segment_prompt(manifest: Manifest, widths: list[int], start: int, width: int) -> str:
    """段跨过的各屏的描述 + 风格锁 + 视角；分层模型靠描述认出画里有哪些东西。"""
    ring = _require_ring(manifest)
    total = sum(widths)
    covered, x = [], 0
    for item_id, screen_width in zip(ring.order, widths):
        # 段可能跨过长图末尾绕回开头：按环判断两段区间是否相交
        for shift in (0, total):
            if x + shift < start + width and start < x + shift + screen_width:
                covered.append(item_id)
                break
        x += screen_width
    items = [_find_item(manifest, item_id) for item_id in covered]
    view = items[0].fields.get("view", "") if items else ""
    parts = [item.prompt for item in items] + [manifest.style.prompt, view]
    unique = list(dict.fromkeys(part.strip() for part in parts if part and part.strip()))
    unique.extend(("no characters", "no text"))
    return ", ".join(unique)


def _segment_done(directory: Path, ring: RingState, key: str) -> bool:
    state = ring.segment_layers.get(key)
    return (state is not None and state.source_sig == ring.cuts_sig
            and len(state.layers) == LAYER_COUNT
            and all(_safe_pack_path(directory, path).is_file() for path in state.layers))


def _rendered_sources(directory: Path, ring: RingState) -> list[Path]:
    # 切段在合成后的图上做：接缝已经补好，切出来的段直接就是最终画面
    if len(ring.order) < 2:
        raise ValueError("循环拼接至少需要两个条目，请先规划循环顺序")
    paths = []
    for index, item_id in enumerate(ring.order):
        path = _safe_pack_path(directory, _slot_file(
            item_id, ring.order[index - 1], ring.order[(index + 1) % len(ring.order)]))
        if not path.is_file():
            raise ValueError(f"第 {index + 1} 屏（{item_id}）还没合成，请先生成接缝")
        paths.append(path)
    return paths


def _slot_file(item_id: str, previous_id: str, next_id: str) -> Path:
    # 同一屏可以在环上出现多次，前后邻居不同合成结果就不同：按“屏 + 左右邻居”命名，
    # 邻居没变的位置沿用旧文件，挪动一张图只需重合成它自己和相邻的几屏
    return SLOT_DIR / item_id / f"{previous_id}~{next_id}.png"


def _ring_long_image(paths: list[Path]) -> Image.Image:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return long_image(images)


def _cuts_signature(paths: list[Path]) -> str:
    return "|".join(file_signature(path) for path in paths)


def _current_cuts_signature(directory: Path, ring: RingState) -> str:
    try:
        return _cuts_signature(_rendered_sources(directory, ring))
    except ValueError:
        # 有屏没合成：签名对不上，按失效处理
        return ""


def _rendered_files(directory: Path, ring: RingState | None) -> list[str]:
    """与 ring.order 一一对应的合成图；还没合成的位置是空串。"""
    if ring is None:
        return []
    files = []
    for index, item_id in enumerate(ring.order):
        relative = _slot_file(
            item_id, ring.order[index - 1], ring.order[(index + 1) % len(ring.order)])
        files.append(relative.as_posix() if _safe_pack_path(directory, relative).is_file() else "")
    return files


def _statuses(ring: RingState, signatures: dict[str, str]) -> dict[str, RingSeamStatus]:
    statuses = {}
    for key, seam in ring.seams.items():
        candidate = next((item for item in seam.candidates if item.id == seam.adopted), None)
        left_id, right_id = _split_key(key)
        stale = bool(seam.adopted) and (
            candidate is None
            or signatures.get(left_id) != candidate.left_sig
            or signatures.get(right_id) != candidate.right_sig
        )
        statuses[key] = RingSeamStatus(
            candidates=list(seam.candidates),
            adopted=seam.adopted,
            stale=stale,
            has_candidates=bool(seam.candidates),
        )
    return statuses


def _generate_source(directory: Path, item: CollectionItem) -> Path:
    state = item.steps.get(GENERATE_STEP)
    if state is None or state.status != "done":
        raise ValueError(f"条目 {item.id} 的 generate 步骤尚未完成")
    if not state.outputs:
        raise ValueError(f"条目 {item.id} 的 generate 步骤没有产物")
    path = _safe_pack_path(directory, state.outputs[0])
    if not path.is_file():
        raise ValueError(f"条目 {item.id} 的 generate 产物不存在：{state.outputs[0]}")
    return path


def _screen_source(directory: Path, ring: RingState | None, item: CollectionItem) -> Path:
    """这一屏当前显示 / 参与合成的图：外扩或改过图的当前版本，没有就是 generate.png。

    版本只有在 base_sig 等于 generate.png 当前签名时才有效：原图被重新生成后，
    旧版本是基于旧图做的，不能再套到新图上。_generate_source 仍是“原图”。"""
    original = _generate_source(directory, item)
    if ring is None:
        return original
    current = _current_version(directory, ring, _screen_target(item.id),
                               file_signature(original))
    return current or original


def _screen_target(item_id: str) -> str:
    return f"screen:{item_id}"


def _segment_target(key: str) -> str:
    return f"segment:{key}"


def _layer_target(key: str, layer_id: str) -> str:
    return f"layer:{key}:{layer_id}"


def _current_version(directory: Path, ring: RingState, target: str,
                     base_sig: str) -> Path | None:
    """目标的当前版本文件；没有历史、历史作废（base_sig 对不上）、选了原图或文件丢了都是 None。"""
    history = ring.versions.get(target)
    if history is None or history.base_sig != base_sig:
        return None
    if not 0 <= history.current < len(history.versions):
        return None
    path = _safe_pack_path(directory, history.versions[history.current].file)
    return path if path.is_file() else None


def _segment_current(directory: Path, ring: RingState, key: str, path: Path) -> Path:
    # 段身份未知（切点已失效）时没有版本可言
    if not key:
        return path
    return _current_version(directory, ring, _segment_target(key), ring.cuts_sig) or path


def _layer_current(directory: Path, ring: RingState, key: str, layer_id: str,
                   relative: str) -> Path:
    original = _safe_pack_path(directory, relative)
    return _current_version(directory, ring, _layer_target(key, layer_id),
                            ring.cuts_sig) or original


def _version_dir(target: str) -> Path:
    return VERSION_DIR / target.replace(":", "_").replace("+", "_")


def _drop_history(directory: Path, target: str, history: VersionHistory) -> None:
    """删掉一条历史的全部版本文件（整个目录按目标独占，直接删目录）。"""
    shutil.rmtree(_safe_pack_path(directory, _version_dir(target)), ignore_errors=True)


def _drop_layer_versions(directory: Path, ring: RingState, key: str) -> None:
    for target in [target for target in ring.versions if target.startswith(f"layer:{key}:")]:
        _drop_history(directory, target, ring.versions.pop(target))


def _max_height(paths: list[Path]) -> int:
    return max(_image_height(path) for path in paths)


def _fitted_edge_strips(path: Path, height: int) -> tuple[np.ndarray, np.ndarray]:
    """同 seamless_loop.edge_strips，但先把图补齐到 height：排序要在同一高度上比较边缘。"""
    with Image.open(path) as image:
        fitted = fit_height(image, height)
    if fitted.width < EDGE_COLUMNS:
        raise ValueError(f"图片宽度不足 {EDGE_COLUMNS} 列：{path}")
    pixels = np.asarray(fitted, dtype=np.float32)
    return pixels[:, -EDGE_COLUMNS:].mean(axis=1), pixels[:, :EDGE_COLUMNS].mean(axis=1)


def _ring_height(directory: Path, manifest: Manifest, ring: RingState) -> int:
    """环高度 = 环上各屏当前图高度的最大值；较矮的屏合成时用边缘行补齐到这个高度。"""
    items = {item.id: item for item in manifest.items}
    return _max_height([_screen_source(directory, ring, items[item_id])
                        for item_id in dict.fromkeys(ring.order) if item_id in items])


def _refresh_slots(directory: Path, manifest: Manifest, ring: RingState,
                   item_ids: set[str]) -> None:
    """重新合成 item_ids 的每个出现位置；环高度变了的话，其余已合成的位置高度也要跟着对齐，
    否则长图拼接会因为高度不一致报错。"""
    count = len(ring.order)
    if count < 2:
        return
    height = _ring_height(directory, manifest, ring)
    for index, item_id in enumerate(ring.order):
        slot = _safe_pack_path(directory, _slot_file(
            item_id, ring.order[index - 1], ring.order[(index + 1) % count]))
        if item_id in item_ids or (slot.is_file() and _image_height(slot) != height):
            _render_slot(directory, manifest, ring, index)


def _resolve_target(directory: Path, manifest: Manifest, ring: RingState,
                    target: str) -> tuple[str, Path, str, Path]:
    """目标键 -> (种类, 原图, 原图签名 base_sig, 当前图)。键不认识或目标不存在时报错。"""
    kind, _, rest = target.partition(":")
    if kind == "screen" and rest:
        item = _find_item(manifest, rest)
        original = _generate_source(directory, item)
        return (kind, original, file_signature(original),
                _screen_source(directory, ring, item))
    if kind == "segment" and rest:
        segments = _confirmed_segments(directory, manifest)
        keys = [key for key, _, _ in segments]
        if rest not in keys:
            raise ValueError(f"没有这一段：{rest}")
        position = keys.index(rest)
        return (kind, _safe_pack_path(directory, ring.segments[position]), ring.cuts_sig,
                segments[position][1])
    if kind == "layer" and rest.count(":") == 1:
        key, _, layer_id = rest.partition(":")
        if not _segment_done(directory, ring, key):
            raise ValueError(f"段 {key} 还没分层")
        stack = _layer_stack(ring.segment_layers[key])
        if layer_id not in stack:
            raise ValueError(f"段 {key} 没有层 {layer_id}")
        return (kind, _safe_pack_path(directory, stack[layer_id][0]), ring.cuts_sig,
                _layer_current(directory, ring, key, layer_id, stack[layer_id][0]))
    raise ValueError(f"无效的目标：{target}")


def _save_version(directory: Path, ring: RingState, target: str, base_sig: str,
                  image: Image.Image, kind: str, prompt: str = "") -> None:
    """把新版本落盘、追加到历史并设为当前。原图变了（base_sig 对不上）时先清掉旧历史。"""
    history = ring.versions.get(target)
    if history is not None and history.base_sig != base_sig:
        _drop_history(directory, target, history)
        history = None
    if history is None:
        history = ring.versions[target] = VersionHistory(base_sig=base_sig)
    relative = _version_dir(target) / f"{len(history.versions) + 1}.png"
    path = _safe_pack_path(directory, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    history.versions.append(ImageVersion(file=relative.as_posix(), kind=kind, prompt=prompt))
    history.current = len(history.versions) - 1


def _run_model_edit(generator, reference: Image.Image, prompt: str) -> Image.Image:
    """用图生图模型改 reference，返回和 reference 同尺寸的结果。

    Qwen-Image-2.1 要求宽高是 32 的倍数且在 256～2048，而画布尺寸（如 700x320）不一定满足，
    面积也要压到编辑分辨率附近（1024 下会忽略指令，且 8GB 显存吃紧）。所以给模型的尺寸按
    比例取合法值，输出再缩放回 reference 的尺寸，调用方拿到的永远和输入一样大。"""
    width, height = reference.size
    model_width, model_height = model_dimensions(width, height, EDIT_SIDE * EDIT_SIDE)
    with tempfile.TemporaryDirectory() as temp:
        reference_path = Path(temp) / "reference.png"
        reference.save(reference_path, format="PNG")
        generator.parse_params({
            "content": prompt,
            "reference_image": str(reference_path),
            "width": model_width,
            "height": model_height,
            "output_dir": "ring_edit",
        })
        output = call_blocking(generator.generate())
        if not output or not Path(output).is_file():
            raise RuntimeError("模型没有输出图片，请查看后端日志")
        with Image.open(output) as opened:
            result = opened.copy()
        Path(output).unlink(missing_ok=True)
    if result.size != (width, height):
        result = result.resize((width, height), Image.LANCZOS)
    return result


def _has_alpha(image: Image.Image) -> bool:
    """输出是否真的带透明信息：RGBA 模式但全不透明等于没有（模型常把透明底铺成实色）。"""
    return image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255


def extend_screens(pack_id: str, item_ids: list[str] | None = None, pad: int = EXTEND_PAD,
                   cancel_event: threading.Event | None = None,
                   on_progress: Callable[[str, float], None] | None = None) -> Manifest:
    """给选中的屏（没选 = 环上全部屏）上下各外扩 pad 像素，结果作为新版本，原图保留。

    一次持有模型租约包住整个循环，每屏做完立刻落盘并重新合成，中途停止或失败后已完成的保留。
    on_progress(屏 id, 0~1)。分段之后不能外扩：长图高度变了段图就全对不上。"""
    if not 16 <= pad <= 256:
        raise ValueError(f"外扩像素必须在 16～256 之间：{pad}")
    directory = library.pack_dir(pack_id)
    cancel_event = cancel_event or threading.Event()
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        if ring.segments:
            raise ValueError("请在分段之前外扩")
        on_ring = list(dict.fromkeys(ring.order))
        if item_ids is None:
            targets = on_ring
        else:
            unknown = sorted(set(item_ids) - set(on_ring))
            if unknown:
                raise ValueError(f"环上没有这些屏：{', '.join(unknown)}")
            targets = list(dict.fromkeys(item_ids))
        if not targets:
            return manifest

        try:
            with GeneratorFactory.acquire(FactoryType.Image, EDIT_MODEL) as generator:
                generator.cancel_event = cancel_event
                try:
                    generator.ensure_model_loaded()
                    for item_id in targets:
                        if cancel_event.is_set():
                            generator.check_cancelled()
                        target = _screen_target(item_id)
                        _, _, base_sig, current = _resolve_target(
                            directory, manifest, ring, target)
                        if on_progress is not None:
                            on_progress(item_id, 0.0)
                            generator.progress_callback = functools.partial(on_progress, item_id)
                        with Image.open(current) as opened:
                            source = opened.convert("RGB")
                        result = _run_model_edit(
                            generator, extend_canvas(source, pad), EXTEND_PROMPT)
                        # 模型会顺手改动中间的原图：把原图像素换回去，只留新补的上下两条
                        _save_version(directory, ring, target, base_sig,
                                      restore_middle(result, source, pad), "extend")
                        _refresh_slots(directory, manifest, ring, {item_id})
                        save_manifest(directory, manifest)
                        if on_progress is not None:
                            on_progress(item_id, 1.0)
                finally:
                    generator.cancel_event = None
                    generator.progress_callback = None
        except Exception:
            # 每屏成功后已单独落盘；此处再写一次，确保失败不会丢失已经推进的状态
            save_manifest(directory, manifest)
            raise
        return manifest


def edit_target(pack_id: str, target: str, prompt: str,
                cancel_event: threading.Event | None = None,
                on_progress: Callable[[str, float], None] | None = None) -> Manifest:
    """按文字描述改一个目标（屏 / 段 / 层）的当前图，结果作为新版本，旧版本保留。

    on_progress(目标键, 0~1)。"""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("请输入修改描述")
    directory = library.pack_dir(pack_id)
    cancel_event = cancel_event or threading.Event()
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        kind, _, base_sig, current = _resolve_target(directory, manifest, ring, target)
        try:
            with GeneratorFactory.acquire(FactoryType.Image, EDIT_MODEL) as generator:
                generator.cancel_event = cancel_event
                try:
                    generator.ensure_model_loaded()
                    if cancel_event.is_set():
                        generator.check_cancelled()
                    if on_progress is not None:
                        on_progress(target, 0.0)
                        generator.progress_callback = functools.partial(on_progress, target)
                    with Image.open(current) as opened:
                        source = opened.convert("RGBA" if kind == "layer" else "RGB")
                    result = _run_model_edit(generator, source, prompt)
                    if kind == "layer":
                        # 层是透明 RGBA。模型输出真带透明就保留；否则沿用原层的 alpha。
                        # 这是权宜做法：模型的 RGB 输出和原 alpha 未必对得上（比如改了形状），
                        # 需要实机确认效果后再决定是否要换成别的处理。
                        if not _has_alpha(result):
                            result = result.convert("RGB")
                            result.putalpha(source.getchannel("A"))
                    else:
                        result = result.convert("RGB")
                    _save_version(directory, ring, target, base_sig, result, "edit", prompt)
                    if kind == "screen":
                        _refresh_slots(directory, manifest, ring, {target.partition(":")[2]})
                    save_manifest(directory, manifest)
                    if on_progress is not None:
                        on_progress(target, 1.0)
                finally:
                    generator.cancel_event = None
                    generator.progress_callback = None
        except Exception:
            save_manifest(directory, manifest)
            raise
        return manifest


def select_version(pack_id: str, target: str, index: int) -> Manifest:
    """切换目标当前使用的版本：-1 是原图，否则是历史里的下标。纯 CPU，不碰模型租约。"""
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        kind, _, base_sig, _ = _resolve_target(directory, manifest, ring, target)
        history = ring.versions.get(target)
        if history is None or history.base_sig != base_sig:
            # 没有有效历史：只有“用原图”说得通
            if index != -1:
                raise ValueError(f"{target} 没有历史版本")
            return manifest
        if not -1 <= index < len(history.versions):
            raise ValueError(f"{target} 没有第 {index} 个版本")
        history.current = index
        if kind == "screen":
            _refresh_slots(directory, manifest, ring, {target.partition(":")[2]})
        save_manifest(directory, manifest)
        return manifest


def _ring_members(manifest: Manifest, item_ids: list[str] | None) -> list[CollectionItem]:
    # 成员按包内顺序返回：排序交给 order_loop，这里只决定谁参与
    if item_ids is None:
        previous = manifest.ring.order if manifest.ring else []
        chosen = set(previous) & {item.id for item in manifest.items}
        if not chosen:
            return list(manifest.items)
    else:
        chosen = set(item_ids)
        unknown = sorted(chosen - {item.id for item in manifest.items})
        if unknown:
            raise ValueError(f"资源包里没有这些条目：{', '.join(unknown)}")
    return [item for item in manifest.items if item.id in chosen]


def _require_ring(manifest: Manifest) -> RingState:
    if manifest.ring is None:
        raise ValueError("请先规划循环拼接顺序")
    return manifest.ring


def _find_item(manifest: Manifest, item_id: str) -> CollectionItem:
    for item in manifest.items:
        if item.id == item_id:
            return item
    raise ValueError(f"循环接缝引用了不存在的条目：{item_id}")


def _build_prompt(left: str, right: str, style: str, view: str, extra: str = "") -> str:
    # 追加的描述放在风格锁之后：风格锁定调子，追加句只改这个接缝里画什么
    parts = list(dict.fromkeys(part.strip() for part in (left, right, style, view, extra)
                               if part and part.strip()))
    parts.extend(("no characters", "no text"))
    return ", ".join(parts)


def _seam_key(left_id: str, right_id: str) -> str:
    return f"{left_id}>{right_id}"


def _split_key(key: str) -> tuple[str, str]:
    parts = key.split(">", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"无效的接缝 key：{key}")
    return parts[0], parts[1]


def _next_candidate_id(seam: SeamState) -> str:
    numbers = [int(candidate.id[1:]) for candidate in seam.candidates
               if candidate.id.startswith("c") and candidate.id[1:].isdigit()]
    return f"c{max(numbers, default=0) + 1}"


def _prune_candidates(directory: Path, seam: SeamState) -> None:
    while len(seam.candidates) > MAX_CANDIDATES:
        old = next((candidate for candidate in seam.candidates
                    if candidate.id != seam.adopted), None)
        if old is None:
            break
        seam.candidates.remove(old)
        _safe_pack_path(directory, old.file).unlink(missing_ok=True)


def _adopted_candidate(directory: Path, ring: RingState,
                       key: str) -> SeamCandidate | None:
    seam = ring.seams.get(key)
    if seam is None or not seam.adopted:
        return None
    candidate = next((item for item in seam.candidates if item.id == seam.adopted), None)
    if candidate is None:
        raise ValueError(f"接缝 {key} 指向不存在的采用候选 {seam.adopted}")
    path = _safe_pack_path(directory, candidate.file)
    if not path.is_file():
        raise FileNotFoundError(f"接缝 {key} 的采用补丁不存在：{candidate.file}")
    return candidate


def _safe_pack_path(directory: Path, relative_path: str | Path) -> Path:
    root = directory.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"资源包路径超出目录：{relative_path}") from exc
    return path
