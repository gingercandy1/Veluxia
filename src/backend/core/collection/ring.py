"""资源包级循环接缝规划、候选生成、采用和成品回贴。"""
import random
import shutil
import threading
from pathlib import Path
from typing import Any

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
from src.backend.core.collection.seamless_loop import (
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
from src.shared.schemas import (
    CollectionItem,
    Manifest,
    RingSeamStatus,
    RingState,
    SeamCandidate,
    SeamState,
    SegmentLayers,
)

MAX_CANDIDATES = 4
HARMONIZE_STRENGTH = 0.3
GENERATE_STEP = "generate"
SEGMENT_DIR = Path("ring") / "segments"
# 每段的分层结果放在各自目录：按段身份命名，切点变了旧目录不会被新段误用
LAYER_DIR = Path("ring") / "layers"
LAYER_STEP = "layer"
LAYER_MODEL = "Qwen-Image-Layered"
# 和用户定的一致：每段固定 4 层，之后在"检查与合并"里按需合并
LAYER_COUNT = 4


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

        sources = [_generate_source(directory, item) for item in members]
        strips = [edge_strips(path) for path in sources]
        order_ids = [members[index].id
                     for index in order_loop(edge_cost_matrix(strips))]
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
        if previous is not None and previous.order == order_ids:
            # 顺序没变，长图坐标也没变：切点和已导出的段继续有效（图变了由签名判断）
            new_ring.cuts, new_ring.cuts_sig = previous.cuts, previous.cuts_sig
            new_ring.segments = previous.segments
        manifest.ring = new_ring
        save_manifest(directory, manifest)
        return manifest


def generate_seams(pack_id: str, keys: list[str] | None = None,
                   prompt_override: str | None = None,
                   cancel_event: threading.Event | None = None,
                   extra_prompt: str | None = None) -> Manifest:
    """按需生成接缝候选；一次持有模型租约，逐缝保存进度以便失败后续跑。"""
    directory = library.pack_dir(pack_id)
    cancel_event = cancel_event or threading.Event()
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        template = load_template(manifest.template)
        model_name = template.step(GENERATE_STEP).params.get("model_name")
        if not model_name:
            raise ValueError(f"模板 {manifest.template} 的 generate 步骤缺少 model_name")

        source_paths = {
            item.id: _generate_source(directory, item)
            for item in manifest.items
        }
        signatures = {item_id: file_signature(path) for item_id, path in source_paths.items()}
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
                        with Image.open(source_paths[left_id]) as left_image, \
                                Image.open(source_paths[right_id]) as right_image:
                            window = build_window(left_image, right_image)
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
                        save_manifest(directory, manifest)
                finally:
                    generator.cancel_event = None
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
        items = {item.id: item for item in manifest.items}
        for index, item_id in enumerate(ring.order):
            item = items.get(item_id)
            if item is None:
                raise ValueError(f"循环顺序引用了不存在的条目：{item_id}")
            source = _generate_source(directory, item)
            with Image.open(source) as source_image:
                rendered = source_image.convert("RGB")
            previous_id = ring.order[index - 1]
            next_id = ring.order[(index + 1) % len(ring.order)]
            previous = _adopted_candidate(directory, ring, _seam_key(previous_id, item_id))
            following = _adopted_candidate(directory, ring, _seam_key(item_id, next_id))
            if previous:
                with Image.open(_safe_pack_path(directory, previous.file)) as patch:
                    rendered = apply_patch(rendered, patch, side="left")
            if following:
                with Image.open(_safe_pack_path(directory, following.file)) as patch:
                    rendered = apply_patch(rendered, patch, side="right")
            target = _safe_pack_path(directory, Path(item_id) / "ring.png")
            target.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(target, format="PNG")
        return manifest


def auto_cuts(pack_id: str, cancel_event: threading.Event | None = None) -> Manifest:
    """估整条环的深度，自动选切段位置：切线落在物体之间的空隙，一棵树不会分进两段。"""
    directory = library.pack_dir(pack_id)
    with library._claim(pack_id):
        manifest = load_manifest(directory)
        ring = _require_ring(manifest)
        paths = _rendered_sources(directory, ring)
        image = _ring_long_image(paths)
        depth = estimate_ring_depth(image, cancel_event)
        ring.cuts = plan_cuts(column_cost(depth))
        ring.cuts_sig = _cuts_signature(paths)
        # 切点变了，之前导出的段图不再对应，等用户确认后重新导出
        ring.segments = []
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
        # 段身份没变的分层结果留着（拖一条切线只影响相邻两段，其余几小时的分层不白做）
        keep = {segment_key(start, width)
                for start, width in zip(ring.cuts, segment_widths(ring.cuts, image.width))}
        for key in [key for key in ring.segment_layers if key not in keep]:
            del ring.segment_layers[key]
            shutil.rmtree(_safe_pack_path(directory, _layer_dir(key)), ignore_errors=True)
        save_manifest(directory, manifest)
        return manifest


def layer_segments(pack_id: str, indices: list[int] | None = None,
                   cancel_event: threading.Event | None = None) -> Manifest:
    """每段交给 Qwen-Layered 分成 LAYER_COUNT 层。

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
                ring.segment_layers.pop(key, None)
        else:
            todo = [segment for segment in segments
                    if not _segment_done(directory, ring, segment[0])]
        if not todo:
            return manifest

        model_name = _layer_model(manifest)
        try:
            with GeneratorFactory.acquire(FactoryType.Image, model_name) as generator:
                generator.cancel_event = cancel_event
                try:
                    generator.ensure_model_loaded()
                    for key, path, prompt in todo:
                        state = ring.segment_layers.get(key)
                        # 只重做去噪时（上次停在去噪阶段）编码可以复用：描述没变就不用再加载编码器
                        if state is not None and state.prompt == prompt and state.embeds and \
                                _safe_pack_path(directory, state.embeds).is_file():
                            continue
                        generator.parse_params({"content": prompt})
                        embeds = generator.save_prompt_embeds()
                        relative = _layer_dir(key) / "encode.pt"
                        move_into(Path(embeds), _safe_pack_path(directory, relative))
                        ring.segment_layers[key] = SegmentLayers(
                            source_sig=ring.cuts_sig, prompt=prompt,
                            embeds=relative.as_posix())
                        save_manifest(directory, manifest)
                    for key, path, prompt in todo:
                        state = ring.segment_layers[key]
                        generator.parse_params({
                            "content": prompt, "layers": LAYER_COUNT,
                            "reference_image": str(path),
                            "prompt_embeds_path": str(_safe_pack_path(directory, state.embeds)),
                        })
                        outputs = call_blocking(generator.generate())
                        if not outputs or not all(Path(p).is_file() for p in outputs):
                            raise RuntimeError(f"段 {key} 没有生成图层，请查看后端日志")
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


def ring_status(pack_id: str) -> dict[str, Any]:
    """返回接缝候选及采用版本是否已因 generate 产物改变而失效。"""
    directory = library.pack_dir(pack_id)
    manifest = load_manifest(directory)
    if manifest.ring is None:
        return {"order": [], "seams": {}, "rendered": _rendered_files(directory, manifest)}
    signatures: dict[str, str] = {}
    for item in manifest.items:
        state = item.steps.get(GENERATE_STEP)
        if state and state.status == "done" and state.outputs:
            path = _safe_pack_path(directory, state.outputs[0])
            if path.is_file():
                signatures[item.id] = file_signature(path)
    ring = manifest.ring
    return {
        "order": list(ring.order),
        "seams": _statuses(ring, signatures),
        "rendered": _rendered_files(directory, manifest),
        "cuts": list(ring.cuts),
        "cuts_stale": bool(ring.cuts) and _current_cuts_signature(directory, ring) != ring.cuts_sig,
        "segments": [path for path in ring.segments
                     if _safe_pack_path(directory, path).is_file()],
        "segment_layers": _segment_layer_files(directory, ring),
    }


def _segment_layer_files(directory: Path, ring: RingState) -> list[list[str]]:
    """与 ring.segments 一一对应的已完成图层；切点失效时全部为空。"""
    try:
        paths = _rendered_sources(directory, ring)
    except ValueError:
        return [[] for _ in ring.segments]
    total = 0
    for path in paths:
        with Image.open(path) as image:
            total += image.width
    if not ring.cuts or _cuts_signature(paths) != ring.cuts_sig:
        return [[] for _ in ring.segments]
    keys = [segment_key(start, width)
            for start, width in zip(ring.cuts, segment_widths(ring.cuts, total))]
    return [list(ring.segment_layers[key].layers) if _segment_done(directory, ring, key) else []
            for key in keys[:len(ring.segments)]]


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
        result.append((segment_key(start, width), path,
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
    for item_id in ring.order:
        path = _safe_pack_path(directory, Path(item_id) / "ring.png")
        if not path.is_file():
            raise ValueError(f"屏 {item_id} 还没合成，请先点“合成”")
        paths.append(path)
    return paths


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


def _rendered_files(directory: Path, manifest: Manifest) -> dict[str, str]:
    rendered = {}
    for item in manifest.items:
        relative = Path(item.id) / "ring.png"
        if _safe_pack_path(directory, relative).is_file():
            rendered[item.id] = relative.as_posix()
    return rendered


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
