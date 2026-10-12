"""分层后的合并层：按叠放顺序合成、原层保留、重做和改切点时作废。不加载权重或 GPU。"""
import pytest
from PIL import Image

from src.backend.core.collection import library, ring
from src.backend.core.collection.manifest import load_manifest, save_manifest
from src.shared.layer_stack import stack_key
from src.shared.schemas import CollectionItem, Manifest, SegmentLayers, StepState

RED, GREEN, BLUE = (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)


def _layered_pack(tmp_path, monkeypatch):
    """三屏、两段，第 0 段已分好 4 层：L1 整片红，L2 左半绿，L3 右半蓝，L4 全透明。"""
    monkeypatch.setattr(library, "library_root", lambda: tmp_path)
    directory = tmp_path / "pack"
    items = []
    for index, item_id in enumerate(("a", "b", "c")):
        (directory / item_id).mkdir(parents=True)
        Image.new("RGB", (700, 64), (index * 70, 20, 100)).save(directory / item_id / "generate.png")
        items.append(CollectionItem(
            id=item_id, prompt=item_id,
            steps={"generate": StepState(status="done", outputs=[f"{item_id}/generate.png"])}))
    save_manifest(directory, Manifest(
        id="pack", type="scene", template="scene_background", items=items))
    manifest = ring.plan_ring("pack")
    for index, item_id in enumerate(manifest.ring.order):
        previous = manifest.ring.order[index - 1]
        following = manifest.ring.order[(index + 1) % 3]
        path = directory / ring._slot_file(item_id, previous, following)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (700, 64), (index * 70, 20, 100)).save(path)
    ring.confirm_cuts("pack", [0, 1050])

    manifest = load_manifest(directory)
    key = ring.segment_key(0, 1050)
    layer_dir = directory / ring._layer_dir(key)
    layer_dir.mkdir(parents=True)
    pictures = [
        Image.new("RGBA", (100, 8), RED),
        Image.new("RGBA", (100, 8), (0, 0, 0, 0)),
        Image.new("RGBA", (100, 8), (0, 0, 0, 0)),
        Image.new("RGBA", (100, 8), (0, 0, 0, 0)),
    ]
    pictures[1].paste(Image.new("RGBA", (50, 8), GREEN), (0, 0))
    pictures[2].paste(Image.new("RGBA", (50, 8), BLUE), (50, 0))
    relatives = []
    for number, picture in enumerate(pictures, start=1):
        relative = (ring._layer_dir(key) / f"layer_{number}.png").as_posix()
        picture.save(directory / relative)
        relatives.append(relative)
    manifest.ring.segment_layers[key] = SegmentLayers(
        source_sig=manifest.ring.cuts_sig, layers=relatives)
    save_manifest(directory, manifest)
    return directory


def _pixel(directory, relative, x):
    with Image.open(directory / relative) as image:
        return image.convert("RGBA").getpixel((x, 4))


def test_stack_key_puts_merged_above_original_at_the_same_position():
    assert stack_key("L2", 2) < stack_key("M1", 2) < stack_key("L3", 3)
    with pytest.raises(ValueError):
        stack_key("X1", 1)


def test_merge_composites_in_stack_order_and_keeps_the_originals(tmp_path, monkeypatch):
    directory = _layered_pack(tmp_path, monkeypatch)

    manifest = ring.merge_layers("pack", 0, ["L2", "L1"])

    state = next(iter(manifest.ring.segment_layers.values()))
    assert len(state.layers) == 4
    merged = state.merged[0]
    assert (merged.id, merged.sources, merged.z) == ("M1", ["L1", "L2"], 2)
    assert _pixel(directory, merged.path, 10) == GREEN
    assert _pixel(directory, merged.path, 90) == RED
    assert all((directory / path).is_file() for path in state.layers)


def test_merge_accepts_a_merged_layer_as_source_and_numbers_the_new_one(tmp_path, monkeypatch):
    directory = _layered_pack(tmp_path, monkeypatch)
    ring.merge_layers("pack", 0, ["L1", "L2"])

    manifest = ring.merge_layers("pack", 0, ["M1", "L3"])

    first, second = next(iter(manifest.ring.segment_layers.values())).merged
    assert (second.id, second.sources, second.z) == ("M2", ["M1", "L3"], 3)
    assert _pixel(directory, second.path, 10) == GREEN
    assert _pixel(directory, second.path, 90) == BLUE
    assert first.id == "M1"


def test_status_lists_merged_layers_per_segment(tmp_path, monkeypatch):
    _layered_pack(tmp_path, monkeypatch)
    ring.merge_layers("pack", 0, ["L1", "L2"])

    status = ring.ring_status("pack")

    assert [merged.id for merged in status["segment_merged"][0]] == ["M1"]
    assert status["segment_merged"][1] == []


def test_merge_rejects_bad_requests(tmp_path, monkeypatch):
    _layered_pack(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="至少选两层"):
        ring.merge_layers("pack", 0, ["L1", "L1"])
    with pytest.raises(ValueError, match="没有这些层"):
        ring.merge_layers("pack", 0, ["L1", "L9"])
    with pytest.raises(ValueError, match="还没分层"):
        ring.merge_layers("pack", 1, ["L1", "L2"])
    with pytest.raises(ValueError, match="没有第 3 段"):
        ring.merge_layers("pack", 2, ["L1", "L2"])


def test_delete_merged_removes_only_that_layer(tmp_path, monkeypatch):
    directory = _layered_pack(tmp_path, monkeypatch)
    ring.merge_layers("pack", 0, ["L1", "L2"])
    ring.merge_layers("pack", 0, ["L1", "L3"])

    manifest = ring.delete_merged("pack", 0, "M1")

    state = next(iter(manifest.ring.segment_layers.values()))
    assert [merged.id for merged in state.merged] == ["M2"]
    assert not list(directory.rglob("merged_1.png"))
    assert (directory / state.merged[0].path).is_file()
    with pytest.raises(ValueError, match="没有合并层"):
        ring.delete_merged("pack", 0, "M1")


def test_redoing_a_segment_discards_its_merged_layers(tmp_path, monkeypatch):
    directory = _layered_pack(tmp_path, monkeypatch)
    manifest = ring.merge_layers("pack", 0, ["L1", "L2"])
    state = next(iter(manifest.ring.segment_layers.values()))
    path = directory / state.merged[0].path

    ring._drop_merged(directory, state)

    assert state.merged == []
    assert not path.exists()


def test_confirming_cuts_discards_all_layers_and_merged_layers(tmp_path, monkeypatch):
    directory = _layered_pack(tmp_path, monkeypatch)
    ring.merge_layers("pack", 0, ["L1", "L2"])

    manifest = ring.confirm_cuts("pack", [0, 900])

    assert manifest.ring.segment_layers == {}
    assert not list(directory.rglob("merged_1.png"))
