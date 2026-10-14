"""框选局部改图后端契约：schema 校验、框解析、裁剪改图只动框内。不连模型，用假生成器。"""
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from src.backend.core.collection import ring
from src.shared.schemas import RingEditRequest


class _StubGenerator:
    """parse_params 记下宽高，generate 返回该尺寸的纯品红图。"""

    def __init__(self, tmp_path):
        self._tmp = Path(tmp_path)
        self.params = {}

    def parse_params(self, raw):
        self.params = dict(raw)

    def generate(self):
        path = self._tmp / "fake.png"
        Image.new("RGB", (self.params["width"], self.params["height"]), (255, 0, 255)).save(path)
        return path


def test_schema_bbox_ok_and_missing():
    req = RingEditRequest(pack_id="p", target="segment:0+10", prompt="steeper",
                          bbox=[0.25, 0.25, 0.5, 0.5])
    assert req.bbox == [0.25, 0.25, 0.5, 0.5]
    assert RingEditRequest(pack_id="p", target="segment:0+10", prompt="x").bbox is None


def test_schema_bbox_rejects_bad_shapes():
    with pytest.raises(ValidationError):
        RingEditRequest(pack_id="p", target="t", prompt="x", bbox=[0.1, 0.2, 0.3])
    with pytest.raises(ValidationError):
        RingEditRequest(pack_id="p", target="t", prompt="x", bbox=[0.9, 0.9, 0.3, 0.3])


def test_parse_bbox_none_and_errors():
    assert ring._parse_bbox(None) is None
    assert ring._parse_bbox([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)
    with pytest.raises(ValueError):
        ring._parse_bbox([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        ring._parse_bbox([0.8, 0.8, 0.5, 0.5])


def test_region_edit_keeps_outside_and_repaints_inside(tmp_path):
    source = Image.new("RGB", (400, 300), (0, 0, 255))
    result = ring._run_region_edit(_StubGenerator(tmp_path), source, "steeper",
                                   (0.25, 0.25, 0.5, 0.5))
    assert result.size == (400, 300) and result.mode == "RGB"
    # 框外原样：左上角还是纯蓝
    assert result.getpixel((10, 10)) == (0, 0, 255)
    assert result.getpixel((390, 290)) == (0, 0, 255)
    # 框中心被换成模型输出（纯品红），羽化只影响边缘
    assert result.getpixel((200, 150)) == (255, 0, 255)


def test_region_edit_rejects_tiny_boxes(tmp_path):
    source = Image.new("RGB", (400, 300), (0, 0, 255))
    with pytest.raises(ValueError):
        ring._run_region_edit(_StubGenerator(tmp_path), source, "x", (0.0, 0.0, 0.02, 0.02))
