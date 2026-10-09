import os

from src.backend.core.collection import generator_runner
from src.backend.core.collection.generator_runner import move_into


def test_move_into_overwrites_existing_target(tmp_path):
    source, target = tmp_path / "new.png", tmp_path / "pack" / "layers_1.png"
    target.parent.mkdir()
    target.write_bytes(b"old")
    source.write_bytes(b"new")
    assert move_into(source, target) == target
    assert target.read_bytes() == b"new"
    assert not source.exists()


def test_move_into_sets_aside_a_target_that_cannot_be_overwritten(tmp_path, monkeypatch):
    """Windows 上旧产物被别的进程打开时，直接覆盖会 WinError 5；改名挪开后应能放进新文件。"""
    source, target = tmp_path / "new.png", tmp_path / "layers_1.png"
    target.write_bytes(b"old")
    source.write_bytes(b"new")
    real_replace = os.replace

    def locked_replace(src, dst):
        if dst == target and os.path.exists(dst):
            raise PermissionError(5, "拒绝访问。")
        real_replace(src, dst)

    monkeypatch.setattr(generator_runner.os, "replace", locked_replace)
    move_into(source, target)
    assert target.read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == ["layers_1.png"]
