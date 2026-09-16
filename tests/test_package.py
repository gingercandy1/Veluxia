"""前后端打包统一入口的单测（ADR 0002）。

Seam：package.py 的纯编排（参数解析、步骤分发、zip 落盘），subprocess 可注入。
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

import package as pk


def test_parse_flows():
    assert pk.parse_args(["front"]).flow == "front"
    assert pk.parse_args(["backend"]).flow == "backend"
    assert pk.parse_args(["all"]).flow == "all"


def test_frontend_invokes_pyinstaller_with_spec(tmp_path):
    calls = []
    pk.build_frontend(ROOT, out=tmp_path, runner=lambda *a, **k: calls.append(a))
    assert len(calls) == 1
    cmd = calls[0][0]
    assert "veluxia.spec" in str(cmd)


def test_backend_assembles_and_zips(tmp_path):
    pkg = pk.build_backend(ROOT, out=tmp_path)
    assert (pkg / "run_backend.bat").exists()
    assert (pkg / "app" / "models.json").exists()
    zips = list(tmp_path.glob("veluxia-backend*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as z:
        names = z.namelist()
    assert any("run_backend.bat" in n for n in names)
    assert any("models.json" in n for n in names)
