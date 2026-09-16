"""FILM/Rife 可实例化契约测试。

Seam：类定义本身（AST 静态）+ generate 分发逻辑。
背景：BaseImageFrameGenerator 要求 generate/_check_model_file；
缺失则 Factory.build_generator 直接 TypeError，路由补了也白搭。
"""
import ast
from pathlib import Path

FILM_SRC = Path(__file__).resolve().parents[1] / "src" / "backend" / "core" / "image_frame" / "film_generator.py"


def _methods() -> set:
    tree = ast.parse(FILM_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FILMInterpolationGenerator":
            return {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise AssertionError("FILMInterpolationGenerator 未找到")


def test_film_implements_abstracts():
    methods = _methods()
    assert "generate" in methods, "缺 generate：Factory 无法实例化"
    assert "_check_model_file" in methods, "缺 _check_model_file：抽象未实现"
