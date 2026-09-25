"""资源包执行器契约（ADR 0004）：按步骤分组、断点续跑、单条失败隔离、取消、遇忙重试。

用假的 runner，无权重、无 GPU 也可跑。
"""
import threading
from contextlib import contextmanager

import pytest

from src.backend.core.collection.executor import CollectionExecutor, compose_prompt
from src.backend.core.collection.manifest import load_manifest, save_manifest
from src.backend.core.collection.steps import StepRunner
from src.backend.core.collection.template import load_template, parse_template
from src.backend.core.exceptions import GenerationCancelled, GeneratorBusyError
from src.shared.schemas import CollectionItem, CollectionStyle, Manifest

TEMPLATE = parse_template({
    "id": "fake",
    "steps": [
        {"id": "a", "type": "fake.a"},
        {"id": "b", "type": "fake.b", "inputs": ["a"]},
    ],
})


class _Recorder(StepRunner):
    """记录调用顺序，并把输入路径和提示词写进产物，方便断言。"""

    def __init__(self, type_name, log, fail_on=(), busy_times=0):
        self.type_name = type_name
        self.log = log
        self.fail_on = set(fail_on)
        self.busy_times = busy_times

    @contextmanager
    def _lease(self):
        self.log.append(("open", self.type_name))
        yield

    def open(self, params, cancel_event):
        if self.busy_times:
            self.busy_times -= 1
            raise GeneratorBusyError("正在运行 别的模型")
        return self._lease()

    def run(self, ctx):
        self.log.append((self.type_name, ctx.item.id))
        if ctx.item.id in self.fail_on:
            raise RuntimeError("boom")
        path = ctx.out_dir / f"{ctx.step_id}.txt"
        path.write_text(f"{ctx.prompt}|{[p.name for p in ctx.inputs]}", encoding="utf-8")
        return [path]


def _make_pack(tmp_path, item_ids=("x", "y", "z"), style=""):
    manifest = Manifest(
        id="pack", template="fake", style=CollectionStyle(prompt=style),
        items=[CollectionItem(id=i, prompt=f"prompt-{i}") for i in item_ids],
    )
    save_manifest(tmp_path, manifest)
    return tmp_path


def _executor(pack, log, **runner_options):
    runners = {
        "fake.a": _Recorder("fake.a", log, **runner_options.get("a", {})),
        "fake.b": _Recorder("fake.b", log, **runner_options.get("b", {})),
    }
    return CollectionExecutor(pack, TEMPLATE, runners=runners, busy_wait_seconds=0)


def _status(pack, item_id, step_id):
    item = next(i for i in load_manifest(pack).items if i.id == item_id)
    return item.steps[step_id].status


def test_steps_run_grouped_with_one_open_per_step(tmp_path):
    log = []
    _executor(_make_pack(tmp_path), log).run()
    assert log == [
        ("open", "fake.a"), ("fake.a", "x"), ("fake.a", "y"), ("fake.a", "z"),
        ("open", "fake.b"), ("fake.b", "x"), ("fake.b", "y"), ("fake.b", "z"),
    ]


def test_outputs_are_relative_and_feed_the_next_step(tmp_path):
    pack = _make_pack(tmp_path, item_ids=("x",), style="dark style")
    _executor(pack, []).run()
    item = load_manifest(pack).items[0]
    assert item.steps["a"].outputs == ["x/a.txt"]
    assert (pack / "x" / "b.txt").read_text(encoding="utf-8") == "prompt-x, dark style|['a.txt']"


def test_failed_item_is_isolated_and_skips_downstream(tmp_path):
    pack = _make_pack(tmp_path)
    log = []
    _executor(pack, log, a={"fail_on": {"y"}}).run()
    assert _status(pack, "y", "a") == "error"
    assert _status(pack, "y", "b") == "pending"
    assert _status(pack, "x", "b") == "done" and _status(pack, "z", "b") == "done"
    assert ("fake.b", "y") not in log
    item = next(i for i in load_manifest(pack).items if i.id == "y")
    assert item.steps["a"].error == "boom"


def test_rerun_skips_done_steps_and_retries_failures(tmp_path):
    pack = _make_pack(tmp_path)
    _executor(pack, [], a={"fail_on": {"y"}}).run()
    log = []
    _executor(pack, log).run()
    assert log == [("open", "fake.a"), ("fake.a", "y"), ("open", "fake.b"), ("fake.b", "y")]


def test_missing_output_reruns_the_step_and_its_downstream(tmp_path):
    pack = _make_pack(tmp_path, item_ids=("x",))
    _executor(pack, []).run()
    (pack / "x" / "a.txt").unlink()
    log = []
    _executor(pack, log).run()
    assert ("fake.a", "x") in log and ("fake.b", "x") in log


def test_cancel_keeps_finished_outputs(tmp_path):
    pack = _make_pack(tmp_path)
    cancel = threading.Event()

    class _CancelAfterFirst(_Recorder):
        def run(self, ctx):
            paths = super().run(ctx)
            cancel.set()
            return paths

    runners = {"fake.a": _CancelAfterFirst("fake.a", []), "fake.b": _Recorder("fake.b", [])}
    executor = CollectionExecutor(pack, TEMPLATE, runners=runners, cancel_event=cancel)
    with pytest.raises(GenerationCancelled):
        executor.run()
    assert _status(pack, "x", "a") == "done"
    assert (pack / "x" / "a.txt").exists()
    assert _status(pack, "y", "a") == "pending"


def test_busy_generator_is_retried_instead_of_failing(tmp_path):
    pack = _make_pack(tmp_path, item_ids=("x",))
    _executor(pack, [], a={"busy_times": 2}).run()
    assert _status(pack, "x", "b") == "done"


def test_busy_wait_stops_when_cancelled(tmp_path):
    pack = _make_pack(tmp_path, item_ids=("x",))
    cancel = threading.Event()
    runner = _Recorder("fake.a", [], busy_times=10**6)
    runner_open = runner.open

    def open_then_cancel(params, cancel_event):
        cancel.set()
        return runner_open(params, cancel_event)

    runner.open = open_then_cancel
    runners = {"fake.a": runner, "fake.b": _Recorder("fake.b", [])}
    executor = CollectionExecutor(pack, TEMPLATE, runners=runners, cancel_event=cancel,
                                  busy_wait_seconds=10)
    with pytest.raises(GenerationCancelled):
        executor.run()


def test_unknown_step_type_fails_before_running(tmp_path):
    with pytest.raises(ValueError, match="fake.b"):
        CollectionExecutor(_make_pack(tmp_path), TEMPLATE, runners={"fake.a": StepRunner()})


@pytest.mark.parametrize("steps, message", [
    ([], "没有任何步骤"),
    ([{"id": "a", "type": "t"}, {"id": "a", "type": "t"}], "重复"),
    ([{"id": "a", "type": "t", "inputs": ["b"]}, {"id": "b", "type": "t"}], "之前"),
])
def test_invalid_template_is_rejected(steps, message):
    with pytest.raises(ValueError, match=message):
        parse_template({"id": "bad", "steps": steps})


def test_downstream_is_transitive():
    template = parse_template({"id": "t", "steps": [
        {"id": "a", "type": "t"}, {"id": "b", "type": "t", "inputs": ["a"]},
        {"id": "c", "type": "t", "inputs": ["b"]}, {"id": "d", "type": "t"},
    ]})
    assert template.downstream_of("a") == ["b", "c"]


def test_builtin_scene_props_template_is_valid():
    template = load_template("scene_props")
    assert template.step_ids() == ["generate", "remove_bg", "upscale", "trim"]


def test_compose_prompt_skips_empty_parts():
    assert compose_prompt("mushroom", "") == "mushroom"
    assert compose_prompt(" ", "dark") == "dark"
