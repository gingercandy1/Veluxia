"""资料库服务与 /library 路由契约（ADR 0004）。

用临时模板目录 + 假 runner，无权重、无 GPU 也可跑。
"""
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.core import model_utils
from src.backend.core.collection import library, steps, template
from src.backend.core.collection.steps import StepContext, StepRunner
from src.backend.core.exceptions import GenerationCancelled
from src.backend.router.library import LibraryRouter


class _EchoRunner(StepRunner):
    """把条目提示词写成文本产物；gate 未放行时阻塞，用来模拟执行中的资源包。"""
    type_name = "test.echo"
    gate = threading.Event()
    started = threading.Event()

    def run(self, ctx: StepContext):
        type(self).started.set()
        while not type(self).gate.wait(0.02):
            if ctx.cancel_event.is_set():
                raise GenerationCancelled()
        path = ctx.out_dir / f"{ctx.step_id}.txt"
        path.write_text(ctx.prompt, encoding="utf-8")
        return [path]


class _DraftRunner(StepRunner):
    """可审阅修改的草稿步骤：修改内容为空时视为不合法。"""
    type_name = "test.draft"

    def run(self, ctx: StepContext):
        path = ctx.out_dir / f"{ctx.step_id}.txt"
        path.write_text(ctx.values["prompt"], encoding="utf-8")
        return [path]

    def validate_edit(self, content: str) -> str:
        if not content.strip():
            raise ValueError("内容不能为空")
        return content.strip()


class _InputsRunner(StepRunner):
    """把输入文件的路径写成产物，用来确认 "@source" 解析到了角色立绘。"""
    type_name = "test.inputs"

    def run(self, ctx: StepContext):
        path = ctx.out_dir / f"{ctx.step_id}.txt"
        path.write_text("\n".join(str(p) for p in ctx.inputs), encoding="utf-8")
        return [path]


@pytest.fixture
def client(tmp_path, monkeypatch):
    media = tmp_path / "media"
    templates = (tmp_path / "templates").resolve()
    templates.mkdir()
    (templates / "echo.json").write_text(json.dumps({
        "id": "echo", "version": 2, "type": "scene",
        "steps": [{"id": "write", "type": "test.echo"}],
    }), encoding="utf-8")
    (templates / "reviewed.json").write_text(json.dumps({
        "id": "reviewed", "type": "scene",
        "steps": [{"id": "draft", "type": "test.draft", "review": True},
                  {"id": "final", "type": "test.echo", "inputs": ["draft"]}],
    }), encoding="utf-8")
    (templates / "cast.json").write_text(json.dumps({
        "id": "cast", "type": "character",
        "fields": [{"id": "voice", "label": "音色"}],
        "steps": [{"id": "voice", "type": "test.echo"}],
    }), encoding="utf-8")
    (templates / "talk.json").write_text(json.dumps({
        "id": "talk", "type": "dialogue",
        "steps": [{"id": "write", "type": "test.echo"}],
    }), encoding="utf-8")
    (templates / "portrait.json").write_text(json.dumps({
        "id": "portrait", "type": "character",
        "steps": [{"id": library.CHARACTER_PORTRAIT_STEP, "type": "test.echo"}],
    }), encoding="utf-8")
    (templates / "motion.json").write_text(json.dumps({
        "id": "motion", "type": "character", "source": "character",
        "steps": [{"id": "video", "type": "test.inputs", "inputs": ["@source"]}],
    }), encoding="utf-8")
    monkeypatch.setattr(model_utils, "get_media_root", lambda: media)
    monkeypatch.setattr(library, "get_media_root", lambda: media)
    monkeypatch.setattr(template, "TEMPLATE_DIR", templates)
    steps.register_runner(_EchoRunner())
    steps.register_runner(_DraftRunner())
    steps.register_runner(_InputsRunner())
    _EchoRunner.gate.set()
    _EchoRunner.started.clear()

    app = FastAPI()
    app.include_router(LibraryRouter().router)
    with TestClient(app) as test_client:
        yield test_client
    _EchoRunner.gate.set()
    steps._RUNNERS.pop(_EchoRunner.type_name, None)
    steps._RUNNERS.pop(_DraftRunner.type_name, None)
    steps._RUNNERS.pop(_InputsRunner.type_name, None)


def _create(client, **overrides):
    body = {"name": "森林", "template": "echo", "style": {"prompt": "teal"},
            "items": [{"id": "mushroom", "prompt": "mushroom"}, {"prompt": "vine"}]}
    body.update(overrides)
    return client.post("/library/packs", json=body)


def _wait_job(client, job_id):
    for _ in range(200):
        status = client.get(f"/library/run/status/{job_id}").json()
        if status["status"] in ("done", "error", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("任务没有在预期时间内结束")


def test_create_list_get_delete(client):
    created = _create(client)
    assert created.status_code == 200
    manifest = created.json()["manifest"]
    assert manifest["template_version"] == 2 and manifest["type"] == "scene"
    # 没填 id 的条目按序号生成
    assert [item["id"] for item in manifest["items"]] == ["mushroom", "02"]
    pack_id = manifest["id"]
    assert created.json()["media_base"] == f"/media/library/{pack_id}"

    listed = client.get("/library/packs").json()["packs"]
    assert [p["manifest"]["id"] for p in listed] == [pack_id]
    assert client.get(f"/library/packs/{pack_id}").json()["manifest"]["name"] == "森林"

    assert client.delete(f"/library/packs/{pack_id}").json()["ok"] is True
    assert client.get(f"/library/packs/{pack_id}").status_code == 404
    assert client.delete(f"/library/packs/{pack_id}").status_code == 404


@pytest.mark.parametrize("overrides", [
    {"template": "missing"},
    {"template": "../echo"},
    {"items": []},
    {"items": [{"id": "../x", "prompt": "a"}]},
    {"items": [{"id": "a", "prompt": "1"}, {"id": "a", "prompt": "2"}]},
])
def test_create_rejects_invalid_requests(client, overrides):
    assert _create(client, **overrides).status_code == 400


def test_pack_id_cannot_escape_library(client):
    assert client.get("/library/packs/..%5Cconfig").status_code == 400
    with pytest.raises(ValueError):
        library.pack_dir("../config")


def test_broken_manifest_is_skipped_in_list(client):
    pack_id = _create(client).json()["manifest"]["id"]
    broken = library.library_root() / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{", encoding="utf-8")
    assert [p["manifest"]["id"] for p in client.get("/library/packs").json()["packs"]] == [pack_id]


def test_run_job_writes_outputs_into_manifest(client):
    pack_id = _create(client).json()["manifest"]["id"]
    job = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    status = _wait_job(client, job["job_id"])
    assert status["status"] == "done", status["error"]

    pack = client.get(f"/library/packs/{pack_id}").json()
    assert pack["running"] is False
    state = pack["manifest"]["items"][0]["steps"]["write"]
    assert state["status"] == "done" and state["error"] is None
    assert state["outputs"] == ["mushroom/write.txt"]
    assert "elapsed" in state["meta"]
    assert (library.pack_dir(pack_id) / "mushroom" / "write.txt").read_text(
        encoding="utf-8") == "mushroom, teal"


def test_running_pack_cannot_be_run_twice_or_deleted(client):
    pack_id = _create(client).json()["manifest"]["id"]
    _EchoRunner.gate.clear()
    first = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    assert _EchoRunner.started.wait(5)

    assert client.get(f"/library/packs/{pack_id}").json()["running"] is True
    second = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    rejected = _wait_job(client, second["job_id"])
    assert rejected["status"] == "error" and "已在执行" in rejected["error"]
    assert client.delete(f"/library/packs/{pack_id}").status_code == 409

    _EchoRunner.gate.set()
    assert _wait_job(client, first["job_id"])["status"] == "done"
    assert client.delete(f"/library/packs/{pack_id}").status_code == 200


def test_cancel_run_keeps_manifest_and_releases_pack(client):
    pack_id = _create(client).json()["manifest"]["id"]
    _EchoRunner.gate.clear()
    job = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    assert _EchoRunner.started.wait(5)

    assert client.post(f"/library/run/cancel/{job['job_id']}").json()["ok"] is True
    assert _wait_job(client, job["job_id"])["status"] == "cancelled"
    pack = client.get(f"/library/packs/{pack_id}").json()
    assert pack["running"] is False
    assert pack["manifest"]["items"][0]["steps"]["write"]["status"] == "pending"


def test_routers_without_factory_type_skip_models_route(client):
    assert client.get("/library/models").status_code == 404


def _run(client, pack_id):
    job = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    status = _wait_job(client, job["job_id"])
    assert status["status"] == "done", status["error"]
    return client.get(f"/library/packs/{pack_id}").json()["manifest"]


def _steps(manifest):
    return manifest["items"][0]["steps"]


def test_review_step_blocks_downstream_until_approved(client):
    pack_id = _create(client, template="reviewed", items=[{"id": "a", "prompt": "hello"}]
                      ).json()["manifest"]["id"]
    manifest = _run(client, pack_id)
    assert _steps(manifest)["draft"]["status"] == "done"
    assert _steps(manifest)["final"]["status"] == "pending"

    body = {"item_id": "a", "step_id": "draft"}
    approved = client.post(f"/library/packs/{pack_id}/approve", json=body)
    assert approved.status_code == 200
    assert _steps(approved.json()["manifest"])["draft"]["approved"] is True
    assert _steps(_run(client, pack_id))["final"]["status"] == "done"


def test_approve_with_edit_rewrites_output_and_resets_downstream(client):
    pack_id = _create(client, template="reviewed", items=[{"id": "a", "prompt": "hello"}]
                      ).json()["manifest"]["id"]
    _run(client, pack_id)
    client.post(f"/library/packs/{pack_id}/approve", json={"item_id": "a", "step_id": "draft"})
    _run(client, pack_id)

    body = {"item_id": "a", "step_id": "draft", "content": "  edited  "}
    manifest = client.post(f"/library/packs/{pack_id}/approve", json=body).json()["manifest"]
    assert _steps(manifest)["draft"]["meta"]["edited"] is True
    assert _steps(manifest)["final"]["status"] == "pending"
    assert (library.pack_dir(pack_id) / "a" / "draft.txt").read_text(encoding="utf-8") == "edited"


@pytest.mark.parametrize("body, status", [
    ({"item_id": "a", "step_id": "final"}, 400),       # 不需要审阅的步骤
    ({"item_id": "x", "step_id": "draft"}, 400),       # 条目不存在
    ({"item_id": "a", "step_id": "nope"}, 400),        # 步骤不存在
    ({"item_id": "a", "step_id": "draft", "content": " "}, 400),  # 修改内容不合法
])
def test_approve_rejects_invalid_requests(client, body, status):
    pack_id = _create(client, template="reviewed", items=[{"id": "a", "prompt": "hello"}]
                      ).json()["manifest"]["id"]
    _run(client, pack_id)
    assert client.post(f"/library/packs/{pack_id}/approve", json=body).status_code == status


def test_approve_requires_finished_step(client):
    pack_id = _create(client, template="reviewed", items=[{"id": "a", "prompt": "hello"}]
                      ).json()["manifest"]["id"]
    body = {"item_id": "a", "step_id": "draft"}
    assert client.post(f"/library/packs/{pack_id}/approve", json=body).status_code == 400
    assert client.post("/library/packs/missing/approve", json=body).status_code == 404


def test_reset_step_marks_it_and_downstream_pending(client):
    pack_id = _create(client, template="reviewed", items=[{"id": "a", "prompt": "hello"}]
                      ).json()["manifest"]["id"]
    _run(client, pack_id)
    client.post(f"/library/packs/{pack_id}/approve", json={"item_id": "a", "step_id": "draft"})
    _run(client, pack_id)

    reset = client.post(f"/library/packs/{pack_id}/reset", json={"item_id": "a", "step_id": "draft"})
    assert reset.status_code == 200
    states = _steps(reset.json()["manifest"])
    assert states["draft"]["status"] == "pending" and states["final"]["status"] == "pending"
    assert states["draft"]["approved"] is False


def test_item_fields_are_validated_and_stored(client):
    items = [{"id": "hero", "prompt": "knight", "fields": {"voice": "低沉"}}]
    manifest = _create(client, template="cast", items=items).json()["manifest"]
    assert manifest["items"][0]["fields"] == {"voice": "低沉"}
    unknown = [{"prompt": "knight", "fields": {"color": "red"}}]
    assert _create(client, template="cast", items=unknown).status_code == 400


def test_dialogue_cast_is_validated(client):
    character = _create(client, template="cast", items=[{"id": "hero", "prompt": "knight"}]
                        ).json()["manifest"]["id"]
    scene = _create(client).json()["manifest"]["id"]
    talk = {"template": "talk", "items": [{"prompt": "相遇"}]}

    ok = [{"name": "骑士", "character": f"{character}/hero"}, {"name": "路人", "description": "老人"}]
    assert _create(client, **talk, cast=ok).status_code == 200
    for cast in (
        [],                                                        # 对话包必须有角色
        [{"name": "骑士"}],                                        # 既没绑定也没描述
        [{"name": "甲", "description": "a"}, {"name": "甲", "description": "b"}],  # 重名
        [{"name": "骑士", "character": f"{character}/nobody"}],     # 条目不存在
        [{"name": "骑士", "character": f"{scene}/mushroom"}],       # 不是角色包
        [{"name": "骑士", "character": "missing/hero"}],            # 角色包不存在
    ):
        assert _create(client, **talk, cast=cast).status_code == 400, cast
    # 非对话包不能带出场角色
    assert _create(client, cast=ok).status_code == 400


def test_cast_resolves_character_voice_sample(client):
    character = _create(client, template="cast", items=[
        {"id": "hero", "prompt": "knight", "fields": {"voice": "低沉"}}]).json()["manifest"]["id"]
    member = library.CastMember(name="骑士", character=f"{character}/hero")

    # 还没生成声线时退回按描述设计
    voice = library._resolve_cast_member(member)
    assert voice.sample_audio is None and voice.voice_prompt == "低沉"
    assert voice.description == "knight"

    manifest = _run(client, character)
    directory = library.pack_dir(character)
    state = manifest["items"][0]["steps"]["voice"]
    # 假 runner 不写 meta.prompt，手动补上样本文本，模拟 TTS 步骤的产物
    stored = library.load_manifest(directory)
    stored.items[0].steps["voice"].meta["prompt"] = "你好"
    library.save_manifest(directory, stored)

    voice = library._resolve_cast_member(member)
    assert voice.sample_audio == directory / state["outputs"][0]
    assert voice.sample_text == "你好"


def test_source_character_is_validated_on_create(client):
    character = _create(client, template="portrait", items=[{"id": "hero", "prompt": "knight"}]
                        ).json()["manifest"]["id"]
    scene = _create(client).json()["manifest"]["id"]
    motion = {"template": "motion", "items": [{"id": "walk", "prompt": "walk"}]}

    created = _create(client, **motion, source=f"{character}/hero")
    assert created.status_code == 200
    assert created.json()["manifest"]["source"] == f"{character}/hero"
    motion_pack = created.json()["manifest"]["id"]
    for source in ("", f"{character}/nobody", f"{scene}/mushroom", "missing/hero",
                   f"{motion_pack}/walk"):  # 动作包本身不能再当角色引用
        assert _create(client, **motion, source=source).status_code == 400, source
    # 不需要来源的模板不能带 source
    assert _create(client, source=f"{character}/hero").status_code == 400


def test_source_portrait_is_resolved_at_run_time(client):
    character = _create(client, template="portrait", items=[{"id": "hero", "prompt": "knight"}]
                        ).json()["manifest"]["id"]
    motion = _create(client, template="motion", source=f"{character}/hero",
                     items=[{"id": "walk", "prompt": "walk"}]).json()["manifest"]["id"]

    # 角色立绘还没生成：直接报错提示，而不是静默停在待执行
    job = client.post("/library/run/submit", json={"pack_id": motion}).json()
    failed = _wait_job(client, job["job_id"])
    assert failed["status"] == "error" and "立绘还没有生成" in failed["error"]

    _run(client, character)
    manifest = _run(client, motion)
    written = (library.pack_dir(motion) / _steps(manifest)["video"]["outputs"][0]).read_text(
        encoding="utf-8")
    portrait = library.pack_dir(character) / "hero" / f"{library.CHARACTER_PORTRAIT_STEP}.txt"
    assert written == str(portrait)


def test_redone_source_portrait_is_noticed_and_refreshed(client):
    character = _create(client, template="portrait", items=[{"id": "hero", "prompt": "knight"}]
                        ).json()["manifest"]["id"]
    motion = _create(client, template="motion", source=f"{character}/hero",
                     items=[{"id": "walk", "prompt": "walk"}]).json()["manifest"]["id"]
    _run(client, character)
    manifest = _run(client, motion)
    assert _steps(manifest)["video"]["meta"]["source"]
    assert client.get(f"/library/packs/{motion}").json()["source_changed"] is False
    # 立绘没变时不需要重做
    assert client.post(f"/library/packs/{motion}/refresh_source").status_code == 400

    # 立绘重做：文件被重写，大小也不同
    portrait = library.pack_dir(character) / "hero" / f"{library.CHARACTER_PORTRAIT_STEP}.txt"
    portrait.write_text("knight, redone", encoding="utf-8")
    assert client.get(f"/library/packs/{motion}").json()["source_changed"] is True

    refreshed = client.post(f"/library/packs/{motion}/refresh_source").json()
    assert refreshed["source_changed"] is False
    assert _steps(refreshed["manifest"])["video"]["status"] == "pending"
    manifest = _run(client, motion)
    assert _steps(manifest)["video"]["status"] == "done"
    assert client.get(f"/library/packs/{motion}").json()["source_changed"] is False


def test_style_presets_crud(client):
    assert client.get("/library/styles").json()["styles"] == []
    styles = client.post("/library/styles", json={"name": " 手绘 ", "prompt": "hand-painted"}
                         ).json()["styles"]
    assert [(s["name"], s["prompt"]) for s in styles] == [("手绘", "hand-painted")]
    style_id = styles[0]["id"]
    assert (library.library_root() / library.STYLES_NAME).is_file()

    # 带 id 是修改；新建同名预设被拒绝
    updated = client.post("/library/styles", json={"id": style_id, "name": "手绘", "prompt": "ink"})
    assert updated.json()["styles"][0]["prompt"] == "ink"
    assert client.post("/library/styles", json={"name": "手绘"}).status_code == 400
    assert client.post("/library/styles", json={"name": "  "}).status_code == 400
    assert client.post("/library/styles", json={"id": "nope", "name": "x"}).status_code == 400

    assert client.delete(f"/library/styles/{style_id}").json()["styles"] == []
    assert client.delete(f"/library/styles/{style_id}").status_code == 404


def test_pack_records_preset_and_syncs_to_its_latest_content(client):
    style = client.post("/library/styles", json={"name": "手绘", "prompt": "hand-painted"}
                        ).json()["styles"][0]
    assert _create(client, style_preset="missing").status_code == 400
    created = _create(client, style={"prompt": "hand-painted"}, style_preset=style["id"])
    manifest = created.json()["manifest"]
    assert manifest["style_preset"] == style["id"]
    pack_id = manifest["id"]

    client.post("/library/styles", json={**style, "prompt": "watercolor", "negative": "3d"})
    # 改预设不会悄悄改掉已建好的包，要用户确认同步
    assert client.get(f"/library/packs/{pack_id}").json()["manifest"]["style"]["prompt"] == \
        "hand-painted"
    synced = client.post(f"/library/packs/{pack_id}/sync_style").json()["manifest"]
    assert synced["style"] == {"prompt": "watercolor", "negative": "3d"}

    plain = _create(client).json()["manifest"]["id"]
    assert client.post(f"/library/packs/{plain}/sync_style").status_code == 400
    client.delete(f"/library/styles/{style['id']}")
    assert client.post(f"/library/packs/{pack_id}/sync_style").status_code == 400
    assert client.get(f"/library/packs/{pack_id}").status_code == 200


def test_templates_route_exposes_fields_and_steps(client):
    templates = {t["id"]: t for t in client.get("/library/templates").json()["templates"]}
    assert templates["motion"]["source"] == "character"
    assert templates["cast"]["fields"][0]["id"] == "voice"
    assert templates["reviewed"]["steps"] == ["draft", "final"]
    assert templates["reviewed"]["step_details"][0]["review"] is True


def test_builtin_templates_load_and_use_registered_runners():
    runners = steps.registered_runners()
    loaded = template.list_templates()
    assert {t.type for t in loaded} >= {"scene", "character", "item", "effect", "dialogue",
                                        "audio"}
    for tpl in loaded:
        assert tpl.name, tpl.id
        missing = [s.type for s in tpl.steps if s.type not in runners]
        assert not missing, (tpl.id, missing)
        assert any(s.deliverable for s in tpl.steps), tpl.id
