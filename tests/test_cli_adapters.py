"""Run with `python -m unittest discover -s tests -v`; no model downloads or CLI login needed."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import types
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explorer"))
from codex_adapter import CodexAdapter, cli_command


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.prompt = self.directory / "prompt.md"
        self.prompt.write_text("Explain in English. Tokens: 外星人.\nUse tools.", encoding="utf-8")
        self.cid = str(uuid.uuid4())
        self.adapter = CodexAdapter(self.directory, self.cid)

    def prepare(self, adapter=None, message="Explain this neuron", files=()):
        with patch("codex_adapter.cli_command", return_value=["codex"]):
            return (adapter or self.adapter).prepare(str(ROOT), str(self.directory), self.cid, 8792,
                self.prompt, message, message, files)

    def test_command_config_and_utf8_stdin(self):
        with patch.dict(os.environ, {"EXPLORER_CODEX_MODEL": "chosen-model"}):
            cmd, payload = self.prepare(message="Explain 外星人")
        overrides = [cmd[i + 1] for i, value in enumerate(cmd) if value == "-c"]
        config = tomllib.loads("\n".join(overrides))
        self.assertIn(self.prompt.read_text(encoding="utf-8"), config["developer_instructions"])
        self.assertIn("mcp__explorer__overview", config["developer_instructions"])
        self.assertEqual(config["mcp_servers"]["explorer"]["default_tools_approval_mode"], "approve")
        self.assertEqual(config["mcp_servers"]["explorer"]["env"]["EXPLORER_CHAT"], self.cid)
        self.assertEqual(config["mcp_servers"]["explorer"]["env"]["EXPLORER_URL"], "http://127.0.0.1:8792")
        self.assertEqual(config["sandbox_mode"], "workspace-write")
        self.assertEqual(cmd[-2:], ["--", "-"])
        self.assertIn("chosen-model", cmd)
        self.assertEqual(payload.decode("utf-8"), "Explain 外星人")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)

    def test_resume_uses_cli_thread_id_after_reload(self):
        self.prepare()
        self.adapter.started()
        sid = str(uuid.uuid4())
        list(self.adapter.events({"type": "thread.started", "thread_id": sid}))
        reloaded = CodexAdapter(self.directory, self.cid)
        cmd, _ = self.prepare(reloaded)
        self.assertEqual(cmd[cmd.index("resume") + 1], sid)
        self.assertNotEqual(sid, self.cid)

    def test_message_updates_are_not_duplicated(self):
        self.prepare()
        events = []
        for kind, text in [("item.updated", "Hello"), ("item.completed", "Hello world")]:
            events += list(self.adapter.events({"type": kind, "item": {"id": "one", "type": "agent_message", "text": text}}))
        self.assertEqual("".join(e["text"] for e in events), "Hello world")
        self.assertEqual(self.adapter.turn["reply"], "Hello world")

    def test_mcp_tools_and_errors(self):
        self.prepare()
        item = {"id": "call", "type": "mcp_tool_call", "tool": "neuron", "arguments": '{"layer": 12, "n": 42}'}
        start = list(self.adapter.events({"type": "item.started", "item": item}))
        self.assertEqual(start[0]["input"], {"layer": 12, "n": 42})
        item.update(status="failed", result={"content": [{"type": "text", "text": "no model"}], "isError": True})
        end = list(self.adapter.events({"type": "item.completed", "item": item}))
        self.assertEqual(len(end), 1)
        self.assertEqual(end[0]["output"], "no model")
        self.assertTrue(end[0]["error"])

    def test_completed_tool_without_start_and_output_limit(self):
        self.prepare()
        events = list(self.adapter.events({"type": "item.completed", "item": {
            "id": "cmd", "type": "command_execution", "command": "python probe.py", "aggregated_output": "x" * 2000,
            "exit_code": 1}}))
        self.assertEqual([e["t"] for e in events], ["tool", "tool_result"])
        self.assertTrue(events[1]["error"])
        self.assertIn("2000 chars", events[1]["output"])

    def test_failure_and_usage(self):
        self.assertEqual(list(self.adapter.events({"type": "turn.failed", "error": {"message": "sign in first"}})),
                         [{"t": "error", "error": "sign in first"}])
        self.assertEqual(list(self.adapter.events({"type": "turn.completed", "usage": {"output_tokens": 4}})),
                         [{"t": "done", "usage": {"output_tokens": 4}}])

    def test_attachments_are_paths_and_images(self):
        files = [{"name": "diagram.png", "type": "image/png", "path": str(self.directory / "a b.png")},
                 {"name": "notes.pdf", "type": "application/pdf", "path": str(self.directory / "notes.pdf")}]
        cmd, payload = self.prepare(files=files)
        self.assertEqual(cmd[cmd.index("--image") + 1], files[0]["path"])
        self.assertIn("notes.pdf", payload.decode("utf-8"))
        self.assertEqual(cmd.count("--image"), 1)

    def test_rewind_rebuilds_only_retained_turns(self):
        self.prepare(message="first")
        self.adapter.started()
        list(self.adapter.events({"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "first answer"}}))
        self.adapter.save()
        second = CodexAdapter(self.directory, self.cid)
        self.prepare(second, message="discard me")
        second.started()
        second.state["session"] = str(uuid.uuid4())
        second.save()
        self.assertTrue(second.rewind("discard me"))
        cmd, payload = self.prepare(CodexAdapter(self.directory, self.cid), message="replacement")
        self.assertNotIn("resume", cmd)
        text = payload.decode("utf-8")
        self.assertIn("first answer", text)
        self.assertIn("replacement", text)
        self.assertNotIn("discard me", text)

    def test_first_turn_rewind_and_no_match(self):
        self.prepare(message="only")
        self.adapter.started()
        self.assertFalse(self.adapter.rewind("missing"))
        self.assertTrue(self.adapter.rewind("only"))
        self.assertEqual(self.adapter.state, {"session": None, "turns": []})

    def test_invalid_chat_and_missing_cli(self):
        with self.assertRaises(ValueError):
            CodexAdapter(self.directory, "../../escape")
        with patch("codex_adapter.shutil.which", return_value=None), patch("codex_adapter._windows_codex", return_value=None), patch.dict(os.environ, {"EXPLORER_CODEX_BIN": ""}):
            with self.assertRaisesRegex(FileNotFoundError, "PATH"):
                cli_command("codex")

    def test_explicit_cli_path_takes_precedence_over_path(self):
        executable = self.directory / "my codex.exe"
        executable.touch()
        with patch.dict(os.environ, {"EXPLORER_CODEX_BIN": str(executable)}), patch("codex_adapter.shutil.which", return_value=None):
            self.assertEqual(cli_command("codex"), [str(executable.resolve())])

    def test_invalid_explicit_path_reports_setting(self):
        with patch.dict(os.environ, {"EXPLORER_CODEX_BIN": str(self.directory / "missing.exe")}), patch("codex_adapter.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "EXPLORER_CODEX_BIN"):
                cli_command("codex")

    def test_desktop_cli_is_found_without_path(self):
        binary = self.directory / "OpenAI/Codex/bin/version/codex.exe"
        binary.parent.mkdir(parents=True)
        binary.touch()
        with patch.dict(os.environ, {"LOCALAPPDATA": str(self.directory), "EXPLORER_CODEX_BIN": ""}), patch("codex_adapter.sys.platform", "win32"), patch("codex_adapter.shutil.which", return_value=None):
            self.assertEqual(cli_command("codex"), [str(binary)])

    def test_path_cli_is_preferred_to_desktop_fallback(self):
        with patch.dict(os.environ, {"EXPLORER_CODEX_BIN": ""}), patch("codex_adapter.shutil.which", return_value="chosen-codex.exe"), patch("codex_adapter._windows_codex") as desktop:
            self.assertEqual(cli_command("codex"), ["chosen-codex.exe"])
            desktop.assert_not_called()


# Import the API without loading Qwen, transformers or downloading a model configuration.
cfg = types.SimpleNamespace(num_hidden_layers=32, hidden_size=2560, intermediate_size=9216,
    layer_types=["linear_attention"] * 32, num_attention_heads=16, num_key_value_heads=4,
    linear_num_key_heads=16, linear_num_value_heads=32)
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *_: types.SimpleNamespace(get_text_config=lambda: cfg))
runtime_stub = types.ModuleType("runtime")
runtime_stub.NO_MODEL = True
runtime_stub.Runtime = lambda *_: types.SimpleNamespace(status="no model")
previous_modules = {name: sys.modules.get(name) for name in ("transformers", "runtime")}
sys.modules.update({"transformers": transformers_stub, "runtime": runtime_stub})
import server
for name, previous in previous_modules.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous
import httpx

FAKE_CLI = r'''
import json, sys, uuid
provider = sys.argv[1]
payload = sys.stdin.read()
def emit(value):
    print(json.dumps(value), flush=True)
if provider == "claude":
    message = json.loads(payload)
    assert message["message"]["content"][0]["type"] == "text"
    emit({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Claude reply"}}})
    emit({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "c1", "name": "mcp__explorer__overview", "input": {}}]}})
    emit({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "c1", "content": [{"type": "text", "text": "{}"}]}]}})
    emit({"type": "result", "total_cost_usd": 0})
else:
    if "stderr" in payload:
        sys.stderr.write("diagnostic " * 100000)
        sys.stderr.flush()
    sid = sys.argv[sys.argv.index("resume") + 1] if "resume" in sys.argv else str(uuid.uuid4())
    emit({"type": "thread.started", "thread_id": sid})
    if "failure" in payload:
        emit({"type": "turn.failed", "error": {"message": "Please sign in"}})
        sys.exit(1)
    emit({"type": "item.started", "item": {"id": "c1", "type": "mcp_tool_call", "tool": "overview", "arguments": {}}})
    if "pause" in payload:
        import time
        time.sleep(60)
    emit({"type": "item.completed", "item": {"id": "c1", "type": "mcp_tool_call", "tool": "overview", "result": {"content": [{"type": "text", "text": "{}"}]}}})
    emit({"type": "item.completed", "item": {"id": "a1", "type": "agent_message", "text": "Codex reply in English"}})
    emit({"type": "turn.completed", "usage": {"output_tokens": 4}})
'''


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        fake = self.directory / "fake_cli.py"
        fake.write_text(FAKE_CLI, encoding="utf-8")
        server._chats.clear()
        for name in ("AGENT_DIR", "UPLOADS"):
            p = patch.object(server, name, str(self.directory / name))
            p.start()
            self.addCleanup(p.stop)
        for module in ("server", "codex_adapter"):
            p = patch(f"{module}.cli_command", side_effect=lambda name: [sys.executable, str(fake), name])
            p.start()
            self.addCleanup(p.stop)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://127.0.0.1:8791")
        self.addAsyncCleanup(self.client.aclose)

    async def chat(self, **body):
        response = await self.client.post("/api/agent/chat", json={"message": "hello", **body})
        self.assertEqual(response.status_code, 200, response.text)
        return [json.loads(line) for line in response.text.splitlines()]

    async def test_codex_stream_restart_and_rewind(self):
        events = await self.chat(provider="codex")
        self.assertEqual([e["t"] for e in events], ["chat", "tool", "tool_result", "text", "done"])
        cid = events[0]["chat"]
        state = CodexAdapter(server.AGENT_DIR, cid).state
        self.assertEqual(state["turns"][0]["reply"], "Codex reply in English")
        server._chats.clear()  # server restart: thread ID must come from disk
        await self.chat(provider="codex", chat=cid, message="follow up")
        self.assertEqual(CodexAdapter(server.AGENT_DIR, cid).state["session"], state["session"])
        result = await self.client.post("/api/agent/rewind", json={"provider": "codex", "chat": cid, "text": "follow up"})
        self.assertTrue(result.json()["cut"])
        self.assertEqual(len(CodexAdapter(server.AGENT_DIR, cid).state["turns"]), 1)

    async def test_claude_remains_the_default(self):
        events = await self.chat()
        self.assertEqual(events[0]["provider"], "claude")
        self.assertEqual([e["t"] for e in events], ["chat", "text", "tool", "tool_result", "done"])
        self.assertEqual(events[2]["name"], "overview")

    async def test_cli_failures_are_visible(self):
        events = await self.chat(provider="codex", message="failure")
        self.assertIn("Please sign in", str(events))
        with patch("codex_adapter.cli_command", side_effect=FileNotFoundError("codex CLI not found")):
            events = await self.chat(provider="codex")
        self.assertEqual(events[-1]["t"], "error")
        self.assertIn("not found", events[-1]["error"])

    async def test_stderr_is_drained_while_stdout_streams(self):
        events = await asyncio.wait_for(self.chat(provider="codex", message="stderr"), timeout=10)
        self.assertEqual(events[-1]["t"], "done")

    async def test_invalid_provider_id_and_provider_mismatch(self):
        for body in ({"provider": "other"}, {"provider": "codex", "chat": "../../escape"}):
            response = await self.client.post("/api/agent/chat", json={"message": "hello", **body})
            self.assertEqual(response.status_code, 400)
        events = await self.chat(provider="codex")
        response = await self.client.post("/api/agent/chat", json={"message": "hello", "provider": "claude", "chat": events[0]["chat"]})
        self.assertEqual(response.status_code, 409)

    async def test_stop_running_process(self):
        options = {"start_new_session": True} if os.name != "nt" else {}
        proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)", **options)
        await asyncio.wait_for(server._stop_turn(proc), timeout=10)
        self.assertIsNotNone(proc.returncode)

    async def test_disconnect_reaps_cli_and_saves_before_unlocking(self):
        class Request:
            async def json(self):
                return {"provider": "codex", "message": "pause"}
        response = await server.agent_chat(Request())
        events = response.body_iterator
        cid = json.loads(await anext(events))["chat"]
        self.assertEqual(json.loads(await anext(events))["t"], "tool")
        chat = server._chats[cid]
        self.assertIsNone(chat["proc"].returncode)
        await asyncio.wait_for(events.aclose(), timeout=10)
        self.assertIsNotNone(chat["proc"].returncode)
        self.assertFalse(chat["lock"].locked())
        self.assertEqual(len(CodexAdapter(server.AGENT_DIR, cid).state["turns"]), 1)


if __name__ == "__main__":
    unittest.main()
