"""Codex CLI turns and JSONL events, adapted to the explorer's chat protocol.

Uses saved CLI login and per-invocation MCP overrides; never edits the user's Codex config.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


def _windows_codex():
    """Find a per-user Codex installation even when the server inherited an older PATH."""
    local = os.environ.get("LOCALAPPDATA")
    if not local or sys.platform != "win32":
        return None
    local = Path(local)
    for path in (local / "Programs/OpenAI/Codex/bin/codex.exe", local / "OpenAI/Codex/bin/codex.exe"):
        if path.is_file():
            return str(path)
    binaries = list((local / "OpenAI/Codex/bin").glob("*/codex.exe"))
    if binaries:
        return str(max(binaries, key=lambda p: p.stat().st_mtime_ns))
    return None


def cli_command(name):
    """Resolve native CLIs, including npm's .cmd launcher on Windows."""
    variable = f"EXPLORER_{name.upper()}_BIN"
    configured = os.environ.get(variable)
    if configured:
        configured = os.path.expandvars(os.path.expanduser(configured))
        executable = shutil.which(configured)
        if not executable and Path(configured).is_file():
            executable = str(Path(configured).resolve())
        if not executable:
            raise FileNotFoundError(f"{variable} points to a missing CLI executable: {configured}")
    else:
        executable = shutil.which(name)
        if not executable and name == "codex":
            executable = _windows_codex()
    if not executable:
        raise FileNotFoundError(f"{name} CLI was not found on PATH. Install it or set {variable} to its executable path, then restart the explorer.")
    if os.name == "nt" and executable.lower().endswith((".cmd", ".bat")):
        # Run npm's JS entry point directly: shell=True would interpret prompt/config arguments.
        entry = Path(executable).parent / "node_modules" / ("@openai/codex/bin/codex.js" if name == "codex"
                                                        else "@anthropic-ai/claude-code/cli.js")
        node = shutil.which("node")
        if node and entry.is_file():
            return [node, str(entry)]
        raise OSError(f"Cannot launch {executable} without a shell. Install a native {name} CLI or its npm package with Node.js.")
    return [executable]


class CodexAdapter:
    def __init__(self, directory, chat):
        # Browser chat IDs and CLI thread IDs are separate; only CLI-issued IDs can be resumed.
        self.path = Path(directory) / "codex" / f"{uuid.UUID(chat)}.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"session": None, "turns": []}
        self.turn = None
        self.sent_text = {}
        self.tools = set()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)

    def prepare(self, root, directory, chat, port, prompt_file, message, user_text, files):
        prompt = message
        images = []
        for file in files:
            path = file["path"]
            if file["type"] in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                images.append(path)
            prompt += f"\n\nAttached file {json.dumps(file['name'])}: {json.dumps(path)}. Read it with your available tools if needed."
        self.turn = {"user_text": user_text, "prompt": prompt, "images": images, "reply": ""}
        if not self.state["session"] and self.state["turns"]:
            # Edit/reroll starts a clean CLI thread, replaying only the retained visible conversation.
            previous = [{"user": t["prompt"], "assistant": t["reply"]} for t in self.state["turns"]]
            prompt = "Earlier conversation (context only):\n" + json.dumps(previous, ensure_ascii=False) + "\n\nCurrent user message:\n" + prompt
            images = [p for t in self.state["turns"] for p in t.get("images", [])] + images
        cmd = cli_command("codex") + ["exec", "--json", "--skip-git-repo-check"]
        # Keep shell writes inside the agent directory, while permitting the explorer's loopback API.
        config = {
            "developer_instructions": ("Your neuron tools are provided by the MCP server named explorer. "
                "Use its namespaced tools, such as mcp__explorer__overview and mcp__explorer__neuron, for explorer data.\n\n"
                + Path(prompt_file).read_text(encoding="utf-8")),
            "approval_policy": "never",
            "sandbox_mode": "workspace-write",
            "sandbox_workspace_write.network_access": True,
            "mcp_servers.explorer.command": sys.executable,
            "mcp_servers.explorer.args": [str(Path(root) / "explorer" / "agent_mcp.py")],
            "mcp_servers.explorer.env.EXPLORER_URL": f"http://127.0.0.1:{port}",
            "mcp_servers.explorer.env.EXPLORER_CHAT": chat,
            "mcp_servers.explorer.required": True,
            "mcp_servers.explorer.default_tools_approval_mode": "approve",
            "mcp_servers.explorer.tool_timeout_sec": 900,
        }
        for key, value in config.items():
            cmd += ["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"]
        if os.environ.get("EXPLORER_CODEX_MODEL"):
            cmd += ["--model", os.environ["EXPLORER_CODEX_MODEL"]]
        if self.state["session"]:
            cmd += ["resume", self.state["session"]]
        for path in dict.fromkeys(images):
            cmd += ["--image", path]
        return cmd + ["--", "-"], prompt.encode("utf-8")

    def started(self):
        self.state["turns"].append(self.turn)
        self.save()

    def events(self, message):
        kind = message.get("type")
        if kind == "thread.started":
            self.state["session"] = str(uuid.UUID(message["thread_id"]))
            self.save()
        elif kind in ("error", "turn.failed"):
            error = message.get("error") or message.get("message") or "Codex turn failed"
            yield {"t": "error", "error": error.get("message", str(error)) if isinstance(error, dict) else str(error)}
        elif kind == "turn.completed":
            yield {"t": "done", "usage": message.get("usage")}
        elif kind in ("item.started", "item.updated", "item.completed"):
            item = message.get("item") or {}
            iid, itype = item.get("id", ""), item.get("type")
            if itype == "agent_message":
                text = item.get("text", "")
                previous = self.sent_text.get(iid, "")
                delta = text[len(previous):] if text.startswith(previous) else text
                self.sent_text[iid] = text
                if delta:
                    if not previous and self.turn["reply"]:
                        delta = "\n\n" + delta
                    self.turn["reply"] += delta
                    yield {"t": "text", "text": delta}
            elif itype in ("mcp_tool_call", "command_execution", "file_change", "web_search"):
                if iid not in self.tools:
                    self.tools.add(iid)
                    args = item.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {"arguments": args}
                    if itype == "command_execution":
                        args = {"command": item.get("command", "")}
                    elif itype == "file_change":
                        args = {"changes": item.get("changes", [])}
                    elif itype == "web_search":
                        args = {"query": item.get("query", "")}
                    yield {"t": "tool", "id": iid, "name": item.get("tool") or itype, "input": args}
                if kind == "item.completed":
                    result = item.get("result")
                    if isinstance(result, dict) and "content" in result:
                        output = "\n".join(x.get("text", "") for x in result["content"] if isinstance(x, dict))
                    else:
                        output = item.get("aggregated_output") or json.dumps(result if result is not None else item, ensure_ascii=False)
                    yield {"t": "tool_result", "id": iid,
                           "error": bool(item.get("error") or item.get("status") == "failed" or item.get("exit_code") not in (None, 0)
                                         or isinstance(result, dict) and result.get("isError")),
                           "output": output[:1500] + (f" … ({len(output)} chars)" if len(output) > 1500 else "")}

    def rewind(self, text):
        for i in range(len(self.state["turns"]) - 1, -1, -1):
            if not text or self.state["turns"][i]["user_text"].strip() == text.strip():
                self.state["turns"] = self.state["turns"][:i]
                self.state["session"] = None
                self.save()
                return True
        return False
