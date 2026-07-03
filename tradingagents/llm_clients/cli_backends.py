"""Subprocess backends for the subscription-CLI providers (Codex CLI / Claude Code).

These runners let TradingAgents use LLMs through locally installed,
subscription-authenticated CLIs (``codex`` with a ChatGPT plan, ``claude``
with a Claude plan) instead of HTTP APIs that require an API key.

Each runner exposes the same two calls:

- ``start(prompt, system)``  — open a fresh conversation, return the reply
  and a session/conversation id (when the CLI surfaces one).
- ``resume(session_id, prompt)`` — continue an existing conversation with an
  incremental prompt. This keeps the provider-side prompt cache warm across
  the multi-turn tool-calling loops the analysts run, instead of re-sending
  the full transcript in a new session every turn.

Session continuation is strictly an optimization: callers must treat any
failure here as "start a fresh conversation with the full transcript".
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A conversation is one prompt -> one reply; deep-thinking models on large
# transcripts can legitimately take minutes.
DEFAULT_TIMEOUT = 600.0

# "default" (or empty) means: do not pass a model flag and let the CLI use
# whatever the user's subscription/config defaults to.
DEFAULT_MODEL_SENTINEL = "default"


class CLIBackendError(RuntimeError):
    """A CLI subprocess call failed (non-zero exit, timeout, bad output)."""


@dataclass
class CLIResult:
    text: str
    session_id: str | None = None


def _is_default_model(model: str | None) -> bool:
    return not model or model.lower() == DEFAULT_MODEL_SENTINEL


def _scratch_cwd() -> str:
    """A stable empty working directory so the CLIs don't ingest repo context."""
    path = os.path.join(tempfile.gettempdir(), "tradingagents-cli-llm")
    os.makedirs(path, exist_ok=True)
    return path


def _login_hint(binary: str) -> str:
    if "claude" in os.path.basename(binary):
        return (
            " If you are not logged in, run `claude` once interactively to "
            "sign in and retry."
        )
    return f" If you are not logged in, run `{binary} login` and retry."


def _extract_cli_error(stdout: str, stderr: str) -> str:
    """Prefer the CLI's own error message over a raw output dump.

    claude -p prints a JSON envelope (with a human-readable ``result``) even
    on failure exits; fall back to raw stderr/stdout otherwise.
    """
    for line in (stdout or "").strip().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(envelope, dict) and envelope.get("result"):
            return str(envelope["result"])
    return (stderr or stdout or "").strip()[:2000]


def _run_subprocess(
    cmd: list[str], stdin_text: str, timeout: float, cwd: str | None = None
) -> subprocess.CompletedProcess:
    """Run a one-shot CLI call; retry once on non-zero exit."""
    last_error = ""
    for attempt in (1, 2):
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            raise CLIBackendError(
                f"CLI binary not found: {cmd[0]!r}. Install it and make sure "
                f"it is on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CLIBackendError(
                f"{cmd[0]} timed out after {timeout:.0f}s"
            ) from exc
        if proc.returncode == 0:
            return proc
        last_error = _extract_cli_error(proc.stdout, proc.stderr)
        logger.warning(
            "%s exited with code %s (attempt %s/2): %s",
            cmd[0], proc.returncode, attempt, last_error,
        )
    raise CLIBackendError(
        f"{cmd[0]} failed with exit code {proc.returncode}: {last_error}."
        + _login_hint(cmd[0])
    )


class ClaudeRunner:
    """One-shot ``claude -p`` calls with ``--resume`` session continuation.

    Measured process overhead is ~1.5s per call, so a persistent process is
    not worth the complexity; ``--resume`` alone keeps the Anthropic prompt
    cache warm across turns (verified: resumed calls report
    ``cache_read_input_tokens`` > 0).
    """

    binary = "claude"

    def __init__(
        self,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        effort: str | None = None,
    ):
        self.model = model
        self.timeout = timeout
        # Claude Code thinking depth: low, medium, high, xhigh, max.
        self.effort = effort

    def _base_cmd(self) -> list[str]:
        # --tools "" disables all of Claude Code's built-in tools: the model
        # must answer in text only, with tool use expressed through the
        # prompt-level protocol the chat model injects.
        cmd = [self.binary, "-p", "--output-format", "json", "--tools", ""]
        if not _is_default_model(self.model):
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    def _parse(self, stdout: str) -> CLIResult:
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise CLIBackendError(
                f"claude -p returned non-JSON output: {stdout[:500]!r}"
            ) from exc
        if envelope.get("is_error"):
            raise CLIBackendError(
                f"claude -p reported an error: {envelope.get('result')!r}"
                + _login_hint(self.binary)
            )
        return CLIResult(
            text=envelope.get("result") or "",
            session_id=envelope.get("session_id"),
        )

    def start(self, prompt: str, system: str | None = None) -> CLIResult:
        cmd = self._base_cmd()
        if system:
            cmd += ["--system-prompt", system]
        proc = _run_subprocess(cmd, prompt, self.timeout, cwd=_scratch_cwd())
        return self._parse(proc.stdout)

    def resume(self, session_id: str, prompt: str) -> CLIResult:
        # NOTE: resume relies on session persistence, so --no-session-persistence
        # must not be added to _base_cmd. Session files land in ~/.claude/projects.
        cmd = self._base_cmd() + ["--resume", session_id]
        proc = _run_subprocess(cmd, prompt, self.timeout, cwd=_scratch_cwd())
        result = self._parse(proc.stdout)
        # Resuming may mint a new session id; chain from whatever came back.
        return CLIResult(result.text, result.session_id or session_id)

    def close(self) -> None:
        pass


class CodexExecRunner:
    """One-shot ``codex exec`` calls with ``codex exec resume`` continuation.

    Fallback path for when the persistent MCP server mode is disabled or
    unavailable. Each call pays ~2s process startup plus the prefill of
    Codex's built-in agent instructions (~18k tokens), so prefer
    CodexMCPRunner.
    """

    binary = "codex"

    def __init__(
        self,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def _common_flags(self) -> list[str]:
        flags = [
            "-s", "read-only",
            "--skip-git-repo-check",
            "--color", "never",
            "--json",
        ]
        if not _is_default_model(self.model):
            flags += ["-m", self.model]
        if self.reasoning_effort:
            flags += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        return flags

    @staticmethod
    def _extract_session_id(jsonl: str) -> str | None:
        """Scan --json event lines for a session/thread identifier."""
        for line in jsonl.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            found = _find_first_key(event, ("session_id", "thread_id", "conversation_id"))
            if found:
                return str(found)
        return None

    def _run(self, argv: list[str], prompt: str) -> CLIResult:
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".txt", prefix="codex-last-", delete=False
        ) as out_file:
            out_path = out_file.name
        try:
            cmd = argv + ["--output-last-message", out_path, "-"]
            proc = _run_subprocess(cmd, prompt, self.timeout, cwd=_scratch_cwd())
            with open(out_path, encoding="utf-8") as fh:
                text = fh.read().strip()
            if not text:
                raise CLIBackendError(
                    f"codex exec produced no final message; stderr: "
                    f"{(proc.stderr or '').strip()[:500]}"
                )
            return CLIResult(text, self._extract_session_id(proc.stdout))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(out_path)

    def start(self, prompt: str, system: str | None = None) -> CLIResult:
        # codex exec has no system-prompt flag; fold it into the prompt.
        if system:
            prompt = f"## System\n{system}\n\n{prompt}"
        return self._run([self.binary, "exec", *self._common_flags()], prompt)

    def resume(self, session_id: str, prompt: str) -> CLIResult:
        result = self._run(
            [self.binary, "exec", "resume", session_id, *self._common_flags()],
            prompt,
        )
        return CLIResult(result.text, result.session_id or session_id)

    def close(self) -> None:
        pass


def _find_first_key(obj, keys: tuple[str, ...]):
    """Depth-first search for the first matching key in nested JSON data."""
    if isinstance(obj, dict):
        for key in keys:
            if obj.get(key):
                return obj[key]
        for value in obj.values():
            found = _find_first_key(value, keys)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_key(item, keys)
            if found:
                return found
    return None


# Without this, Codex ships ~18k tokens of coding-agent instructions on every
# conversation. base-instructions replaces them wholesale; the chat model's
# own system/protocol text arrives via the prompt.
_CODEX_BASE_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's request directly as text. "
    "Do not run shell commands, do not read or write files, and do not use "
    "any built-in tools: the only tools you may reference are the ones "
    "explicitly declared inside the conversation, and those are invoked by "
    "replying with the JSON format the conversation describes."
)


class CodexMCPRunner:
    """Persistent ``codex mcp-server`` process (JSON-RPC over stdio).

    One long-lived server handles every call: ``start`` invokes the ``codex``
    tool (fresh conversation each time, custom base-instructions replacing
    the ~18k-token built-in agent prompt), ``resume`` invokes ``codex-reply``
    with the conversationId so the provider-side prompt cache stays warm.

    The server is model-agnostic — the model is a per-call argument — so a
    single module-level instance is shared by the deep- and quick-thinking
    chat models.
    """

    binary = "codex"
    _PROTOCOL_VERSION = "2025-06-18"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        self._lock = threading.Lock()
        atexit.register(self.close)

    # -- process management -------------------------------------------------

    def _spawn(self) -> None:
        self.close()
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(
            [self.binary, "mcp-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=_scratch_cwd(),
        )
        reader = threading.Thread(
            target=self._read_stdout, args=(self._proc,), daemon=True
        )
        reader.start()
        self._handshake()

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self._lines.put(line)
        self._lines.put(None)  # EOF marker

    def _ensure_proc(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # -- JSON-RPC ------------------------------------------------------------

    def _send(self, payload: dict) -> None:
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CLIBackendError("codex mcp-server pipe is broken") from exc

    def _wait_for(self, request_id: int, timeout: float) -> dict:
        """Read lines until the response with ``request_id`` arrives."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CLIBackendError(
                    f"codex mcp-server timed out after {timeout:.0f}s"
                )
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                raise CLIBackendError(
                    "codex mcp-server exited unexpectedly" + _login_hint(self.binary)
                )
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == request_id and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise CLIBackendError(
                        f"codex mcp-server error: {msg['error']}"
                    )
                return msg["result"]
            # Notifications (codex/event streams etc.) are skipped; the
            # conversationId is read from the tools/call result instead.

    def _rpc(self, method: str, params: dict, timeout: float | None = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._send({
            "jsonrpc": "2.0", "id": request_id,
            "method": method, "params": params,
        })
        return self._wait_for(request_id, timeout or self.timeout)

    def _handshake(self) -> None:
        self._rpc(
            "initialize",
            {
                "protocolVersion": self._PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "tradingagents", "version": "1.0"},
            },
            timeout=30.0,
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    # -- runner interface ----------------------------------------------------

    @staticmethod
    def _result_text(result: dict) -> str:
        parts = [
            block.get("text", "")
            for block in result.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            structured = result.get("structuredContent")
            if isinstance(structured, dict):
                text = str(
                    _find_first_key(
                        structured, ("lastAgentMessage", "last_agent_message")
                    )
                    or ""
                ).strip()
        return text

    def _call_tool(self, name: str, arguments: dict) -> CLIResult:
        with self._lock:
            for attempt in (1, 2):
                self._ensure_proc()
                try:
                    result = self._rpc(
                        "tools/call", {"name": name, "arguments": arguments}
                    )
                    break
                except CLIBackendError:
                    self.close()
                    if attempt == 2:
                        raise
                    logger.warning(
                        "codex mcp-server call failed; restarting server and retrying"
                    )
        text = self._result_text(result)
        if not text:
            raise CLIBackendError(
                f"codex mcp-server returned an empty reply: {str(result)[:500]}"
            )
        session_id = _find_first_key(
            result, ("conversationId", "conversation_id", "threadId", "session_id")
        )
        return CLIResult(text, str(session_id) if session_id else None)

    def start(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> CLIResult:
        base = _CODEX_BASE_INSTRUCTIONS
        if system:
            base = f"{base}\n\n{system}"
        arguments = {
            "prompt": prompt,
            "sandbox": "read-only",
            "approval-policy": "never",
            "cwd": _scratch_cwd(),
            "base-instructions": base,
        }
        if not _is_default_model(model):
            arguments["model"] = model
        if reasoning_effort:
            arguments["config"] = {"model_reasoning_effort": reasoning_effort}
        return self._call_tool("codex", arguments)

    def resume(self, session_id: str, prompt: str) -> CLIResult:
        result = self._call_tool(
            "codex-reply", {"conversationId": session_id, "prompt": prompt}
        )
        return CLIResult(result.text, result.session_id or session_id)


class CodexMCPModelRunner:
    """Per-model facade over the shared CodexMCPRunner.

    Gives the persistent backend the same ``start(prompt, system)`` /
    ``resume(session_id, prompt)`` surface as the one-shot runners, with the
    model and reasoning effort bound at construction time.
    """

    binary = "codex"

    def __init__(
        self,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._shared = get_shared_codex_mcp_runner(timeout)

    def start(self, prompt: str, system: str | None = None) -> CLIResult:
        return self._shared.start(
            prompt, system, model=self.model, reasoning_effort=self.reasoning_effort
        )

    def resume(self, session_id: str, prompt: str) -> CLIResult:
        return self._shared.resume(session_id, prompt)

    def close(self) -> None:
        # The shared server outlives individual chat models; atexit owns it.
        pass


_shared_codex_mcp: CodexMCPRunner | None = None
_shared_codex_mcp_lock = threading.Lock()


def get_shared_codex_mcp_runner(timeout: float = DEFAULT_TIMEOUT) -> CodexMCPRunner:
    """Shared server for all chat-model instances (model is a per-call arg)."""
    global _shared_codex_mcp
    with _shared_codex_mcp_lock:
        if _shared_codex_mcp is None:
            _shared_codex_mcp = CodexMCPRunner(timeout=timeout)
        else:
            _shared_codex_mcp.timeout = max(_shared_codex_mcp.timeout, timeout)
        return _shared_codex_mcp


def cli_binary_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def new_call_id() -> str:
    """Tool-call ids for prompt-emulated tool calls."""
    return f"call_{uuid.uuid4().hex[:12]}"
