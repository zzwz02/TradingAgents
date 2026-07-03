"""Tests for the subscription-CLI providers (codex-cli / claude-code).

Everything here is offline: CLI subprocesses are replaced with fakes, so the
tests exercise factory dispatch, message rendering, prompt-emulated tool-call
parsing, structured output, the session-resume registry, and the MCP stdio
client against a scripted stand-in server.
"""

import json
import stat
import sys
import textwrap

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

from tradingagents.llm_clients.cli_backends import (
    ClaudeRunner,
    CLIBackendError,
    CLIResult,
    CodexExecRunner,
    CodexMCPRunner,
    _run_subprocess,
)
from tradingagents.llm_clients.cli_chat_model import (
    CLIChatModel,
    ToolCallParseError,
    _extract_tool_calls,
    _render_messages,
)
from tradingagents.llm_clients.factory import create_llm_client


class FakeRunner:
    """Scripted runner: pops one (text, session_id) reply per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def start(self, prompt, system=None):
        self.calls.append(("start", prompt, system))
        text, session_id = self.replies.pop(0)
        return CLIResult(text, session_id)

    def resume(self, session_id, prompt):
        self.calls.append(("resume", session_id, prompt))
        text, new_id = self.replies.pop(0)
        return CLIResult(text, new_id or session_id)

    def close(self):
        pass


def make_model(replies, backend="claude"):
    model = CLIChatModel(model="default", backend=backend)
    model._runner = FakeRunner(replies)
    return model


def get_stock_price(ticker: str) -> str:
    """Get the latest price for a ticker."""
    return "100.0"


# -- factory / registration ---------------------------------------------------


def test_factory_dispatch():
    from tradingagents.llm_clients.cli_client import (
        ClaudeCodeCLIClient,
        CodexCLIClient,
    )

    codex = create_llm_client("codex-cli", "default")
    claude = create_llm_client("claude-code", "sonnet")
    assert isinstance(codex, CodexCLIClient)
    assert isinstance(claude, ClaudeCodeCLIClient)
    assert codex.validate_model() and claude.validate_model()


def test_no_api_key_required():
    from tradingagents.llm_clients.api_key_env import get_api_key_env

    assert get_api_key_env("codex-cli") is None
    assert get_api_key_env("claude-code") is None


def test_get_llm_passes_kwargs():
    client = create_llm_client(
        "codex-cli", "default",
        reasoning_effort="high", temperature=0.2, cli_persistent=False,
    )
    llm = client.get_llm()
    assert isinstance(llm, CLIChatModel)
    assert llm.reasoning_effort == "high"
    assert llm.persistent is False
    # temperature is accepted (cross-provider kwarg) even though CLIs ignore it
    assert llm.temperature == 0.2


def test_provider_menu_and_catalog():
    from cli.utils import _llm_provider_table
    from tradingagents.llm_clients.model_catalog import get_model_options

    keys = {key for _, key, _ in _llm_provider_table()}
    assert {"codex-cli", "claude-code"} <= keys
    assert any(v == "default" for _, v in get_model_options("codex-cli", "deep"))
    assert any(v == "opus" for _, v in get_model_options("claude-code", "deep"))


# -- provider-specific defaults -------------------------------------------------


def _reload_config_with_env(monkeypatch, **overrides):
    import importlib

    import tradingagents.default_config as default_config_module

    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_codex_cli_defaults_to_gpt55_max_reasoning(monkeypatch):
    dc = _reload_config_with_env(
        monkeypatch, TRADINGAGENTS_LLM_PROVIDER="codex-cli"
    )
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gpt-5.5"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.5"
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] == "xhigh"


def test_claude_code_defaults_to_fable_and_opus_ultra(monkeypatch):
    dc = _reload_config_with_env(
        monkeypatch, TRADINGAGENTS_LLM_PROVIDER="claude-code"
    )
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "claude-fable-5"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "claude-opus-4-8"
    assert dc.DEFAULT_CONFIG["anthropic_effort"] == "xhigh"


def test_cli_provider_defaults_respect_explicit_env(monkeypatch):
    dc = _reload_config_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="claude-code",
        TRADINGAGENTS_QUICK_THINK_LLM="haiku",
        TRADINGAGENTS_ANTHROPIC_EFFORT="high",
    )
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "claude-fable-5"  # still default
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "haiku"
    assert dc.DEFAULT_CONFIG["anthropic_effort"] == "high"


def test_cli_defaults_do_not_touch_api_providers(monkeypatch):
    dc = _reload_config_with_env(
        monkeypatch, TRADINGAGENTS_LLM_PROVIDER="openai"
    )
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.4-mini"
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] is None


def test_provider_kwargs_default_to_max_thinking():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    def kwargs_for(config):
        graph = object.__new__(TradingAgentsGraph)  # skip heavy __init__
        graph.config = config
        return graph._get_provider_kwargs()

    assert kwargs_for({"llm_provider": "codex-cli"})["reasoning_effort"] == "xhigh"
    assert kwargs_for({"llm_provider": "claude-code"})["reasoning_effort"] == "xhigh"
    assert kwargs_for(
        {"llm_provider": "claude-code", "anthropic_effort": "low"}
    )["reasoning_effort"] == "low"


def test_claude_runner_effort_flag():
    assert "--effort" not in ClaudeRunner("sonnet")._base_cmd()
    cmd = ClaudeRunner("claude-fable-5", effort="xhigh")._base_cmd()
    assert cmd[cmd.index("--effort") + 1] == "xhigh"


# -- plain invoke -------------------------------------------------------------


def test_plain_invoke():
    model = make_model([("The market looks calm.", None)])
    result = model.invoke([HumanMessage(content="How is the market?")])
    assert result.content == "The market looks calm."
    assert result.tool_calls == []


def test_system_message_goes_to_system_channel():
    model = make_model([("ok", None)])
    model.invoke([
        SystemMessage(content="You are a trading assistant."),
        HumanMessage(content="hi"),
    ])
    kind, prompt, system = model._runner.calls[0]
    assert kind == "start"
    assert "trading assistant" in system
    assert "trading assistant" not in prompt
    assert "## User\nhi" in prompt


# -- prompt-emulated tool calling ----------------------------------------------


def tool_reply(calls):
    return "```json\n" + json.dumps({"tool_calls": calls}) + "\n```"


def test_bind_tools_parses_tool_calls():
    model = make_model([
        (tool_reply([{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]), "s1"),
    ])
    bound = model.bind_tools([get_stock_price])
    result = bound.invoke([HumanMessage(content="price of AAPL?")])
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "get_stock_price"
    assert call["args"] == {"ticker": "AAPL"}
    assert call["id"].startswith("call_")
    assert result.content == ""
    # The tool schema and protocol went into the system channel.
    _, _, system = model._runner.calls[0]
    assert "get_stock_price" in system and "tool_calls" in system


def test_final_answer_with_tools_bound():
    model = make_model([("Final report: buy.", "s1")])
    bound = model.bind_tools([get_stock_price])
    result = bound.invoke([HumanMessage(content="analyze")])
    assert result.tool_calls == []
    assert result.content == "Final report: buy."


def test_unfenced_tool_call_json_is_found():
    raw = 'I will call a tool. {"tool_calls": [{"name": "get_stock_price", "arguments": {"ticker": "TSLA"}}]}'
    calls = _extract_tool_calls(raw)
    assert calls[0]["args"] == {"ticker": "TSLA"}


def test_string_arguments_are_decoded():
    raw = tool_reply([{"name": "get_stock_price", "arguments": '{"ticker": "MSFT"}'}])
    assert _extract_tool_calls(raw)[0]["args"] == {"ticker": "MSFT"}


def test_malformed_tool_json_raises_parse_error():
    with pytest.raises(ToolCallParseError):
        _extract_tool_calls('```json\n{"tool_calls": [{"name": }]}\n```')
    with pytest.raises(ToolCallParseError):
        _extract_tool_calls('{"tool_calls": "not-a-list"}')


def test_malformed_tool_json_recovers_via_session_retry():
    model = make_model([
        ('{"tool_calls": [{"name": }]}', "s1"),  # malformed
        (tool_reply([{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}]), "s1"),
    ])
    bound = model.bind_tools([get_stock_price])
    result = bound.invoke([HumanMessage(content="go")])
    assert result.tool_calls[0]["args"] == {"ticker": "NVDA"}
    kinds = [call[0] for call in model._runner.calls]
    assert kinds == ["start", "resume"]
    assert "invalid" in model._runner.calls[1][2]


def test_malformed_tool_json_without_session_degrades_to_text():
    reply = '{"tool_calls": [{"name": }]}'
    model = make_model([(reply, None)])  # no session id -> no retry channel
    bound = model.bind_tools([get_stock_price])
    result = bound.invoke([HumanMessage(content="go")])
    assert result.tool_calls == []
    assert result.content == reply


# -- session registry ---------------------------------------------------------


def test_tool_loop_resumes_session_with_delta_only():
    model = make_model([
        (tool_reply([{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]), "s1"),
        ("Report: AAPL trades at 100.", "s1"),
    ])
    bound = model.bind_tools([get_stock_price])
    history = [
        SystemMessage(content="Analyst system prompt"),
        HumanMessage(content="Analyze AAPL"),
    ]
    first = bound.invoke(history)
    history = history + [first, ToolMessage(
        content="100.0", tool_call_id=first.tool_calls[0]["id"],
        name="get_stock_price",
    )]
    second = bound.invoke(history)
    assert second.content == "Report: AAPL trades at 100."

    kind, session_id, delta = model._runner.calls[1]
    assert kind == "resume" and session_id == "s1"
    assert "Tool result" in delta and "100.0" in delta
    # The delta must not replay the earlier turns.
    assert "Analyze AAPL" not in delta


def test_prefix_mismatch_starts_fresh_session():
    model = make_model([
        ("first", "s1"),
        ("second", "s2"),
    ])
    model.invoke([HumanMessage(content="conversation A")])
    model.invoke([HumanMessage(content="conversation B, unrelated")])
    kinds = [call[0] for call in model._runner.calls]
    assert kinds == ["start", "start"]


def test_resume_failure_falls_back_to_full_transcript():
    class FlakyRunner(FakeRunner):
        def resume(self, session_id, prompt):
            self.calls.append(("resume", session_id, prompt))
            raise CLIBackendError("session expired")

    model = CLIChatModel(model="default", backend="claude")
    model._runner = FlakyRunner([("first", "s1"), ("recovered", "s3")])
    history = [HumanMessage(content="hello")]
    first = model.invoke(history)
    history = history + [first, HumanMessage(content="continue")]
    second = model.invoke(history)
    assert second.content == "recovered"
    kinds = [call[0] for call in model._runner.calls]
    assert kinds == ["start", "resume", "start"]
    # The fallback start re-sends the full transcript.
    assert "hello" in model._runner.calls[2][1]


# -- structured output ---------------------------------------------------------


class Decision(BaseModel):
    action: str
    confidence: float


def test_structured_output_happy_path():
    payload = '```json\n{"action": "BUY", "confidence": 0.8}\n```'
    model = make_model([(payload, None)])
    structured = model.with_structured_output(Decision)
    result = structured.invoke("What is your decision?")
    assert isinstance(result, Decision)
    assert result.action == "BUY"


def test_structured_output_retries_then_raises():
    model = make_model([
        ("not json at all", "s1"),
        ("still not json", "s1"),
    ])
    structured = model.with_structured_output(Decision)
    with pytest.raises(ValueError):
        structured.invoke("decision?")
    kinds = [call[0] for call in model._runner.calls]
    assert kinds == ["start", "resume"]


def test_structured_output_correction_recovers():
    model = make_model([
        ('{"action": "BUY"}', "s1"),  # missing required field
        ('{"action": "BUY", "confidence": 0.5}', "s1"),
    ])
    structured = model.with_structured_output(Decision)
    result = structured.invoke("decision?")
    assert result.confidence == 0.5


def test_structured_output_unsupported_schema_raises_not_implemented():
    model = make_model([])
    with pytest.raises(NotImplementedError):
        model.with_structured_output({"type": "object"})
    with pytest.raises(NotImplementedError):
        model.with_structured_output(Decision, include_raw=True)


def test_bind_structured_helper_integration():
    from tradingagents.agents.utils.structured import bind_structured

    model = make_model([('{"action": "SELL", "confidence": 0.9}', None)])
    structured = bind_structured(model, Decision, "test-agent")
    assert structured is not None
    assert structured.invoke("go").action == "SELL"


# -- message rendering ----------------------------------------------------------


def test_render_messages_includes_tool_calls_and_results():
    ai = AIMessage(content="", tool_calls=[
        {"name": "get_stock_price", "args": {"ticker": "AAPL"},
         "id": "call_1", "type": "tool_call"},
    ])
    tool = ToolMessage(content="100.0", tool_call_id="call_1", name="get_stock_price")
    rendered = _render_messages([HumanMessage(content="hi"), ai, tool])
    assert "## User\nhi" in rendered
    assert '"tool_calls"' in rendered
    assert "## Tool result (name=get_stock_price, id=call_1)" in rendered


def test_block_content_is_flattened():
    message = AIMessage(content=[
        {"type": "reasoning", "reasoning": "hmm"},
        {"type": "text", "text": "hello"},
    ])
    assert "hello" in _render_messages([message])
    assert "hmm" not in _render_messages([message])


# -- subprocess plumbing ---------------------------------------------------------


def test_run_subprocess_failure_includes_stderr():
    script = f"{sys.executable} -c \"import sys; sys.stderr.write('login required'); sys.exit(3)\""
    with pytest.raises(CLIBackendError) as excinfo:
        _run_subprocess(["sh", "-c", script], "", timeout=30)
    assert "login required" in str(excinfo.value)


def test_run_subprocess_missing_binary():
    with pytest.raises(CLIBackendError) as excinfo:
        _run_subprocess(["definitely-not-a-real-binary-xyz"], "", timeout=5)
    assert "not found" in str(excinfo.value)


def test_claude_runner_parses_envelope():
    runner = ClaudeRunner("sonnet")
    result = runner._parse(json.dumps({
        "is_error": False, "result": "pong", "session_id": "abc",
    }))
    assert result.text == "pong" and result.session_id == "abc"
    with pytest.raises(CLIBackendError):
        runner._parse(json.dumps({"is_error": True, "result": "limit reached"}))
    with pytest.raises(CLIBackendError):
        runner._parse("garbage")


def test_codex_exec_session_id_extraction():
    jsonl = "\n".join([
        "codex banner line",
        json.dumps({"type": "item.completed"}),
        json.dumps({"type": "thread.started", "thread_id": "t-123"}),
    ])
    assert CodexExecRunner._extract_session_id(jsonl) == "t-123"
    assert CodexExecRunner._extract_session_id("no json here") is None


def test_default_model_omits_model_flag():
    assert "--model" not in ClaudeRunner("default")._base_cmd()
    assert "--model" in ClaudeRunner("sonnet")._base_cmd()
    assert "-m" not in CodexExecRunner("default")._common_flags()
    assert "-m" in CodexExecRunner("gpt-5.5")._common_flags()


# -- persistent MCP runner against a scripted server -----------------------------


FAKE_MCP_SERVER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, sys

    def send(payload):
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        msg = json.loads(line)
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {"capabilities": {}}})
        elif method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"]["arguments"]
            if name == "codex":
                assert args["sandbox"] == "read-only"
                assert "base-instructions" in args
                # Interleave a notification to prove the client skips them.
                send({"jsonrpc": "2.0", "method": "codex/event", "params": {}})
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "reply:" + args["prompt"]}],
                    "structuredContent": {"conversationId": "conv-7"},
                }})
            elif name == "codex-reply":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": "resumed:" + args["conversationId"]}],
                }})
    """
)


@pytest.fixture()
def fake_mcp_runner(tmp_path):
    server = tmp_path / "fake-codex"
    server.write_text(FAKE_MCP_SERVER)
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    runner = CodexMCPRunner(timeout=30)
    runner.binary = str(server)
    yield runner
    runner.close()


def test_mcp_runner_start_and_resume(fake_mcp_runner):
    result = fake_mcp_runner.start("hello", system="be brief", model="default")
    assert result.text == "reply:hello"
    assert result.session_id == "conv-7"
    resumed = fake_mcp_runner.resume(result.session_id, "more")
    assert resumed.text == "resumed:conv-7"
    assert resumed.session_id == "conv-7"


def test_mcp_runner_restarts_dead_server(fake_mcp_runner):
    first = fake_mcp_runner.start("one", model="default")
    assert first.text == "reply:one"
    # Kill the server behind the runner's back; the next call must recover.
    fake_mcp_runner._proc.kill()
    fake_mcp_runner._proc.wait()
    second = fake_mcp_runner.start("two", model="default")
    assert second.text == "reply:two"


def test_mcp_runner_broken_server_raises(tmp_path):
    server = tmp_path / "broken-codex"
    server.write_text("#!/bin/sh\nexit 1\n")
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    runner = CodexMCPRunner(timeout=5)
    runner.binary = str(server)
    try:
        with pytest.raises(CLIBackendError):
            runner.start("hello", model="default")
    finally:
        runner.close()
