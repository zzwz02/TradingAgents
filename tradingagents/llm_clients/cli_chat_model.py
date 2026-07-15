"""LangChain chat model backed by a local subscription CLI (Codex / Claude Code).

The graph only ever needs three things from an LLM object (see
``tradingagents/agents``): ``invoke`` returning a message with string
``content``, ``bind_tools`` whose result carries ``tool_calls`` the LangGraph
``ToolNode`` can execute, and (optionally) ``with_structured_output``. This
model satisfies all three by shelling out to ``codex`` / ``claude``:

- **Tool calling is prompt-emulated.** The CLIs speak plain text, not the
  function-calling wire protocol, so tool JSON Schemas are injected into the
  system text and the model is instructed to answer with a fenced
  ``{"tool_calls": [...]}`` JSON block, which is parsed back into
  ``AIMessage.tool_calls``.
- **Multi-turn goes through session resume, not transcript replay.** Each
  reply's session id is remembered against a fingerprint of the message
  prefix; when the next call extends that prefix (the analyst tool loop),
  only the delta is sent via the CLI's resume mechanism, keeping the
  provider-side prompt cache warm. Any mismatch or failure falls back to a
  fresh conversation with the full transcript — sessions are an optimization,
  never a correctness dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, PrivateAttr

from .cli_backends import (
    DEFAULT_TIMEOUT,
    ClaudeRunner,
    CLIBackendError,
    CLIResult,
    CodexExecRunner,
    CodexMCPModelRunner,
    new_call_id,
)

logger = logging.getLogger(__name__)

_MAX_TRACKED_SESSIONS = 32

_FENCED_JSON = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)

_TOOL_PROTOCOL_TEMPLATE = """\
# Tool-calling protocol

You can use the following tools. Tool definitions (JSON Schema):

{tool_schemas}

To call one or more tools, reply with ONLY a single fenced json block and no \
other text:

```json
{{"tool_calls": [{{"name": "<tool_name>", "arguments": {{ ... }}}}]}}
```

Rules:
- "arguments" must be a JSON object satisfying the tool's parameter schema.
- Tool results will come back as "## Tool result" blocks; you may then call \
more tools or write your final answer.
- The final answer must be normal markdown and must NOT contain any \
{{"tool_calls": ...}} JSON.
- These declared tools are the only tools that exist. Do not invent tool \
names."""


class ToolCallParseError(ValueError):
    """The reply contained a tool_calls JSON candidate that failed to parse."""


class CLIChatModel(BaseChatModel):
    """Chat model that routes completions through ``codex`` or ``claude``."""

    model: str
    backend: Literal["codex", "claude"]
    timeout: float = DEFAULT_TIMEOUT
    # Thinking depth. Codex: model_reasoning_effort (minimal..max);
    # Claude Code: --effort (low..max).
    reasoning_effort: str | None = None
    # Accepted for cross-provider config compatibility; the CLIs expose no
    # sampling temperature, so this is intentionally unused.
    temperature: float | None = None
    # Codex only: use the persistent `codex mcp-server` (fast path) instead of
    # spawning `codex exec` per call.
    persistent: bool = True

    _runner: Any = PrivateAttr(default=None)
    # fingerprint(messages[:n]) -> (session_id, n); LRU-bounded.
    _sessions: OrderedDict = PrivateAttr(default_factory=OrderedDict)

    @property
    def _llm_type(self) -> str:
        return f"{self.backend}-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "backend": self.backend}

    # -- runner ---------------------------------------------------------------

    def _get_runner(self):
        if self._runner is None:
            if self.backend == "claude":
                self._runner = ClaudeRunner(
                    self.model,
                    timeout=self.timeout,
                    effort=self.reasoning_effort,
                )
            elif self.persistent:
                self._runner = CodexMCPModelRunner(
                    self.model,
                    timeout=self.timeout,
                    reasoning_effort=self.reasoning_effort,
                )
            else:
                self._runner = CodexExecRunner(
                    self.model,
                    timeout=self.timeout,
                    reasoning_effort=self.reasoning_effort,
                )
        return self._runner

    # -- LangChain surface ------------------------------------------------------

    def bind_tools(self, tools, **kwargs) -> Runnable:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._complete(list(messages), kwargs.get("tools"))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        if include_raw or not (
            isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
            # bind_structured() catches this and switches the agent to
            # free-text generation.
            raise NotImplementedError(
                "CLIChatModel.with_structured_output supports plain Pydantic "
                "schemas only"
            )
        schema_json = json.dumps(
            schema.model_json_schema(), ensure_ascii=False, indent=2
        )
        instruction = (
            "Respond with ONLY a single fenced ```json block containing one "
            "JSON object that conforms to this JSON Schema. No prose, no "
            "markdown outside the block.\n\nJSON Schema:\n" + schema_json
        )

        def _invoke(input_: Any, config: Any = None) -> BaseModel:
            messages = list(self._convert_input(input_).to_messages())
            messages.append(HumanMessage(content=instruction))
            reply, session_id = self._call_with_session(messages, tools=None)
            for attempt in (1, 2):
                try:
                    return schema.model_validate(_extract_json_object(reply))
                except Exception as exc:
                    if attempt == 2:
                        raise
                    logger.warning(
                        "structured output parse failed (%s); asking the "
                        "model to correct it", exc,
                    )
                    reply, session_id = self._followup(
                        messages,
                        reply,
                        session_id,
                        "Your previous reply was not a valid JSON object for "
                        f"the schema ({exc}). Re-emit ONLY the corrected "
                        "fenced json block.",
                    )
            raise AssertionError("unreachable")

        return RunnableLambda(_invoke)

    # -- completion core -------------------------------------------------------

    def _complete(
        self, messages: list[BaseMessage], tools: list[dict] | None
    ) -> AIMessage:
        text, session_id = self._call_with_session(messages, tools)

        if tools:
            for attempt in (1, 2):
                try:
                    tool_calls = _extract_tool_calls(text)
                except ToolCallParseError as exc:
                    if attempt == 2 or session_id is None:
                        logger.warning(
                            "tool-call JSON unrecoverable (%s); treating the "
                            "reply as a final text answer", exc,
                        )
                        tool_calls = None
                        break
                    logger.warning(
                        "tool-call JSON malformed (%s); asking the model to "
                        "correct it", exc,
                    )
                    text, session_id = self._followup(
                        messages,
                        text,
                        session_id,
                        f"Your tool-call JSON was invalid ({exc}). Re-emit "
                        "ONLY the corrected fenced json block per the "
                        "tool-calling protocol.",
                        tools=tools,
                    )
                    continue
                break
            if tool_calls:
                ai = AIMessage(content="", tool_calls=tool_calls)
                self._remember_session(messages, ai, session_id)
                return ai

        ai = AIMessage(content=text)
        self._remember_session(messages, ai, session_id)
        return ai

    def _call_with_session(
        self, messages: list[BaseMessage], tools: list[dict] | None
    ) -> tuple[str, str | None]:
        """Resume the longest matching session prefix, else start fresh."""
        runner = self._get_runner()

        match = self._find_session(messages)
        if match is not None:
            key, session_id, n = match
            delta = _render_messages(messages[n:])
            try:
                result = runner.resume(session_id, delta)
            except CLIBackendError as exc:
                logger.warning(
                    "session resume failed (%s); starting a fresh "
                    "conversation with the full transcript", exc,
                )
                self._sessions.pop(key, None)
            else:
                return result.text, result.session_id

        system_parts = [
            _message_text(m) for m in messages if isinstance(m, SystemMessage)
        ]
        if tools:
            system_parts.append(
                _TOOL_PROTOCOL_TEMPLATE.format(
                    tool_schemas=json.dumps(tools, ensure_ascii=False, indent=2)
                )
            )
        prompt = _render_messages(
            [m for m in messages if not isinstance(m, SystemMessage)]
        )
        result: CLIResult = runner.start(
            prompt, system="\n\n".join(system_parts) or None
        )
        return result.text, result.session_id

    def _followup(
        self,
        messages: list[BaseMessage],
        last_reply: str,
        session_id: str | None,
        correction: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, str | None]:
        """Send a correction turn, preferring the live session."""
        runner = self._get_runner()
        if session_id is not None:
            try:
                result = runner.resume(session_id, correction)
                return result.text, result.session_id
            except CLIBackendError as exc:
                logger.warning("correction resume failed (%s)", exc)
        retry_messages = messages + [
            AIMessage(content=last_reply),
            HumanMessage(content=correction),
        ]
        return self._call_with_session(retry_messages, tools=tools)

    # -- session registry --------------------------------------------------------

    def _find_session(
        self, messages: list[BaseMessage]
    ) -> tuple[str, str, int] | None:
        best: tuple[str, str, int] | None = None
        for key, (session_id, n) in self._sessions.items():
            if (
                n < len(messages)
                and (best is None or n > best[2])
                and _fingerprint(messages[:n]) == key
            ):
                best = (key, session_id, n)
        if best is not None:
            self._sessions.move_to_end(best[0])
        return best

    def _remember_session(
        self,
        messages: list[BaseMessage],
        reply: AIMessage,
        session_id: str | None,
    ) -> None:
        if not session_id:
            return
        # The next call in a tool loop will present `messages + [reply] +
        # tool results`, so the resumable prefix ends after our own reply.
        prefix = messages + [reply]
        self._sessions[_fingerprint(prefix)] = (session_id, len(prefix))
        while len(self._sessions) > _MAX_TRACKED_SESSIONS:
            self._sessions.popitem(last=False)


# -- message rendering ------------------------------------------------------


def _message_text(message: BaseMessage) -> str:
    """Flatten possibly block-structured content to plain text."""
    content = message.content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _render_message(message: BaseMessage) -> str:
    text = _message_text(message)
    if isinstance(message, SystemMessage):
        return f"## System\n{text}"
    if isinstance(message, HumanMessage):
        return f"## User\n{text}"
    if isinstance(message, AIMessage):
        rendered = f"## Assistant\n{text}".rstrip()
        if message.tool_calls:
            calls = [
                {"name": call["name"], "arguments": call["args"]}
                for call in message.tool_calls
            ]
            rendered += (
                "\n```json\n"
                + json.dumps({"tool_calls": calls}, ensure_ascii=False)
                + "\n```"
            )
        return rendered
    if isinstance(message, ToolMessage):
        label = message.name or "tool"
        return (
            f"## Tool result (name={label}, id={message.tool_call_id})\n"
            f"```\n{text}\n```"
        )
    return f"## {message.type}\n{text}"


def _render_messages(messages: list[BaseMessage]) -> str:
    return "\n\n".join(_render_message(m) for m in messages)


def _fingerprint(messages: list[BaseMessage]) -> str:
    payload = "\x00".join(_render_message(m) for m in messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -- reply parsing -----------------------------------------------------------


def _extract_tool_calls(text: str) -> list[dict] | None:
    """Parse the prompt-emulated tool-call envelope out of a CLI reply.

    Returns LangChain-shaped tool calls, ``None`` when the reply is a final
    text answer, and raises :class:`ToolCallParseError` when a candidate
    envelope is present but unusable (so the caller can ask for a fix).
    """
    candidate = _find_json_candidate(text, required_key='"tool_calls"')
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"invalid JSON: {exc}") from exc
    calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not isinstance(calls, list) or not calls:
        raise ToolCallParseError('"tool_calls" must be a non-empty list')

    tool_calls = []
    for call in calls:
        if not isinstance(call, dict) or not call.get("name"):
            raise ToolCallParseError(f"malformed tool call entry: {call!r}")
        args = call.get("arguments", call.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolCallParseError(
                    f"tool arguments are not valid JSON: {exc}"
                ) from exc
        if not isinstance(args, dict):
            raise ToolCallParseError('"arguments" must be a JSON object')
        tool_calls.append(
            {
                "name": str(call["name"]),
                "args": args,
                "id": new_call_id(),
                "type": "tool_call",
            }
        )
    return tool_calls


def _extract_json_object(text: str) -> Any:
    candidate = _find_json_candidate(text, required_key=None)
    if candidate is None:
        raise ValueError("no JSON object found in the reply")
    return json.loads(candidate)


def _find_json_candidate(text: str, required_key: str | None) -> str | None:
    """Locate a JSON object in the reply: fenced blocks first, then raw."""
    for match in _FENCED_JSON.finditer(text):
        block = match.group(1).strip()
        if not block.startswith("{"):
            continue
        if required_key is None or required_key in block:
            return block

    if required_key is not None:
        anchor = text.find(required_key)
        if anchor == -1:
            return None
        start = text.rfind("{", 0, anchor)
        if start == -1:
            return None
    else:
        start = text.find("{")
        if start == -1:
            return None
    return _balanced_json_slice(text, start)


def _balanced_json_slice(text: str, start: int) -> str:
    """Slice a brace-balanced JSON object starting at ``start``.

    String-aware so braces inside string values don't break the balance;
    returns the (possibly truncated) tail when unbalanced, letting
    ``json.loads`` produce the actual error message.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]
