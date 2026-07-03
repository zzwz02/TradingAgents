"""Clients for the subscription-CLI providers (Codex CLI / Claude Code).

These providers require no API key: authentication is whatever the locally
installed ``codex`` / ``claude`` binary is logged in with (a ChatGPT or
Claude subscription). ``base_url`` is accepted for interface compatibility
but ignored — there is no HTTP endpoint to point at.
"""

from typing import Any

from .base_client import BaseLLMClient
from .cli_chat_model import CLIChatModel

_PASSTHROUGH_KWARGS = ("timeout", "reasoning_effort", "temperature", "callbacks")


class _CLIProviderClient(BaseLLMClient):
    provider: str
    backend: str

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model, "backend": self.backend}
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # config["cli_persistent"] = False switches Codex from the persistent
        # `codex mcp-server` fast path to one-shot `codex exec` calls.
        if self.kwargs.get("cli_persistent") is not None:
            llm_kwargs["persistent"] = bool(self.kwargs["cli_persistent"])
        return CLIChatModel(**llm_kwargs)

    def validate_model(self) -> bool:
        # The CLIs accept aliases ("sonnet", "default") and whatever model
        # the subscription serves; any model string is allowed.
        return True


class CodexCLIClient(_CLIProviderClient):
    """LLM access through the local ``codex`` binary (ChatGPT subscription)."""

    provider = "codex-cli"
    backend = "codex"


class ClaudeCodeCLIClient(_CLIProviderClient):
    """LLM access through the local ``claude`` binary (Claude subscription)."""

    provider = "claude-code"
    backend = "claude"
