"""Regression tests for Web LLM defaults and startup diagnostics."""

import importlib


def test_web_model_config_exposes_subscription_cli_providers_first(monkeypatch):
    import tradingagents.default_config as default_config_module
    import web.components.sidebar as sidebar

    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)

    dc = importlib.reload(default_config_module)
    sidebar = importlib.reload(sidebar)

    assert sidebar._PROVIDER_KEYS[:2] == ["codex-cli", "claude-code"]
    assert sidebar._PROVIDER_KEYS[0] == dc.DEFAULT_CONFIG["llm_provider"]


def test_web_runner_prints_astock_pipeline_confirmation(capsys):
    from web.runner import _log_astock_pipeline_confirmation

    _log_astock_pipeline_confirmation(
        "000001",
        "2026-07-04",
        {
            "llm_provider": "codex-cli",
            "quick_think_llm": "gpt-5.5",
            "deep_think_llm": "gpt-5.5",
        },
    )

    out = capsys.readouterr().out
    assert "[A-STOCK PIPELINE]" in out
    assert "mootdx + 东财 + 新浪 + 同花顺" in out
    assert "Analyst=7 个" in out
    assert "T+1、涨跌停、最小手数、交易时段" in out
    assert "llm_provider=codex-cli" in out
