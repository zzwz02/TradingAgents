"""CLI/API run parity and user position context.

The CLI streams the graph directly instead of calling propagate(), so it
must share the same pre/post-run steps via prepare_run()/finalize_run():
resolving pending memory-log entries, injecting past context, writing the
full_states_log_<date>.json state log, and appending the memory-log decision
entry. The user's average cost basis (CLI Step 3 / propagate(avg_cost=...))
flows through position_context into the trader and PM prompts and the JSON
state log.
"""

import functools
import json
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_utils import build_position_context
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph

DECISION_BUY = "Rating: Buy\nEnter at $189-192, 6% portfolio cap."


def _fake_final_state(position_context=""):
    return {
        "final_trade_decision": DECISION_BUY,
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-10",
        "position_context": position_context,
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "",
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "history": "", "judge_decision": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 1, "latest_speaker": "",
        },
    }


def _mock_graph(tmp_path):
    """MagicMock graph with a real memory log and real log/finalize methods."""
    mock_graph = MagicMock()
    mock_graph.memory_log = TradingMemoryLog(
        {"memory_log_path": str(tmp_path / "trading_memory.md")}
    )
    mock_graph.log_states_dict = {}
    mock_graph.config = {"results_dir": str(tmp_path)}
    mock_graph._log_state = functools.partial(
        TradingAgentsGraph._log_state, mock_graph
    )
    return mock_graph


# ---------------------------------------------------------------------------
# finalize_run / prepare_run: the shared CLI/API pre- and post-run steps
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFinalizeRun:
    def test_writes_json_state_log(self, tmp_path):
        mock_graph = _mock_graph(tmp_path)
        state = _fake_final_state(position_context="USER POSITION: avg cost 100.")
        TradingAgentsGraph.finalize_run(mock_graph, "NVDA", "2026-01-10", state)

        log_path = (
            tmp_path / "NVDA" / "TradingAgentsStrategy_logs"
            / "full_states_log_2026-01-10.json"
        )
        assert log_path.exists()
        logged = json.loads(log_path.read_text(encoding="utf-8"))
        assert logged["company_of_interest"] == "NVDA"
        assert logged["final_trade_decision"] == DECISION_BUY
        assert logged["position_context"] == "USER POSITION: avg cost 100."

    def test_appends_pending_memory_entry(self, tmp_path):
        mock_graph = _mock_graph(tmp_path)
        TradingAgentsGraph.finalize_run(
            mock_graph, "NVDA", "2026-01-10", _fake_final_state()
        )
        entries = mock_graph.memory_log.load_entries()
        assert len(entries) == 1
        assert entries[0]["ticker"] == "NVDA"
        assert entries[0]["pending"] is True

    def test_sets_ticker_and_curr_state(self, tmp_path):
        mock_graph = _mock_graph(tmp_path)
        state = _fake_final_state()
        TradingAgentsGraph.finalize_run(mock_graph, "NVDA", "2026-01-10", state)
        assert mock_graph.ticker == "NVDA"
        assert mock_graph.curr_state is state


@pytest.mark.unit
class TestPrepareRun:
    def test_resolves_pending_and_returns_past_context(self, tmp_path):
        log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-05", 0.05, 0.02, 5, "Correct call.")

        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.memory_log = log
        context = TradingAgentsGraph.prepare_run(mock_graph, "NVDA")

        mock_graph._resolve_pending_entries.assert_called_once_with("NVDA")
        assert "Correct call." in context


# ---------------------------------------------------------------------------
# Position context: build, state injection, and prompt injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPositionContext:
    def test_none_avg_cost_yields_empty(self):
        assert build_position_context("NVDA", None) == ""

    def test_includes_ticker_and_cost(self):
        context = build_position_context("NVDA", 123.45)
        assert "NVDA" in context
        assert "123.45" in context
        assert "USER POSITION" in context


@pytest.mark.unit
class TestPositionContextInState:
    def test_position_context_in_initial_state(self):
        state = Propagator().create_initial_state(
            "NVDA", "2026-01-10", position_context="USER POSITION: avg cost 100."
        )
        assert state["position_context"] == "USER POSITION: avg cost 100."

    def test_position_context_defaults_to_empty(self):
        state = Propagator().create_initial_state("NVDA", "2026-01-10")
        assert state["position_context"] == ""


def _structured_llm(captured, result):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or result
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _trader_state(position_context=""):
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy",
        "position_context": position_context,
    }


def _pm_state(position_context=""):
    return {
        "company_of_interest": "NVDA",
        "past_context": "",
        "position_context": position_context,
        "risk_debate_state": {
            "history": "Risk debate history.",
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "judge_decision": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 1,
        },
        "investment_plan": "Research plan.",
        "trader_investment_plan": "Trader plan.",
    }


@pytest.mark.unit
class TestPositionContextInPrompts:
    def test_trader_prompt_includes_position(self):
        captured = {}
        llm = _structured_llm(
            captured, TraderProposal(action=TraderAction.BUY, reasoning="ok")
        )
        trader = create_trader(llm)
        trader(_trader_state(build_position_context("NVDA", 123.45)))
        assert any("USER POSITION" in m["content"] for m in captured["prompt"])

    def test_trader_prompt_omits_position_when_absent(self):
        captured = {}
        llm = _structured_llm(
            captured, TraderProposal(action=TraderAction.BUY, reasoning="ok")
        )
        trader = create_trader(llm)
        trader(_trader_state())
        assert not any("USER POSITION" in m["content"] for m in captured["prompt"])

    def test_pm_prompt_includes_position(self):
        captured = {}
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold.",
            investment_thesis="Balanced.",
        )
        llm = _structured_llm(captured, decision)
        pm_node = create_portfolio_manager(llm)
        pm_node(_pm_state(build_position_context("NVDA", 123.45)))
        assert "USER POSITION" in captured["prompt"]

    def test_pm_prompt_omits_position_when_absent(self):
        captured = {}
        decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold.",
            investment_thesis="Balanced.",
        )
        llm = _structured_llm(captured, decision)
        pm_node = create_portfolio_manager(llm)
        pm_node(_pm_state())
        assert "USER POSITION" not in captured["prompt"]


# ---------------------------------------------------------------------------
# CLI average-cost prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAverageCost:
    def test_blank_skips(self):
        from unittest import mock

        import cli.main as m
        with mock.patch.object(m.typer, "prompt", return_value=""):
            assert m.get_average_cost() is None

    def test_valid_float(self):
        from unittest import mock

        import cli.main as m
        with mock.patch.object(m.typer, "prompt", return_value="123.45"):
            assert m.get_average_cost() == 123.45

    def test_invalid_then_valid_reprompts(self):
        from unittest import mock

        import cli.main as m
        with mock.patch.object(
            m.typer, "prompt", side_effect=["abc", "-5", "100"]
        ):
            assert m.get_average_cost() == 100.0
