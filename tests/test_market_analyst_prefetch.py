"""Tests for deterministic technical data prefetching in the market analyst."""

from tradingagents.agents.analysts import market_analyst


class _FakeTool:
    def __init__(self, name: str, response: str | Exception):
        self.name = name
        self.response = response
        self.calls: list[dict] = []

    def invoke(self, payload: dict) -> str:
        self.calls.append(payload)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_prefetch_market_data_fetches_stock_and_key_indicators(monkeypatch):
    stock_tool = _FakeTool("get_stock_data", "Date,Open,High,Low,Close,Volume")
    indicator_tool = _FakeTool("get_indicators", "## rsi values\n2026-07-03: 55")
    monkeypatch.setattr(market_analyst, "get_stock_data", stock_tool)
    monkeypatch.setattr(market_analyst, "get_indicators", indicator_tool)

    context = market_analyst._build_prefetched_market_data("000001", "2026-07-04")

    assert stock_tool.calls == [
        {
            "symbol": "000001",
            "start_date": "2026-05-20",
            "end_date": "2026-07-04",
        }
    ]
    assert indicator_tool.calls == [
        {
            "symbol": "000001",
            "indicator": "close_10_ema,close_50_sma,close_200_sma,macd,macds,macdh,rsi,atr",
            "curr_date": "2026-07-04",
            "look_back_days": 30,
        }
    ]
    assert "已预取的 A 股技术面数据" in context
    assert "Date,Open,High,Low,Close,Volume" in context
    assert "## rsi values" in context


def test_prefetch_market_data_records_tool_errors(monkeypatch):
    stock_tool = _FakeTool("get_stock_data", RuntimeError("network down"))
    indicator_tool = _FakeTool("get_indicators", "indicator ok")
    monkeypatch.setattr(market_analyst, "get_stock_data", stock_tool)
    monkeypatch.setattr(market_analyst, "get_indicators", indicator_tool)

    context = market_analyst._build_prefetched_market_data("000001", "bad-date")

    assert stock_tool.calls[0]["start_date"] == "bad-date"
    assert "[数据缺失: get_stock_data] RuntimeError: network down" in context
    assert "indicator ok" in context
