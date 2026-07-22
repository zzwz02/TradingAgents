import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


class _Response:
    def __init__(self, payload):
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None


def _mootdx_frame():
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-07-21 15:00:00"]),
            "open": [148.0],
            "high": [161.4],
            "low": [140.58],
            "close": [160.0],
            "volume": [113980756],
        }
    ).set_index("datetime")


def test_sina_retries_000300_as_shanghai_index(monkeypatch):
    from tradingagents.dataflows import a_stock

    symbols = []

    def fake_get(url, *, params, timeout):
        symbols.append(params["symbol"])
        if params["symbol"] == "sz000300":
            return _Response(None)
        return _Response(
            [
                {
                    "day": "2026-07-21",
                    "open": "4817.766",
                    "high": "4862.302",
                    "low": "4778.029",
                    "close": "4847.023",
                    "volume": "21462296900",
                }
            ]
        )

    monkeypatch.setattr(a_stock._requests, "get", fake_get)

    result = a_stock._sina_kline_fallback("000300", "2026-07-01", "2026-07-22")

    assert symbols == ["sz000300", "sh000300"]
    assert result["Close"].item() == 4847.023


def test_mootdx_kline_uses_index_endpoint_when_stock_bars_are_empty(monkeypatch):
    from tradingagents.dataflows import a_stock

    calls = []

    def fake_call(method, **kwargs):
        calls.append(method)
        if method == "bars":
            return pd.DataFrame()
        return _mootdx_frame()

    monkeypatch.setattr(a_stock, "_call_mootdx", fake_call)

    result, source = a_stock._mootdx_kline("000300")

    assert calls == ["bars", "index_bars"]
    assert not result.empty
    assert source == "mootdx index (TCP)"


def test_find_working_tdx_server_requires_protocol_health(monkeypatch):
    from tradingagents.dataflows import a_stock

    monkeypatch.setattr(
        a_stock,
        "_TDX_SERVERS",
        [("port-open-only", 7709), ("healthy", 7709)],
    )
    monkeypatch.setattr(
        a_stock,
        "_probe_tdx",
        lambda ip, port: ip == "healthy",
    )

    assert a_stock._find_working_tdx_server() == ("healthy", 7709)


def test_find_working_tdx_server_falls_back_to_complete_pool(monkeypatch):
    from tradingagents.dataflows import a_stock

    monkeypatch.setattr(a_stock, "_TDX_SERVERS", [("stale-preferred", 7709)])
    monkeypatch.setattr(
        a_stock,
        "_tdx_server_candidates",
        lambda: [("stale-preferred", 7709), ("healthy-fallback", 80)],
    )
    monkeypatch.setattr(
        a_stock,
        "_probe_tdx",
        lambda ip, port: ip == "healthy-fallback",
    )
    monkeypatch.setattr(a_stock, "_tdx_quarantined_until", {})

    assert a_stock._find_working_tdx_server() == ("healthy-fallback", 80)


def test_failed_tdx_client_is_quarantined(monkeypatch):
    from tradingagents.dataflows import a_stock

    class FakeClient:
        server = ("failed", 7709)

        def close(self):
            return None

    monkeypatch.setattr(a_stock, "_mootdx_client", FakeClient())
    monkeypatch.setattr(a_stock, "_tdx_quarantined_until", {})

    a_stock._close_mootdx_client(quarantine=True)

    assert ("failed", 7709) in a_stock._tdx_quarantined_until
    assert a_stock._mootdx_client is None


def test_mootdx_calls_are_serialized_across_tool_threads(monkeypatch):
    from tradingagents.dataflows import a_stock

    state_lock = threading.Lock()

    class FakeClient:
        active = 0
        max_active = 0

        def bars(self, **kwargs):
            with state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with state_lock:
                self.active -= 1
            return _mootdx_frame()

    client = FakeClient()
    monkeypatch.setattr(a_stock, "_get_mootdx_client", lambda: client)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: a_stock._call_mootdx("bars", symbol="688981"),
                range(8),
            )
        )

    assert all(not result.empty for result in results)
    assert client.max_active == 1
