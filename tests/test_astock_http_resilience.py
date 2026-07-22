import warnings

from requests.exceptions import ConnectionError, ProxyError


class _Response:
    def __init__(self, text=""):
        self.text = text
        self.encoding = None


def test_eastmoney_proxy_failure_retries_without_environment_proxy(monkeypatch):
    from tradingagents.dataflows import a_stock

    expected = _Response("direct response")
    calls = []

    def proxy_get(*args, **kwargs):
        calls.append("proxy")
        raise ProxyError("proxy disconnected")

    def direct_get(*args, **kwargs):
        calls.append("direct")
        return expected

    monkeypatch.setattr(a_stock._EM_SESSION, "get", proxy_get)
    monkeypatch.setattr(a_stock._EM_DIRECT_SESSION, "get", direct_get)
    monkeypatch.setattr(a_stock, "_EM_MIN_INTERVAL", 0.0)

    result = a_stock._em_get("https://push2.eastmoney.com/test")

    assert result is expected
    assert calls == ["proxy", "direct"]
    assert a_stock._EM_DIRECT_SESSION.trust_env is False


def test_eastmoney_push2_failure_uses_backup_endpoint(monkeypatch):
    from tradingagents.dataflows import a_stock

    expected = _Response("backup response")
    calls = []

    def session_get(url, *args, **kwargs):
        calls.append(("session", url))
        if "push2delay.eastmoney.com" in url:
            return expected
        raise ProxyError("proxy disconnected")

    def direct_get(url, *args, **kwargs):
        calls.append(("direct", url))
        raise ConnectionError("remote closed connection")

    monkeypatch.setattr(a_stock._EM_SESSION, "get", session_get)
    monkeypatch.setattr(a_stock._EM_DIRECT_SESSION, "get", direct_get)
    monkeypatch.setattr(a_stock, "_EM_MIN_INTERVAL", 0.0)

    result = a_stock._em_get("https://push2.eastmoney.com/api/qt/stock/get")

    assert result is expected
    assert calls == [
        ("session", "https://push2.eastmoney.com/api/qt/stock/get"),
        ("direct", "https://push2.eastmoney.com/api/qt/stock/get"),
        ("session", "https://push2delay.eastmoney.com/api/qt/stock/get"),
    ]


def test_ths_read_html_uses_stringio_without_future_warning(monkeypatch):
    from tradingagents.dataflows import a_stock

    html = """
    <table>
      <thead><tr><th>年度</th><th>预测机构数</th><th>最小值</th><th>均值</th></tr></thead>
      <tbody><tr><td>2026</td><td>10</td><td>2.5</td><td>3.0</td></tr></tbody>
    </table>
    """
    monkeypatch.setattr(
        a_stock._requests,
        "get",
        lambda *args, **kwargs: _Response(html),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = a_stock._ths_eps_forecast("601899")

    assert not result.empty
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
