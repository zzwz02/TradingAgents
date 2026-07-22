import json

import pandas as pd


class _JsonResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def test_sina_nested_financial_reports_are_normalized_and_date_filtered():
    from tradingagents.dataflows import a_stock

    result = {
        "report_date": [
            {"date_value": "20260331", "date_description": "2026一季报"},
            {"date_value": "20260930", "date_description": "2026三季报"},
        ],
        "report_list": {
            "20260331": {
                "publish_date": "20260422",
                "rType": "合并期末",
                "rCurrency": "CNY",
                "is_audit": "未审计",
                "data": [
                    {
                        "item_field": "BIZINCO",
                        "item_title": "营业收入",
                        "item_value": "98497579591",
                        "item_tongbi": 0.24794,
                    },
                    {
                        "item_field": "PARENETP",
                        "item_title": "归属于母公司所有者的净利润",
                        "item_value": "20079391507",
                        "item_tongbi": 0.975,
                    },
                ],
            },
            "20260930": {
                "publish_date": "20261020",
                "data": [
                    {
                        "item_field": "BIZINCO",
                        "item_title": "营业收入",
                        "item_value": "999",
                    }
                ],
            },
        },
    }

    df = a_stock._parse_sina_financial_reports(
        result, "lrb", "quarterly", "2026-07-22"
    )

    assert list(df["报告日"]) == ["2026-03-31"]
    assert df.iloc[0]["营业收入"] == 98497579591.0
    assert df.iloc[0]["营业收入同比"] == "24.79%"
    assert df.iloc[0]["归属于母公司所有者的净利润同比"] == "97.50%"


def test_financial_report_uses_eastmoney_when_sina_is_empty(monkeypatch):
    from tradingagents.dataflows import a_stock

    expected = pd.DataFrame([{"报告日": "2026-03-31", "资产总计": 100.0}])
    monkeypatch.setattr(
        a_stock, "_get_financial_report_sina", lambda *args: pd.DataFrame()
    )
    monkeypatch.setattr(
        a_stock, "_get_financial_report_eastmoney", lambda *args: expected
    )

    result, source = a_stock._get_financial_report(
        "601899", "资产负债表", "quarterly", "2026-07-22"
    )

    pd.testing.assert_frame_equal(result, expected)
    assert source == "Eastmoney datacenter fallback"


def test_tencent_market_cap_fields_are_not_reversed(monkeypatch):
    from tradingagents.dataflows import a_stock

    values = [""] * 60
    values[1] = "紫金矿业"
    values[3] = "32.20"
    values[44] = "6633.78"
    values[45] = "8562.21"
    raw = f'v_sh601899="{"~".join(values)}";'.encode("gbk")

    class _UrlResponse:
        def read(self):
            return raw

    monkeypatch.setattr(a_stock.urllib.request, "urlopen", lambda *args, **kwargs: _UrlResponse())

    quote = a_stock._tencent_quote(["601899"])["601899"]

    assert quote["mcap_yi"] == 8562.21
    assert quote["float_mcap_yi"] == 6633.78


def test_lockup_uses_current_eastmoney_field_names(monkeypatch):
    from tradingagents.dataflows import a_stock

    calls = 0

    def datacenter(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                {
                    "FREE_DATE": "2025-12-08 00:00:00",
                    "FREE_SHARES_TYPE": "股权激励限售股份",
                    "CURRENT_FREE_SHARES": 75.48,
                    "FREE_RATIO": 0.000036660899,
                    "LIFT_MARKET_CAP": 2377.62,
                }
            ]
        return []

    monkeypatch.setattr(a_stock, "_eastmoney_datacenter", datacenter)

    result = a_stock.get_lockup_expiry("601899", "2026-07-22")

    assert "股权激励限售股份" in result
    assert "75.48 万股" in result
    assert "0.0037%" in result
    assert "0.24 亿元" in result


def test_incomplete_northbound_series_does_not_create_false_total(monkeypatch):
    from tradingagents.dataflows import a_stock

    payload = {
        "time": ["14:59", "15:00"],
        "hgt": [-10.03, -9.28],
        "sgt": [379.75],
    }
    monkeypatch.setattr(
        a_stock._requests, "get", lambda *args, **kwargs: _JsonResponse(payload)
    )
    monkeypatch.setattr(
        a_stock,
        "_save_northbound_snapshot",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not cache")),
    )
    real_datetime = a_stock.datetime

    class _FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 22, 15, 1)

    monkeypatch.setattr(a_stock, "datetime", _FixedDatetime)

    result = a_stock.get_northbound_flow("2026-07-22")

    assert "SGT 序列不完整 (1/2)" in result
    assert "北向资金合计" in result
    assert "Total=370.47" not in result
    assert "Signal:" not in result


def test_guba_sentiment_exposes_post_engagement(monkeypatch):
    from tradingagents.dataflows import a_stock

    payload = {
        "re": [
            {
                "post_publish_time": "2026-07-22 10:00:00",
                "post_title": "放量突破，看多",
                "post_content": "准备加仓",
                "post_click_count": 1200,
                "post_comment_count": 30,
                "post_like_count": 18,
            }
        ],
        "count": 12345,
    }
    html = "<script>var article_list=" + json.dumps(payload, ensure_ascii=False) + ";</script>"
    monkeypatch.setattr(
        a_stock, "_em_get", lambda *args, **kwargs: _JsonResponse(text=html)
    )

    result = a_stock.get_social_sentiment("601899", "2026-07-22", 7, 10)

    assert "board total=12345" in result
    assert "Posts=1 Reads=1200 Comments=30 Likes=18" in result
    assert "positive=1" in result
    assert "放量突破，看多" in result


def test_eastmoney_announcements_include_category_and_detail_link(monkeypatch):
    from tradingagents.dataflows import a_stock

    payload = {
        "data": {
            "list": [
                {
                    "art_code": "AN202607091826849528",
                    "notice_date": "2026-07-10 00:00:00",
                    "title": "紫金矿业2026年半年度业绩预增公告",
                    "columns": [
                        {"column_code": "001002004001", "column_name": "业绩预告"}
                    ],
                }
            ]
        }
    }
    monkeypatch.setattr(
        a_stock, "_em_get", lambda *args, **kwargs: _JsonResponse(payload)
    )

    result = a_stock._fetch_announcements_eastmoney("601899")

    assert result[0]["source"] == "东方财富公告"
    assert result[0]["content"] == "公告分类: 业绩预告"
    assert result[0]["url"].endswith(
        "/601899/AN202607091826849528.html"
    )


def test_graph_registers_new_prefetch_and_social_tools():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = object.__new__(TradingAgentsGraph)
    nodes = graph._create_tool_nodes()

    assert "get_fundamentals" in nodes["market"].tools_by_name
    assert "get_social_sentiment" in nodes["social"].tools_by_name
