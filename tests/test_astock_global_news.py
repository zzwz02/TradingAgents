import json
import logging


class _HtmlResponse:
    def __init__(self, text: str):
        self.content = text.encode("utf-8")

    def raise_for_status(self):
        return None


class _JsonResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self):
        return self._payload


def test_fetch_cls_news_from_mobile_next_data(monkeypatch):
    from tradingagents.dataflows import a_stock

    payload = {
        "props": {
            "initialState": {
                "roll_data": [
                    {
                        "title": "",
                        "brief": "财联社7月22日电，测试快讯。",
                        "content": "财联社7月22日电，测试快讯正文。",
                        "ctime": 1784705093,
                    }
                ]
            }
        }
    }
    html = (
        "<script>__NEXT_DATA__ = "
        + json.dumps(payload, ensure_ascii=False)
        + "\nmodule = {};</script>"
    )
    monkeypatch.setattr(
        a_stock._requests,
        "get",
        lambda *args, **kwargs: _HtmlResponse(html),
    )

    result = a_stock._fetch_cls_global_news(10)

    assert len(result) == 1
    assert result[0]["title"] == "财联社7月22日电，测试快讯。"
    assert result[0]["source"] == "CLS Wire"


def test_global_news_silently_uses_eastmoney_when_cls_fails(monkeypatch, caplog):
    from tradingagents.dataflows import a_stock

    monkeypatch.setattr(
        a_stock,
        "_fetch_cls_global_news",
        lambda limit: (_ for _ in ()).throw(ValueError("CLS HTML changed")),
    )
    monkeypatch.setattr(
        a_stock,
        "_em_get",
        lambda *args, **kwargs: _JsonResponse(
            {
                "data": {
                    "fastNewsList": [
                        {
                            "title": "东财备用快讯",
                            "summary": "备用源正常",
                            "showTime": "2026-07-22 15:00:00",
                        }
                    ]
                }
            }
        ),
    )

    with caplog.at_level(logging.WARNING):
        result = a_stock.get_global_news("2026-07-22", limit=10)

    assert "东财备用快讯" in result
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_global_news_warns_only_when_every_source_fails(monkeypatch, caplog):
    from tradingagents.dataflows import a_stock

    monkeypatch.setattr(
        a_stock,
        "_fetch_cls_global_news",
        lambda limit: (_ for _ in ()).throw(ValueError("CLS unavailable")),
    )
    monkeypatch.setattr(
        a_stock,
        "_em_get",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("EM unavailable")),
    )

    with caplog.at_level(logging.WARNING):
        result = a_stock.get_global_news("2026-07-22", limit=10)

    assert result == "No global news found for 2026-07-22"
    warnings = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "CLS unavailable" in warnings[0]
    assert "EM unavailable" in warnings[0]
