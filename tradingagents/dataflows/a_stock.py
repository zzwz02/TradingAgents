"""A-stock (China mainland) data vendor for TradingAgents.

Zero third-party data dependency (no akshare). All sources are direct HTTP APIs
or mootdx TCP.

Data sources:
- mootdx (TCP 7709): OHLCV K-lines, financial snapshots, F10 text
- Tencent Finance (HTTP GBK): PE/PB/market cap/turnover
- 东方财富 push2 / datacenter-web (direct HTTP): stock info, dragon-tiger, lockup
- 新浪财经 (direct HTTP): K-line fallback, financial statements
- 同花顺 (direct HTTP): consensus EPS, hot stocks, northbound capital flow
- 财联社 (direct HTTP): global news wire
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json as _json
import os
import logging
import math
import random
import re as _re
import threading
import time
import uuid
import urllib.request

import pandas as pd
import requests as _requests

from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: ticker format & market detection
# ---------------------------------------------------------------------------

def _get_prefix(code: str) -> str:
    """6-digit A-stock code -> market prefix for Tencent API."""
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith("8"):
        return "bj"
    return "sz"


def _normalize_ticker(symbol: str) -> str:
    """Strip exchange prefix/suffix, return pure 6-digit code.

    Handles: '688017', 'SH688017', '688017.SH', 'sh688017'
    """
    s = symbol.strip().upper()
    # Remove .SH / .SZ / .BJ suffix
    for suffix in (".SH", ".SZ", ".BJ"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    # Remove SH / SZ / BJ prefix
    for prefix in ("SH", "SZ", "BJ"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return safe_ticker_component(s)


# ---------------------------------------------------------------------------
# Stock name <-> code mapping (cached)
# ---------------------------------------------------------------------------

_name_to_code: dict[str, str] | None = None
_code_to_name: dict[str, str] | None = None


def _build_name_code_map() -> tuple[dict[str, str], dict[str, str]]:
    """Build name→code and code→name maps via mootdx (both SH & SZ markets)."""
    global _name_to_code, _code_to_name
    if _name_to_code is not None:
        return _name_to_code, _code_to_name

    n2c: dict[str, str] = {}
    c2n: dict[str, str] = {}

    try:
        for market in (0, 1):  # 0=SZ, 1=SH
            stocks = _call_mootdx("stocks", market=market)
            if stocks is None or stocks.empty:
                continue
            for _, row in stocks.iterrows():
                code = str(row["code"]).strip()
                name = str(row["name"]).strip()
                if not _re.match(r"^[036]\d{5}$", code):
                    continue
                clean_name = name.replace(" ", "").replace("　", "")
                n2c[clean_name] = code
                c2n[code] = clean_name
    except Exception as e:
        # 网络抖动/通达信不可达时给出明确提示，而非冒泡成风马牛不相及的报错（#46/#66）
        raise ValueError(
            "无法通过 mootdx 解析股票名称（通达信服务暂时不可达）：%s。"
            "请稍后重试，或直接输入 6 位股票代码。" % e
        ) from e

    _name_to_code = n2c
    _code_to_name = c2n
    logger.info("Built stock name-code map: %d entries", len(n2c))
    return _name_to_code, _code_to_name


def resolve_ticker(user_input: str) -> str:
    """Resolve user input (code or Chinese name) to a 6-digit A-stock code.

    Accepts: '600379', 'SH600379', '600379.SH', '宝光股份'
    Returns: '600379'
    Raises: ValueError if not resolvable.
    """
    s = user_input.strip()
    if not s:
        raise ValueError("输入不能为空")

    has_chinese = any("一" <= ch <= "鿿" for ch in s)

    if not has_chinese:
        return _normalize_ticker(s)

    clean = s.replace(" ", "").replace("　", "")
    n2c, _ = _build_name_code_map()

    if clean in n2c:
        return n2c[clean]

    matches = {name: code for name, code in n2c.items() if clean in name}
    if len(matches) == 1:
        return next(iter(matches.values()))
    if len(matches) > 1:
        examples = ", ".join(f"{n}({c})" for n, c in list(matches.items())[:5])
        raise ValueError(f"'{s}' 匹配到多只股票: {examples}，请输入完整名称或代码")

    raise ValueError(f"找不到股票 '{s}'，请检查名称是否正确")


# ---------------------------------------------------------------------------
# mootdx client (serialized singleton with protocol health checks)
# ---------------------------------------------------------------------------

_mootdx_client = None
_mootdx_lock = threading.RLock()
_mootdx_unavailable_until = 0.0
_tdx_quarantined_until: dict[tuple[str, int], float] = {}

_TDX_PROBE_TIMEOUT_SECONDS = 1.5
_TDX_RETRY_COOLDOWN_SECONDS = 300.0

# Fast first-tier nodes, protocol-verified with both 601899 stock bars and
# 000300 index bars on 2026-07-22.  If these age out, the second tier is built
# dynamically from tdxpy + mootdx's complete packaged host lists.
_TDX_SERVERS = [
    ("115.238.56.198", 7709),
    ("115.238.90.165", 7709),
    ("60.191.117.167", 7709),
    ("60.12.136.250", 7709),
    ("180.153.18.170", 7709),
    ("180.153.18.172", 80),
    ("202.108.253.139", 80),
    ("117.34.114.13", 7709),
    ("218.106.92.182", 7709),
    ("59.36.5.11", 7709),
]


def _probe_tdx(
    ip: str,
    port: int,
    timeout: float = _TDX_PROBE_TIMEOUT_SECONDS,
) -> bool:
    """Verify that a TDX endpoint answers an actual quote-protocol request.

    A plain TCP handshake is not sufficient: retired TDX nodes often keep port
    7709 open while every market-data request times out.  That was the root
    cause of repeatedly caching an unusable endpoint in the Streamlit process.
    """
    from tdxpy.hq import TdxHq_API

    api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
    try:
        with api.connect(ip, port, time_out=timeout):
            count = api.get_security_count(0)
            bars = api.get_security_bars(9, 1, "600000", 0, 1)
            return bool(count and bars)
    except Exception:
        return False


def _tdx_server_candidates() -> list[tuple[str, int]]:
    """Build a deduplicated TDX server pool from overrides and package data."""
    candidates: list[tuple[str, int]] = []

    # Optional emergency override without a code release:
    # TRADINGAGENTS_TDX_SERVERS=1.2.3.4:7709,5.6.7.8:80
    for value in os.getenv("TRADINGAGENTS_TDX_SERVERS", "").split(","):
        host, separator, port = value.strip().rpartition(":")
        if not separator or not host:
            continue
        try:
            candidates.append((host, int(port)))
        except ValueError:
            logger.warning("Ignoring invalid TRADINGAGENTS_TDX_SERVERS entry: %s", value)

    candidates.extend(_TDX_SERVERS)

    try:
        from tdxpy.constants import hq_hosts

        candidates.extend((str(item[1]), int(item[2])) for item in hq_hosts)
    except Exception:
        logger.debug("Could not load tdxpy host list", exc_info=True)

    try:
        from mootdx.consts import HQ_HOSTS

        candidates.extend((str(item[1]), int(item[2])) for item in HQ_HOSTS)
    except Exception:
        logger.debug("Could not load mootdx host list", exc_info=True)

    return list(dict.fromkeys(candidates))


def _available_tdx_servers(
    candidates: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Exclude nodes quarantined after a failed market-data call."""
    now = time.monotonic()
    expired = [server for server, until in _tdx_quarantined_until.items() if until <= now]
    for server in expired:
        del _tdx_quarantined_until[server]
    return [server for server in candidates if server not in _tdx_quarantined_until]


def _probe_tdx_tier(candidates: list[tuple[str, int]]) -> tuple[str, int] | None:
    if not candidates:
        return None
    with ThreadPoolExecutor(max_workers=min(24, len(candidates))) as executor:
        healthy = list(executor.map(lambda server: _probe_tdx(*server), candidates))
    return next(
        (server for server, is_healthy in zip(candidates, healthy) if is_healthy),
        None,
    )


def _find_working_tdx_server() -> tuple[str, int] | None:
    """Return a protocol-healthy server from the fast or complete pool.

    The small first tier keeps normal startup near one probe timeout.  Only if
    all preferred nodes fail do we scan the complete package-supplied pool.
    """
    preferred = _available_tdx_servers(list(_TDX_SERVERS))
    server = _probe_tdx_tier(preferred)
    if server is not None:
        return server

    preferred_set = set(_TDX_SERVERS)
    fallback = _available_tdx_servers(
        [server for server in _tdx_server_candidates() if server not in preferred_set]
    )
    return _probe_tdx_tier(fallback)


def _close_mootdx_client(*, quarantine: bool = False) -> None:
    """Close and forget the cached TDX connection."""
    global _mootdx_client
    client, _mootdx_client = _mootdx_client, None
    if client is not None:
        server = getattr(client, "server", None)
        if quarantine and isinstance(server, (tuple, list)) and len(server) == 2:
            _tdx_quarantined_until[(str(server[0]), int(server[1]))] = (
                time.monotonic() + _TDX_RETRY_COOLDOWN_SECONDS
            )
        try:
            client.close()
        except Exception:
            logger.debug("Ignoring mootdx close failure", exc_info=True)


def _get_mootdx_client():
    """Return a protocol-verified mootdx client or fail fast during cooldown.

    The client is shared because creating it is expensive, but every request is
    serialized by :func:`_call_mootdx`; tdxpy's socket is not thread-safe.
    """
    global _mootdx_client, _mootdx_unavailable_until

    with _mootdx_lock:
        if _mootdx_client is not None:
            try:
                if not _mootdx_client.closed:
                    return _mootdx_client
            except Exception:
                pass
            _close_mootdx_client(quarantine=True)

        remaining = _mootdx_unavailable_until - time.monotonic()
        if remaining > 0:
            raise RuntimeError(
                f"通达信行情服务暂不可用，{remaining:.0f} 秒后再探测；"
                "本次将直接使用 HTTP 备用数据源。"
            )

        server = _find_working_tdx_server()
        if server is None:
            _mootdx_unavailable_until = (
                time.monotonic() + _TDX_RETRY_COOLDOWN_SECONDS
            )
            raise RuntimeError(
                "内置通达信服务器的 TCP 端口可达，但行情协议均无响应；"
                "已暂停重试 5 分钟并切换到 HTTP 备用数据源。"
            )

        from mootdx.quotes import Quotes

        try:
            _mootdx_client = Quotes.factory(
                market="std",
                server=server,
                timeout=3,
                heartbeat=False,
                auto_retry=False,
                raise_exception=True,
            )
        except Exception as exc:
            _tdx_quarantined_until[server] = (
                time.monotonic() + _TDX_RETRY_COOLDOWN_SECONDS
            )
            _mootdx_unavailable_until = (
                time.monotonic() + _TDX_RETRY_COOLDOWN_SECONDS
            )
            raise RuntimeError(f"连接通达信服务器 {server[0]}:{server[1]} 失败：{exc}") from exc

        _mootdx_unavailable_until = 0.0
        return _mootdx_client


def _is_empty_mootdx_result(value) -> bool:
    if value is None:
        return True
    if isinstance(value, pd.DataFrame):
        return value.empty
    if isinstance(value, str):
        return not value.strip()
    return False


def _call_mootdx(method: str, *, allow_empty: bool = False, **kwargs):
    """Call mootdx safely, reconnecting once after a broken socket.

    LangGraph may execute tool calls on multiple worker threads.  Serializing
    access prevents request/response bytes from interleaving on the shared raw
    TDX socket.  A failed retry opens the circuit so the rest of the analysis
    falls back immediately instead of repeatedly waiting on the same dead node.
    """
    global _mootdx_unavailable_until

    last_error: Exception | None = None
    with _mootdx_lock:
        for attempt in range(2):
            try:
                client = _get_mootdx_client()
                result = getattr(client, method)(**kwargs)
                if not allow_empty and _is_empty_mootdx_result(result):
                    raise RuntimeError(f"mootdx.{method} 返回空数据")
                return result
            except Exception as exc:
                last_error = exc
                _close_mootdx_client(quarantine=True)
                if attempt == 0 and _mootdx_unavailable_until <= time.monotonic():
                    logger.info("mootdx.%s failed; reconnecting once: %s", method, exc)
                    continue
                break

        _mootdx_unavailable_until = time.monotonic() + _TDX_RETRY_COOLDOWN_SECONDS
        raise RuntimeError(f"mootdx.{method} 调用失败：{last_error}") from last_error


def _log_mootdx_fallback(operation: str, code: str, exc: Exception) -> None:
    """Log one warning per outage; cooldown fallbacks stay at INFO level."""
    message = str(exc)
    log = (
        logger.info
        if "暂不可用" in message or "已暂停重试" in message
        else logger.warning
    )
    log("mootdx %s failed for %s: %s; using HTTP fallback", operation, code, exc)


def _mootdx_kline(code: str, offset: int = 800) -> tuple[pd.DataFrame, str]:
    """Fetch stock bars, falling back to the index endpoint for index symbols."""
    df = _call_mootdx(
        "bars",
        allow_empty=True,
        symbol=code,
        frequency=9,
        offset=offset,
    )
    if not _is_empty_mootdx_result(df):
        return df, "mootdx (TCP)"

    # Codes such as 000300 have no security-bars rows but do have index bars.
    df = _call_mootdx(
        "index_bars",
        allow_empty=True,
        symbol=code,
        frequency=9,
        offset=offset,
    )
    if _is_empty_mootdx_result(df):
        raise ValueError(f"No stock or index OHLCV data from mootdx for {code}")
    return df, "mootdx index (TCP)"


# ---------------------------------------------------------------------------
# Tencent Finance API
# ---------------------------------------------------------------------------

def _tencent_quote(codes: list[str]) -> dict[str, dict]:
    """Batch real-time quotes from Tencent Finance (qt.gtimg.cn).

    Returns dict[code] -> {name, price, pe_ttm, pb, mcap_yi, ...}
    """
    prefixed = [f"{_get_prefix(c)}{c}" for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    resp = urllib.request.urlopen(req, timeout=10)
    raw = resp.read().decode("gbk")

    result = {}
    for line in raw.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        vals = line.split('"')[1].split("~")
        if len(vals) < 53:
            continue
        code = key[2:]  # strip sh/sz/bj prefix
        result[code] = {
            "name": vals[1],
            "price": float(vals[3]) if vals[3] else 0,
            "last_close": float(vals[4]) if vals[4] else 0,
            "open": float(vals[5]) if vals[5] else 0,
            "change_pct": float(vals[32]) if vals[32] else 0,
            "high": float(vals[33]) if vals[33] else 0,
            "low": float(vals[34]) if vals[34] else 0,
            "turnover_pct": float(vals[38]) if vals[38] else 0,
            "pe_ttm": float(vals[39]) if vals[39] else 0,
            # Tencent field 44 is circulating market cap and field 45 is total
            # market cap (both in 100M CNY).  The old mapping was reversed.
            "mcap_yi": float(vals[45]) if vals[45] else 0,
            "float_mcap_yi": float(vals[44]) if vals[44] else 0,
            "pb": float(vals[46]) if vals[46] else 0,
            "limit_up": float(vals[47]) if vals[47] else 0,
            "limit_down": float(vals[48]) if vals[48] else 0,
            "pe_static": float(vals[52]) if vals[52] else 0,
        }
    return result


# ---------------------------------------------------------------------------
# Eastmoney Datacenter unified helper (龙虎榜/解禁 etc.)
# ---------------------------------------------------------------------------

_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


# ---------------------------------------------------------------------------
# 东财防封：全局节流 + 会话复用 (Eastmoney anti-ban: throttle + Keep-Alive)
# ---------------------------------------------------------------------------
# 东财系 HTTP 接口（push2 / push2his / datacenter-web / search-api / np-weblist）
# 有风控：每秒 >5 次 / 单 IP 并发 ≥10 / 1 分钟 ≥200 次 / 5 分钟 ≥300 次 → 临时封 IP。
# 多 Agent 投研跑批量分析时会高频请求东财，是被封的头号元凶。所有 eastmoney.com
# 请求一律走 _em_get()：串行限流（最小间隔 + 随机抖动）+ 复用 Keep-Alive 会话 + 默认 UA。
# 注意：仅东财接口走此入口；mootdx(TCP) / 腾讯 / 新浪 / 同花顺 / 财联社 / 百度 等
# 不限流（实测不封 IP 或风控极弱）。批量任务可调大 EM_MIN_INTERVAL 进一步降速。
_EM_SESSION = _requests.Session()
_EM_SESSION.headers.update({"User-Agent": _UA})
_EM_DIRECT_SESSION = _requests.Session()
_EM_DIRECT_SESSION.trust_env = False
_EM_DIRECT_SESSION.headers.update({"User-Agent": _UA})
_em_lock = threading.Lock()
# 两次东财请求最小间隔(秒)；批量多 Agent 场景可设环境变量 EM_MIN_INTERVAL=1.5~2 降速。
_EM_MIN_INTERVAL = float(os.environ.get("EM_MIN_INTERVAL", "1.0"))
_em_last_call = [0.0]  # 模块级上次东财请求时间戳


def _em_get(url, params=None, headers=None, timeout=15, **kwargs):
    """东财统一请求入口：自动节流 + 代理/直连/备用域名故障转移。

    所有 eastmoney.com 接口都应通过它请求，避免多 Agent 高频拉数据被封 IP。
    串行限流：与上次东财请求间隔 < EM_MIN_INTERVAL 时 sleep 补足 + 0.1~0.5s 随机抖动。
    传入的 headers 会覆盖 session 默认 UA（用于保留各端点自己的 Referer/Origin）。
    """
    urls = [url]
    if isinstance(url, str) and "://push2.eastmoney.com/" in url:
        # push2 主站在部分网络/出口 IP 上会直接断开连接；delay 站提供相同 API。
        urls.append(
            url.replace(
                "://push2.eastmoney.com/",
                "://push2delay.eastmoney.com/",
                1,
            )
        )

    with _em_lock:
        wait = _EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
        if wait > 0:
            time.sleep(wait + random.uniform(0.1, 0.5))
        try:
            last_error = None
            for candidate in urls:
                try:
                    response = _EM_SESSION.get(
                        candidate,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                        **kwargs,
                    )
                    if candidate != url:
                        logger.info(
                            "Eastmoney request recovered via backup endpoint: %s",
                            candidate,
                        )
                    return response
                except _requests.exceptions.ProxyError as proxy_error:
                    last_error = proxy_error
                    try:
                        response = _EM_DIRECT_SESSION.get(
                            candidate,
                            params=params,
                            headers=headers,
                            timeout=timeout,
                            **kwargs,
                        )
                        logger.info(
                            "Eastmoney proxy unavailable; request recovered directly: %s",
                            candidate,
                        )
                        return response
                    except _requests.exceptions.RequestException as direct_error:
                        last_error = direct_error
                except _requests.exceptions.RequestException as request_error:
                    last_error = request_error

                if candidate != urls[-1]:
                    logger.info(
                        "Eastmoney primary endpoint unavailable; trying backup endpoint"
                    )

            if last_error is not None:
                raise last_error
            raise RuntimeError("Eastmoney request failed without an exception")
        finally:
            _em_last_call[0] = time.time()


def _eastmoney_datacenter(
    report_name: str,
    columns: str = "ALL",
    filter_str: str = "",
    page_size: int = 50,
    sort_columns: str = "",
    sort_types: str = "-1",
) -> list[dict]:
    """东财数据中心统一查询 — 龙虎榜/解禁 共用."""
    params = {
        "reportName": report_name,
        "columns": columns,
        "filter": filter_str,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
        "source": "WEB",
        "client": "WEB",
    }
    r = _em_get(_DATACENTER_URL, params=params, timeout=15)
    d = r.json()
    if d.get("result") and d["result"].get("data"):
        return d["result"]["data"]
    return []


def _latest_eastmoney_financial_indicators(
    code: str, curr_date: str | None
) -> dict:
    """Return the latest disclosed Eastmoney financial indicator snapshot.

    Both report date and notice date are checked so historical analyses cannot
    accidentally see a report that had not been published by ``curr_date``.
    """
    market_suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[_get_prefix(code)]
    rows = _eastmoney_datacenter(
        "RPT_F10_FINANCE_MAINFINADATA",
        filter_str=f'(SECUCODE="{code}.{market_suffix}")',
        page_size=20,
        sort_columns="REPORT_DATE",
        sort_types="-1",
    )
    cutoff = pd.to_datetime(curr_date, errors="coerce") if curr_date else None
    for row in rows:
        report_date = pd.to_datetime(row.get("REPORT_DATE"), errors="coerce")
        notice_date = pd.to_datetime(row.get("NOTICE_DATE"), errors="coerce")
        if cutoff is not None and not pd.isna(cutoff):
            if not pd.isna(report_date) and report_date > cutoff:
                continue
            if not pd.isna(notice_date) and notice_date > cutoff:
                continue
        return row
    return {}


# ---------------------------------------------------------------------------
# 同花顺 EPS forecast helper (direct HTTP, no akshare)
# ---------------------------------------------------------------------------


def _ths_eps_forecast(code: str) -> pd.DataFrame:
    """Fetch consensus EPS forecast from 同花顺 (direct HTTP).

    Returns DataFrame with columns roughly: 年度, 预测机构数, 最小值, 均值, 最大值.
    """
    url = f"https://basic.10jqka.com.cn/new/{code}/worth.html"
    headers = {
        "User-Agent": _UA,
        "Referer": "https://basic.10jqka.com.cn/",
    }
    r = _requests.get(url, headers=headers, timeout=15)
    r.encoding = "gbk"
    dfs = pd.read_html(StringIO(r.text))
    # Find the table containing EPS data
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if any("每股收益" in c or "均值" in c for c in cols):
            return df
    # Fallback: return first table if exists
    return dfs[0] if dfs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Sina K-line fallback helper (direct HTTP, no akshare)
# ---------------------------------------------------------------------------


def _sina_kline_fallback(code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch daily K-line from Sina HTTP API as mootdx fallback.

    Sina separates stock and index symbols by exchange prefix.  Some index
    codes (notably CSI 300 / 000300) overlap Shenzhen's six-digit stock-code
    range, so try the normal stock prefix first and then Shanghai's index
    prefix when the first response is empty.

    Returns DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    primary_prefix = "sh" if code.startswith("6") else "sz"
    symbols = [f"{primary_prefix}{code}"]
    if primary_prefix == "sz" and code.startswith(("000", "399")):
        symbols.append(f"sh{code}")

    url = (
        "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "CN_MarketData.getKLineData"
    )

    data = None
    last_error: Exception | None = None
    for symbol in symbols:
        try:
            r = _requests.get(
                url,
                params={
                    "symbol": symbol,
                    "scale": "240",  # daily
                    "ma": "no",
                    "datalen": "800",
                },
                timeout=15,
            )
            r.raise_for_status()
            data = _json.loads(r.text)
        except Exception as exc:
            last_error = exc
            continue
        if data:
            break

    if not data:
        if last_error is not None:
            raise last_error
        return pd.DataFrame()

    rows = []
    for item in data:
        rows.append({
            "Date": item["day"],
            "Open": float(item["open"]),
            "High": float(item["high"]),
            "Low": float(item["low"]),
            "Close": float(item["close"]),
            "Volume": int(item["volume"]),
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    return df


def _last_ohlcv_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """Return the latest OHLCV Date in a normalized dataframe."""
    if df is None or df.empty or "Date" not in df.columns:
        return None
    dates = pd.to_datetime(df["Date"], errors="coerce")
    if dates.dropna().empty:
        return None
    return dates.max().normalize()


def _normalize_ohlcv_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV Date values to daily granularity."""
    if df is None or df.empty or "Date" not in df.columns:
        return df
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["Date"])


def _needs_sina_supplement(df: pd.DataFrame, target_date: str | None) -> bool:
    """True when mootdx/cache data is older than the requested cutoff date."""
    if not target_date:
        return False
    last_date = _last_ohlcv_date(df)
    if last_date is None:
        return True
    target = pd.to_datetime(target_date).normalize()
    return last_date < target


def _merge_ohlcv(primary: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    """Merge OHLCV frames, preferring supplement rows on duplicate dates."""
    frames = [frame for frame in (primary, supplement) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    combined = pd.concat(frames, ignore_index=True)
    combined = _normalize_ohlcv_dates(combined)
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined


def _supplement_stale_ohlcv_with_sina(
    code: str,
    df: pd.DataFrame,
    target_date: str | None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Use Sina daily K-line to fill dates missing from mootdx/cache data."""
    if not _needs_sina_supplement(df, target_date):
        return df, False
    try:
        sina_df = _sina_kline_fallback(code, start_date, target_date)
    except Exception as e:
        logger.warning("sina K-line supplement failed for %s: %s", code, e)
        return df, False
    if sina_df.empty:
        return df, False
    merged = _merge_ohlcv(df, sina_df)
    return merged, _last_ohlcv_date(merged) != _last_ohlcv_date(df)


# ---------------------------------------------------------------------------
# OHLCV loading with cache (mootdx -> CSV)
# ---------------------------------------------------------------------------

def _load_ohlcv_astock(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV via mootdx, cache to CSV, filter by curr_date.

    Mirrors stockstats_utils.load_ohlcv but uses mootdx instead of yfinance.
    Returns DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    from .config import get_config

    code = _normalize_ticker(symbol)
    config = get_config()
    cache_dir = config.get(
        "data_cache_dir", os.path.expanduser("~/.tradingagents/cache")
    )
    cache_file: str | None = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{code}-astock-daily.csv")
    except OSError as exc:
        logger.info("OHLCV cache directory unavailable; using memory only: %s", exc)

    if cache_file and os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if mtime.date() == datetime.now().date():
            data = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
            data = _normalize_ohlcv_dates(data)
            data, supplemented = _supplement_stale_ohlcv_with_sina(
                code, data, curr_date, start_date=None
            )
            if supplemented:
                try:
                    data.to_csv(cache_file, index=False, encoding="utf-8")
                except OSError as exc:
                    logger.info(
                        "OHLCV cache is read-only; continuing with memory data: %s",
                        exc,
                    )
            cutoff = pd.to_datetime(curr_date)
            return data[data["Date"] <= cutoff]

    # Fetch from mootdx — 800 daily bars (~3 years of trading days)
    try:
        df, _ = _mootdx_kline(code, offset=800)

        if df is None or df.empty:
            raise ValueError(f"No OHLCV data from mootdx for {code}")

        # mootdx returns index named 'datetime' AND a column named 'datetime'
        # (plus year/month/day/hour/minute/volume). Drop duplicates before reset.
        df = df.drop(columns=["datetime", "year", "month", "day", "hour", "minute"], errors="ignore")
        df = df.reset_index()  # moves index 'datetime' → column 'datetime'
        rename_map = {
            "datetime": "Date",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }
        df = df.rename(columns=rename_map)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = _normalize_ohlcv_dates(df)
    except Exception as e:
        _log_mootdx_fallback("OHLCV", code, e)
        # Fallback: Sina direct HTTP API
        try:
            df = _sina_kline_fallback(code)
            if df.empty:
                raise ValueError(f"No OHLCV data from sina for {code}")
        except Exception:
            raise ValueError(f"No OHLCV data from mootdx/sina for {code}")

    df, _ = _supplement_stale_ohlcv_with_sina(code, df, curr_date, start_date=None)

    # Cache to disk
    if cache_file:
        try:
            df.to_csv(cache_file, index=False, encoding="utf-8")
        except OSError as exc:
            logger.info(
                "OHLCV cache is read-only; continuing with memory data: %s",
                exc,
            )

    # Filter by curr_date to prevent look-ahead bias
    cutoff = pd.to_datetime(curr_date)
    return df[df["Date"] <= cutoff]


# ===========================================================================
# 9 Vendor Methods (matching interface.py VENDOR_METHODS signatures)
# ===========================================================================


# ---- 1. get_stock_data ----


def get_stock_data(
    symbol: Annotated[str, "A-stock code (e.g. 688017, SH688017)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock price data via mootdx."""
    code = _normalize_ticker(symbol)

    try:
        df, data_source = _mootdx_kline(code, offset=800)

        if df is None or df.empty:
            raise ValueError(f"No data from mootdx for {code}")

        # Drop duplicate datetime column + extra columns before reset_index
        df = df.drop(
            columns=["datetime", "year", "month", "day", "hour", "minute"],
            errors="ignore",
        )
        df = df.reset_index()  # index 'datetime' → column 'datetime'
        df = df.rename(
            columns={
                "datetime": "Date",
                "open": "Open",
                "close": "Close",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
                "amount": "Amount",
            }
        )
        df = _normalize_ohlcv_dates(df)

    except Exception as e:
        _log_mootdx_fallback("K-line", code, e)
        # Fallback: Sina direct HTTP API
        try:
            df = _sina_kline_fallback(code, start_date, end_date)
            if df.empty:
                return "K线数据获取失败：mootdx和新浪备用源均不可用，请检查网络连接"
            data_source = "sina HTTP (fallback)"
        except Exception:
            return "K线数据获取失败：mootdx和新浪备用源均不可用，请检查网络连接"

    df, supplemented = _supplement_stale_ohlcv_with_sina(code, df, end_date, start_date)
    if supplemented:
        data_source = f"{data_source} + sina HTTP supplement"

    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]

    if df.empty:
        return (
            f"No data found for A-stock '{code}' "
            f"between {start_date} and {end_date}"
        )

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    csv_out = df[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(
        index=False
    )

    header = f"# Stock data for {code} (A-stock) from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data source: {data_source}\n"
    header += (
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    return header + csv_out


# ---- 2. get_indicators ----

# Supported technical indicators with descriptions
_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50 SMA: Medium-term trend indicator.",
    "close_200_sma": "200 SMA: Long-term trend benchmark.",
    "close_10_ema": "10 EMA: Responsive short-term average.",
    "macd": "MACD: Momentum via EMA differences.",
    "macds": "MACD Signal: EMA smoothing of MACD line.",
    "macdh": "MACD Histogram: Gap between MACD and signal.",
    "rsi": "RSI: Momentum overbought/oversold indicator (70/30 thresholds).",
    "boll": "Bollinger Middle: 20 SMA basis for Bollinger Bands.",
    "boll_ub": "Bollinger Upper Band: 2 std devs above middle.",
    "boll_lb": "Bollinger Lower Band: 2 std devs below middle.",
    "atr": "ATR: Average True Range volatility measure.",
    "vwma": "VWMA: Volume-weighted moving average.",
    "mfi": "MFI: Money Flow Index (volume + price momentum).",
}


def get_indicators(
    symbol: Annotated[str, "A-stock code"],
    indicator: Annotated[
        str, "technical indicator (e.g. rsi, macd, close_50_sma)"
    ],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Get technical indicators using stockstats on mootdx OHLCV data."""
    from stockstats import wrap

    code = _normalize_ticker(symbol)

    if indicator not in _INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} not supported. "
            f"Choose from: {list(_INDICATOR_DESCRIPTIONS.keys())}"
        )

    try:
        data = _load_ohlcv_astock(code, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        # Trigger stockstats calculation
        df[indicator]

        # Build date -> value lookup
        ind_dict = {}
        for _, row in df.iterrows():
            d = row["Date"]
            v = row[indicator]
            ind_dict[d] = "N/A" if pd.isna(v) else str(round(float(v), 4))

        # Generate output for look_back window
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        before = curr_dt - relativedelta(days=look_back_days)

        lines = []
        dt = curr_dt
        while dt >= before:
            ds = dt.strftime("%Y-%m-%d")
            val = ind_dict.get(ds, "N/A: Not a trading day (weekend or holiday)")
            lines.append(f"{ds}: {val}")
            dt -= relativedelta(days=1)

        result = (
            f"## {indicator} values for {code} "
            f"from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + "\n".join(lines)
            + "\n\n"
            + _INDICATOR_DESCRIPTIONS.get(indicator, "")
        )
        return result

    except Exception as e:
        return f"Error calculating {indicator} for {code}: {str(e)}"


# ---- 3. get_fundamentals ----


def get_fundamentals(
    ticker: Annotated[str, "A-stock code"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get company fundamentals from Tencent + mootdx + Eastmoney + 同花顺."""
    code = _normalize_ticker(ticker)

    try:
        lines = []

        # --- Tencent: real-time valuation ---
        try:
            tq = _tencent_quote([code])
            if code in tq:
                q = tq[code]
                lines.extend(
                    [
                        f"Name: {q['name']}",
                        f"Price: {q['price']}",
                        f"PE (TTM): {q['pe_ttm']}",
                        f"PE (Static): {q['pe_static']}",
                        f"PB: {q['pb']}",
                        f"Market Cap (100M CNY): {q['mcap_yi']}",
                        f"Float Market Cap (100M CNY): {q['float_mcap_yi']}",
                        f"Turnover Rate: {q['turnover_pct']}%",
                        f"Change: {q['change_pct']}%",
                        f"Limit Up: {q['limit_up']}",
                        f"Limit Down: {q['limit_down']}",
                    ]
                )
        except Exception as e:
            logger.warning("Tencent quote failed for %s: %s", code, e)

        # --- mootdx: financial snapshot (quarterly) ---
        try:
            fin = _call_mootdx("finance", symbol=code)
            if fin is not None and not (
                isinstance(fin, pd.DataFrame) and fin.empty
            ):
                row = fin.iloc[0] if isinstance(fin, pd.DataFrame) else fin
                field_map = {
                    "eps": "EPS (Quarterly)",
                    "bvps": "Book Value Per Share",
                    "roe": "ROE (%)",
                    "profit": "Net Profit",
                    "income": "Revenue",
                    "liutongguben": "Float Shares",
                    "zongguben": "Total Shares",
                }
                idx = row.index if hasattr(row, "index") else []
                for field, label in field_map.items():
                    if field in idx:
                        val = row[field]
                        if val is not None and str(val) != "nan":
                            lines.append(f"{label}: {val}")
        except Exception as e:
            _log_mootdx_fallback("finance", code, e)

        # --- Eastmoney push2: basic stock info (direct HTTP) ---
        try:
            market_code = 1 if code.startswith("6") else 0
            _info_url = "https://push2.eastmoney.com/api/qt/stock/get"
            _info_params = {
                "fltt": "2",
                "invt": "2",
                "fields": "f57,f58,f84,f85,f127,f116,f117,f189,f43",
                "secid": f"{market_code}.{code}",
            }
            r = _em_get(_info_url, params=_info_params, timeout=10)
            d = r.json().get("data", {})
            if d:
                if d.get("f127"):
                    lines.append(f"行业: {d['f127']}")
                if d.get("f84"):
                    lines.append(f"总股本: {d['f84']}")
                if d.get("f85"):
                    lines.append(f"流通股本: {d['f85']}")
                if d.get("f116"):
                    lines.append(f"总市值: {d['f116']}")
                if d.get("f117"):
                    lines.append(f"流通市值: {d['f117']}")
                if d.get("f189"):
                    lines.append(f"上市日期: {d['f189']}")
        except Exception as e:
            logger.warning("eastmoney push2 stock info failed for %s: %s", code, e)

        # --- Eastmoney datacenter: disclosed financial quality indicators ---
        try:
            fin_main = _latest_eastmoney_financial_indicators(code, curr_date)
            if fin_main:
                report_date = str(fin_main.get("REPORT_DATE", ""))[:10]
                lines.append("\n--- Latest Disclosed Financial Indicators (东财) ---")
                if report_date:
                    lines.append(f"Financial Report Date: {report_date}")
                financial_fields = (
                    ("TOTALOPERATEREVE", "Revenue"),
                    ("TOTALOPERATEREVETZ", "Revenue YoY (%)"),
                    ("PARENTNETPROFIT", "Net Profit Attributable to Parent"),
                    ("PARENTNETPROFITTZ", "Parent Net Profit YoY (%)"),
                    ("KCFJCXSYJLR", "Deducted Parent Net Profit"),
                    ("KCFJCXSYJLRTZ", "Deducted Net Profit YoY (%)"),
                    ("ROEJQ", "Weighted ROE (%)"),
                    ("XSMLL", "Gross Margin (%)"),
                    ("XSJLL", "Net Margin (%)"),
                    ("ZCFZL", "Debt-to-Asset Ratio (%)"),
                    ("TOTAL_ASSETS_PK", "Total Assets"),
                    ("LIABILITY", "Total Liabilities"),
                    ("NETCASH_OPERATE_PK", "Operating Cash Flow"),
                    ("NCO_NETPROFIT", "Operating Cash Flow / Net Profit"),
                )
                for field, label in financial_fields:
                    value = fin_main.get(field)
                    if value is not None and str(value) != "nan":
                        lines.append(f"{label}: {value}")
        except Exception as e:
            logger.info(
                "Eastmoney financial indicator snapshot unavailable for %s: %s",
                code,
                e,
            )

        # --- 同花顺 direct HTTP: consensus EPS forecast ---
        try:
            forecast_df = _ths_eps_forecast(code)
            if forecast_df is not None and not forecast_df.empty:
                lines.append("\n--- Consensus EPS Forecast (同花顺) ---")
                eps_by_year = {}
                for _, row in forecast_df.iterrows():
                    year = str(row.iloc[0]) if len(row) > 0 else ""
                    mean_eps_val = row.iloc[3] if len(row) > 3 else 0
                    count_val = row.iloc[1] if len(row) > 1 else 0
                    min_eps_val = row.iloc[2] if len(row) > 2 else "N/A"
                    max_eps_val = row.iloc[4] if len(row) > 4 else "N/A"
                    try:
                        mean_eps = float(mean_eps_val)
                    except (ValueError, TypeError):
                        mean_eps = 0
                    try:
                        count = int(count_val)
                    except (ValueError, TypeError):
                        count = 0
                    lines.append(
                        f"FY{year}: EPS={mean_eps} "
                        f"(range {min_eps_val}~{max_eps_val}, {count} analysts)"
                    )
                    if count < 3:
                        lines.append("  Warning: low coverage (<3 analysts)")
                    eps_by_year[year] = mean_eps

                # Forward PE / PEG / PE digestion
                try:
                    tq = _tencent_quote([code])
                    if code in tq:
                        price = tq[code]["price"]
                        years_sorted = sorted(eps_by_year.keys())
                        if years_sorted and eps_by_year.get(years_sorted[0], 0) > 0:
                            eps_cur = eps_by_year[years_sorted[0]]
                            fwd_pe = price / eps_cur
                            lines.append(
                                f"\nForward PE (FY{years_sorted[0]}): "
                                f"{fwd_pe:.1f}x (price={price}, EPS={eps_cur})"
                            )
                            if (
                                len(years_sorted) >= 2
                                and eps_by_year.get(years_sorted[1], 0) > 0
                            ):
                                eps_next = eps_by_year[years_sorted[1]]
                                cagr = eps_next / eps_cur - 1
                                if cagr > 0:
                                    peg = fwd_pe / (cagr * 100)
                                    lines.append(
                                        f"PEG: {peg:.2f} "
                                        f"(EPS CAGR={cagr * 100:.0f}%)"
                                    )
                                    if fwd_pe > 30:
                                        digest = math.log(fwd_pe / 30) / math.log(
                                            1 + cagr
                                        )
                                        lines.append(
                                            f"PE Digestion to 30x: {digest:.1f} years"
                                        )
                                    else:
                                        lines.append("PE already below 30x target")
                                else:
                                    lines.append(
                                        f"EPS declining ({cagr * 100:.0f}%), "
                                        f"PEG not applicable"
                                    )
                except Exception as e:
                    logger.warning("Forward PE calc failed for %s: %s", code, e)
        except Exception as e:
            logger.warning("Consensus EPS forecast failed for %s: %s", code, e)

        if not lines:
            return f"No fundamentals data found for A-stock '{code}'"

        header = f"# Company Fundamentals for {code} (A-stock)\n"
        header += (
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {code}: {str(e)}"


# ---- 4. get_balance_sheet ----


def _sina_stock_code(code: str) -> str:
    """Pure 6-digit code → sina format (sh688017 / sz000001 / bj832000)."""
    return f"{_get_prefix(code)}{code}"


_SINA_FINANCIAL_FIELDS = {
    "fzb": (
        "CURFDS",
        "TRADFINASSET",
        "NOTESACCORECE",
        "INVE",
        "TOTCURRASSET",
        "FIXEDASSECLEATOT",
        "CONSPROGTOT",
        "INTAASSET",
        "GOODWILL",
        "TOTALNONCASSETS",
        "TOTASSET",
        "SHORTTERMBORR",
        "NOTESACCOPAYA",
        "TOTALCURRLIAB",
        "LONGBORR",
        "BDSPAYA",
        "TOTALNONCLIAB",
        "TOTLIAB",
        "PAIDINCAPI",
        "PARESHARRIGH",
        "RIGHAGGR",
    ),
    "lrb": (
        "BIZTOTINCO",
        "BIZINCO",
        "BIZTOTCOST",
        "BIZCOST",
        "DEVEEXPE",
        "SALESEXPE",
        "MANAEXPE",
        "FINEXPE",
        "INVEINCO",
        "PERPROFIT",
        "TOTPROFIT",
        "INCOTAXEXPE",
        "NETPROFIT",
        "PARENETP",
        "BASICEPS",
        "DILUTEDEPS",
    ),
    "llb": (
        "LABORGETCASH",
        "BIZCASHINFL",
        "BIZCASHOUTF",
        "MANANETR",
        "INVCASHINFL",
        "ACQUASSETCASH",
        "INVCASHOUTF",
        "INVNETCASHFLOW",
        "FINCASHINFL",
        "FINCASHOUTF",
        "FINNETCFLOW",
        "CASHNETR",
        "INICASHBALA",
        "FINALCASHBALA",
    ),
}


def _parse_sina_financial_reports(
    result: dict,
    source_type: str,
    freq: str,
    curr_date: str | None,
) -> pd.DataFrame:
    """Normalize Sina's current nested report_list response into report rows."""
    # Compatibility with the older API shape used before Sina's 2022 endpoint
    # migration.  It returned a ready-made list under fzb/lrb/llb.
    legacy_items = result.get(source_type, [])
    if isinstance(legacy_items, list) and legacy_items:
        legacy_df = pd.DataFrame(legacy_items)
        if curr_date and "报告日" in legacy_df.columns:
            report_dates = pd.to_datetime(legacy_df["报告日"], errors="coerce")
            legacy_df = legacy_df[report_dates <= pd.to_datetime(curr_date)]
        if freq.lower() == "annual" and "报告日" in legacy_df.columns:
            report_dates = pd.to_datetime(legacy_df["报告日"], errors="coerce")
            legacy_df = legacy_df[report_dates.dt.month == 12]
        return legacy_df.head(8)

    report_list = result.get("report_list", {})
    if not isinstance(report_list, dict) or not report_list:
        return pd.DataFrame()

    descriptions = {
        str(item.get("date_value", "")): item.get("date_description", "")
        for item in result.get("report_date", [])
        if isinstance(item, dict)
    }
    cutoff = pd.to_datetime(curr_date, errors="coerce") if curr_date else None
    selected_fields = set(_SINA_FINANCIAL_FIELDS[source_type])
    rows: list[dict] = []

    for date_value, report in report_list.items():
        if not isinstance(report, dict):
            continue
        report_date = pd.to_datetime(str(date_value), format="%Y%m%d", errors="coerce")
        if pd.isna(report_date):
            continue
        if cutoff is not None and not pd.isna(cutoff) and report_date > cutoff:
            continue
        if freq.lower() == "annual" and report_date.month != 12:
            continue

        publish_date = pd.to_datetime(
            str(report.get("publish_date", "")),
            format="%Y%m%d",
            errors="coerce",
        )
        if cutoff is not None and not pd.isna(cutoff):
            if not pd.isna(publish_date) and publish_date > cutoff:
                continue

        row = {
            "报告日": report_date.strftime("%Y-%m-%d"),
            "报告期": descriptions.get(str(date_value), ""),
            "公告日": "" if pd.isna(publish_date) else publish_date.strftime("%Y-%m-%d"),
            "报表口径": report.get("rType", ""),
            "币种": report.get("rCurrency", ""),
            "审计状态": report.get("is_audit", ""),
        }
        for item in report.get("data", []):
            if not isinstance(item, dict) or item.get("item_field") not in selected_fields:
                continue
            title = str(item.get("item_title") or "").strip()
            value = item.get("item_value")
            if not title or value in (None, ""):
                continue
            try:
                row[title] = float(value)
            except (TypeError, ValueError):
                row[title] = value

            yoy = item.get("item_tongbi")
            if yoy not in (None, ""):
                try:
                    row[f"{title}同比"] = f"{float(yoy) * 100:.2f}%"
                except (TypeError, ValueError):
                    row[f"{title}同比"] = yoy
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("报告日", ascending=False).head(8)


def _get_financial_report_sina(
    code: str, report_type: str, freq: str, curr_date: str = None,
) -> pd.DataFrame:
    """Shared helper: fetch financial report via Sina direct HTTP API.

    report_type: '资产负债表' | '利润表' | '现金流量表'
    """
    _report_type_map = {
        "资产负债表": "fzb",
        "利润表": "lrb",
        "现金流量表": "llb",
    }
    source_type = _report_type_map.get(report_type, "lrb")

    prefix = "sh" if code.startswith("6") else "sz"
    paper_code = f"{prefix}{code}"
    url = "https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022"
    params = {
        "paperCode": paper_code,
        "source": source_type,
        "type": "0",
        "page": "1",
        "num": "20",
    }
    r = _requests.get(url, params=params, headers={"User-Agent": _UA}, timeout=15)
    d = r.json()

    result = d.get("result", {}).get("data", {})
    return _parse_sina_financial_reports(result, source_type, freq, curr_date)


_EASTMONEY_FINANCIAL_REPORTS = {
    "资产负债表": (
        "RPT_DMSK_FN_BALANCE",
        {
            "REPORT_DATE": "报告日",
            "NOTICE_DATE": "公告日",
            "TOTAL_ASSETS": "资产总计",
            "FIXED_ASSET": "固定资产",
            "MONETARYFUNDS": "货币资金",
            "ACCOUNTS_RECE": "应收账款",
            "INVENTORY": "存货",
            "TOTAL_LIABILITIES": "负债合计",
            "ACCOUNTS_PAYABLE": "应付账款",
            "TOTAL_EQUITY": "所有者权益合计",
            "DEBT_ASSET_RATIO": "资产负债率(%)",
        },
    ),
    "利润表": (
        "RPT_DMSK_FN_INCOME",
        {
            "REPORT_DATE": "报告日",
            "NOTICE_DATE": "公告日",
            "TOTAL_OPERATE_INCOME": "营业总收入",
            "TOI_RATIO": "营业总收入同比(%)",
            "TOTAL_OPERATE_COST": "营业总成本",
            "OPERATE_COST": "营业成本",
            "SALE_EXPENSE": "销售费用",
            "MANAGE_EXPENSE": "管理费用",
            "FINANCE_EXPENSE": "财务费用",
            "OPERATE_PROFIT": "营业利润",
            "TOTAL_PROFIT": "利润总额",
            "PARENT_NETPROFIT": "归母净利润",
            "PARENT_NETPROFIT_RATIO": "归母净利润同比(%)",
            "DEDUCT_PARENT_NETPROFIT": "扣非归母净利润",
            "DPN_RATIO": "扣非归母净利润同比(%)",
        },
    ),
    "现金流量表": (
        "RPT_DMSK_FN_CASHFLOW",
        {
            "REPORT_DATE": "报告日",
            "NOTICE_DATE": "公告日",
            "NETCASH_OPERATE": "经营活动现金流量净额",
            "NETCASH_OPERATE_RATIO": "经营活动现金流量净额同比(%)",
            "SALES_SERVICES": "销售商品提供劳务收到的现金",
            "PAY_STAFF_CASH": "支付给职工的现金",
            "NETCASH_INVEST": "投资活动现金流量净额",
            "CONSTRUCT_LONG_ASSET": "购建长期资产支付的现金",
            "NETCASH_FINANCE": "筹资活动现金流量净额",
            "CCE_ADD": "现金及现金等价物净增加额",
        },
    ),
}


def _get_financial_report_eastmoney(
    code: str, report_type: str, freq: str, curr_date: str | None
) -> pd.DataFrame:
    """Fetch a compact financial statement from Eastmoney as Sina fallback."""
    report_name, field_map = _EASTMONEY_FINANCIAL_REPORTS[report_type]
    market_suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[_get_prefix(code)]
    rows = _eastmoney_datacenter(
        report_name,
        filter_str=f'(SECUCODE="{code}.{market_suffix}")',
        page_size=20,
        sort_columns="REPORT_DATE",
        sort_types="-1",
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    report_dates = pd.to_datetime(df.get("REPORT_DATE"), errors="coerce")
    if curr_date:
        cutoff = pd.to_datetime(curr_date)
        notice_dates = pd.to_datetime(df.get("NOTICE_DATE"), errors="coerce")
        df = df[(report_dates <= cutoff) & (notice_dates <= cutoff)]
        report_dates = pd.to_datetime(df.get("REPORT_DATE"), errors="coerce")
    if freq.lower() == "annual":
        df = df[report_dates.dt.month == 12]
    available = [field for field in field_map if field in df.columns]
    return df[available].rename(columns=field_map).head(8)


def _get_financial_report(
    code: str, report_type: str, freq: str, curr_date: str | None
) -> tuple[pd.DataFrame, str]:
    """Fetch a financial report using Sina first and Eastmoney as fallback."""
    try:
        df = _get_financial_report_sina(code, report_type, freq, curr_date)
        if not df.empty:
            return df, "sina direct HTTP"
    except Exception as exc:
        logger.info("Sina %s unavailable for %s: %s", report_type, code, exc)

    df = _get_financial_report_eastmoney(code, report_type, freq, curr_date)
    if not df.empty:
        return df, "Eastmoney datacenter fallback"
    return pd.DataFrame(), "sina + Eastmoney unavailable"


def get_balance_sheet(
    ticker: Annotated[str, "A-stock code"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get balance sheet via Sina direct HTTP API."""
    code = _normalize_ticker(ticker)

    try:
        df, source = _get_financial_report(
            code, "资产负债表", freq, curr_date
        )

        if df.empty:
            return f"No balance sheet data found for A-stock '{code}'"

        csv_string = df.to_csv(index=False)

        header = f"# Balance Sheet for {code} (A-stock, {freq})\n"
        header += f"# Data source: {source}\n"
        header += (
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        return header + csv_string

    except Exception as e:
        return f"Error retrieving balance sheet for {code}: {str(e)}"


# ---- 5. get_cashflow ----


def get_cashflow(
    ticker: Annotated[str, "A-stock code"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get cash flow statement via Sina direct HTTP API."""
    code = _normalize_ticker(ticker)

    try:
        df, source = _get_financial_report(
            code, "现金流量表", freq, curr_date
        )

        if df.empty:
            return f"No cash flow data found for A-stock '{code}'"

        csv_string = df.to_csv(index=False)

        header = f"# Cash Flow for {code} (A-stock, {freq})\n"
        header += f"# Data source: {source}\n"
        header += (
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        return header + csv_string

    except Exception as e:
        return f"Error retrieving cash flow for {code}: {str(e)}"


# ---- 6. get_income_statement ----


def get_income_statement(
    ticker: Annotated[str, "A-stock code"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get income statement via Sina direct HTTP API."""
    code = _normalize_ticker(ticker)

    try:
        df, source = _get_financial_report(code, "利润表", freq, curr_date)

        if df.empty:
            return f"No income statement data found for A-stock '{code}'"

        csv_string = df.to_csv(index=False)

        header = f"# Income Statement for {code} (A-stock, {freq})\n"
        header += f"# Data source: {source}\n"
        header += (
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        return header + csv_string

    except Exception as e:
        return f"Error retrieving income statement for {code}: {str(e)}"


# ---- 7. get_news ----


def _fetch_news_eastmoney(code: str, page_size: int = 20) -> list[dict]:
    """Direct East Money search API for individual stock news."""
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    inner_param = {
        "uid": "",
        "keyword": code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": page_size,
                "preTag": "",
                "postTag": "",
            }
        },
    }
    params = {
        "cb": "callback",
        "param": _json.dumps(inner_param, ensure_ascii=False),
        "_": "1",
    }
    headers = {
        "Referer": "https://so.eastmoney.com/",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
    }

    resp = _em_get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    text = resp.text
    text = text[text.index("(") + 1 : text.rindex(")")]
    data = _json.loads(text)

    articles: list[dict] = []
    for item in data.get("result", {}).get("cmsArticleWebOld", []):
        articles.append({
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "time": item.get("date", ""),
            "source": item.get("mediaName", "东方财富"),
            "url": item.get("url", ""),
        })
    return articles


def _fetch_news_sina(code: str, page_size: int = 20) -> list[dict]:
    """Sina Finance stock news API (backup source)."""
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    url = (
        f"https://vip.stock.finance.sina.com.cn/corp/view/"
        f"vCB_AllNewsStock.php?symbol={prefix}{code}&Page=1"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
        "Referer": "https://finance.sina.com.cn/",
    }

    resp = _requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    resp.encoding = "gb2312"
    html = resp.text

    articles: list[dict] = []
    rows = _re.findall(
        r"(\d{4}-\d{2}-\d{2})\s*(?:&nbsp;)*(\d{2}:\d{2})\s*(?:&nbsp;)*"
        r"<a[^>]+href='([^']+)'[^>]*>([^<]+)</a>",
        html,
    )
    for date_str, time_str, link, title in rows[:page_size]:
        articles.append({
            "title": title.strip(),
            "content": "",
            "time": f"{date_str} {time_str}",
            "source": "新浪财经",
            "url": link,
        })
    return articles


def _fetch_announcements_eastmoney(code: str, page_size: int = 50) -> list[dict]:
    """Fetch listed-company announcements from Eastmoney's public notice API."""
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    response = _em_get(
        url,
        params={
            "sr": "-1",
            "page_size": str(page_size),
            "page_index": "1",
            "ann_type": "A",
            "client_source": "web",
            "stock_list": code,
            "f_node": "0",
            "s_node": "0",
        },
        headers={"User-Agent": _UA, "Referer": "https://data.eastmoney.com/"},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json().get("data", {}).get("list", [])
    announcements = []
    for item in rows:
        article_code = str(item.get("art_code") or "")
        columns = ", ".join(
            str(column.get("column_name") or "")
            for column in item.get("columns", [])
            if isinstance(column, dict) and column.get("column_name")
        )
        announcements.append(
            {
                "title": item.get("title") or item.get("title_ch") or "",
                "content": f"公告分类: {columns}" if columns else "上市公司公告",
                "time": str(item.get("notice_date") or item.get("display_time") or ""),
                "source": "东方财富公告",
                "url": (
                    f"https://data.eastmoney.com/notices/detail/{code}/{article_code}.html"
                    if article_code
                    else ""
                ),
            }
        )
    return announcements


def _fetch_eastmoney_guba_posts(code: str) -> tuple[list[dict], int]:
    """Fetch public Eastmoney stock-board posts embedded in the list page."""
    url = f"https://guba.eastmoney.com/list,{code}.html"
    response = _em_get(
        url,
        headers={"User-Agent": _UA, "Referer": "https://guba.eastmoney.com/"},
        timeout=15,
    )
    response.raise_for_status()
    marker = _re.search(r"var\s+article_list\s*=\s*", response.text)
    if marker is None:
        raise ValueError("Eastmoney Guba page is missing article_list")
    payload, _ = _json.JSONDecoder().raw_decode(response.text, marker.end())
    posts = payload.get("re", [])
    if not isinstance(posts, list):
        raise ValueError("Eastmoney Guba article_list has an unexpected shape")
    return posts, int(payload.get("count") or 0)


def get_social_sentiment(
    ticker: Annotated[str, "A-stock code"],
    curr_date: Annotated[str, "Current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "Days to look back"] = 7,
    limit: Annotated[int, "Maximum posts to return"] = 20,
) -> str:
    """Get a verifiable Eastmoney Guba discussion and engagement sample."""
    from collections import Counter

    code = _normalize_ticker(ticker)
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=look_back_days)
    try:
        raw_posts, board_count = _fetch_eastmoney_guba_posts(code)
    except Exception as exc:
        return f"Error fetching Eastmoney Guba sentiment for {code}: {exc}"

    posts = []
    for post in raw_posts:
        publish_time = str(post.get("post_publish_time") or "")
        published = pd.to_datetime(publish_time, errors="coerce")
        if pd.isna(published) or published < start_dt or published > end_dt + pd.Timedelta(days=1):
            continue
        posts.append(post)

    if not posts:
        return (
            f"No Eastmoney Guba posts found for {code} between "
            f"{start_dt:%Y-%m-%d} and {curr_date}"
        )

    positive_words = ("看多", "上涨", "突破", "买入", "加仓", "利好", "新高", "涨停", "牛")
    negative_words = ("看空", "下跌", "套牢", "卖出", "减仓", "利空", "风险", "跌停", "套人")
    tone_counts = Counter()
    daily_counts = Counter()
    for post in posts:
        text = f"{post.get('post_title', '')} {post.get('post_content', '')}"
        positive_hits = sum(text.count(word) for word in positive_words)
        negative_hits = sum(text.count(word) for word in negative_words)
        tone = "positive" if positive_hits > negative_hits else (
            "negative" if negative_hits > positive_hits else "neutral"
        )
        tone_counts[tone] += 1
        daily_counts[str(post.get("post_publish_time", ""))[:10]] += 1

    total_reads = sum(int(post.get("post_click_count") or 0) for post in posts)
    total_comments = sum(int(post.get("post_comment_count") or 0) for post in posts)
    total_likes = sum(int(post.get("post_like_count") or 0) for post in posts)
    ranked = sorted(
        posts,
        key=lambda post: (
            int(post.get("post_comment_count") or 0),
            int(post.get("post_click_count") or 0),
        ),
        reverse=True,
    )[: max(1, min(limit, 50))]

    lines = [
        f"# Eastmoney Guba Sentiment Sample for {code}",
        f"# Window: {start_dt:%Y-%m-%d} to {curr_date}",
        f"# Source: 东方财富股吧公开帖子；current-page sample={len(posts)}, board total={board_count}",
        "# Tone counts are keyword heuristics, not a statistically representative poll.",
        "",
        "## Sample Engagement",
        f"Posts={len(posts)} Reads={total_reads} Comments={total_comments} Likes={total_likes}",
        (
            "Heuristic tone: "
            f"positive={tone_counts['positive']} "
            f"negative={tone_counts['negative']} "
            f"neutral={tone_counts['neutral']}"
        ),
        "Daily post counts: "
        + ", ".join(f"{day}={count}" for day, count in sorted(daily_counts.items())),
        "",
        "## Top Posts by Comments",
        "Time | Title | Reads | Comments | Likes",
    ]
    for post in ranked:
        title = _re.sub(r"[|\r\n]+", " ", str(post.get("post_title") or "")).strip()
        lines.append(
            f"{str(post.get('post_publish_time') or '')[:16]} | {title[:100]} "
            f"| {int(post.get('post_click_count') or 0)} "
            f"| {int(post.get('post_comment_count') or 0)} "
            f"| {int(post.get('post_like_count') or 0)}"
        )
    return "\n".join(lines)


def get_news(
    ticker: Annotated[str, "A-stock code"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Get stock-specific news via East Money direct API (Sina as fallback)."""
    code = _normalize_ticker(ticker)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    articles: list[dict] = []
    source_label = ""

    try:
        articles.extend(_fetch_announcements_eastmoney(code))
    except Exception as e:
        logger.info("Eastmoney announcement fetch failed for %s: %s", code, e)

    news_articles: list[dict] = []
    try:
        news_articles = _fetch_news_eastmoney(code)
        source_label = "东方财富"
    except Exception as e:
        logger.warning("East Money news fetch failed for %s: %s", code, e)

    if not news_articles:
        try:
            news_articles = _fetch_news_sina(code)
            source_label = "新浪财经"
        except Exception as e:
            logger.warning("Sina news fetch failed for %s: %s", code, e)

    articles.extend(news_articles)

    if not articles:
        return f"No news found for A-stock '{code}'"

    news_str = ""
    count = 0
    seen: set[tuple[str, str]] = set()
    for art in articles:
        pub_time = art.get("time", "")
        try:
            pub_dt = datetime.strptime(pub_time[:10], "%Y-%m-%d")
            if pub_dt < start_dt or pub_dt > end_dt:
                continue
        except (ValueError, IndexError):
            pass

        title = art["title"]
        dedupe_key = (title, pub_time[:10])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        content = art.get("content", "")
        source = art.get("source", source_label)
        link = art.get("url", "")

        news_str += f"### {title} (source: {source})\n"
        if content:
            snippet = content[:300] + "..." if len(content) > 300 else content
            news_str += f"{snippet}\n"
        if link and link != "nan":
            news_str += f"Link: {link}\n"
        news_str += "\n"
        count += 1

    if count == 0:
        return (
            f"No news found for A-stock '{code}' "
            f"between {start_date} and {end_date}"
        )

    return (
        f"## {code} (A-stock) News & Announcements, "
        f"from {start_date} to {end_date} ({count} items):\n\n"
        + news_str
    )


# ---- 8. get_global_news ----


def _fetch_cls_global_news(limit: int) -> list[dict]:
    """Fetch CLS telegraph items from the official mobile page bootstrap data.

    The former ``/nodeapi/telegraphList`` endpoint now returns a 404 HTML page.
    The public mobile telegraph page still embeds the same ``roll_data`` in its
    ``__NEXT_DATA__`` bootstrap assignment.  ``raw_decode`` is intentional: the
    script contains more JavaScript statements after the JSON object.
    """
    url = "https://m.cls.cn/telegraph"
    headers = {"User-Agent": _UA, "Referer": "https://www.cls.cn/"}
    response = _requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    text = response.content.decode("utf-8", errors="replace")

    marker = _re.search(r"__NEXT_DATA__\s*=\s*", text)
    if marker is None:
        raise ValueError("CLS telegraph page is missing __NEXT_DATA__")
    payload, _ = _json.JSONDecoder().raw_decode(text, marker.end())
    rows = (
        payload.get("props", {})
        .get("initialState", {})
        .get("roll_data", [])
    )
    if not isinstance(rows, list):
        raise ValueError("CLS telegraph roll_data has an unexpected shape")

    news: list[dict] = []
    for item in rows[:limit]:
        title = item.get("title", "") or item.get("brief", "")
        content = item.get("content", "") or item.get("brief", "")
        ctime = item.get("ctime", "")
        pub_time = ""
        if ctime:
            try:
                pub_time = datetime.fromtimestamp(int(ctime)).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except (ValueError, TypeError, OSError):
                pub_time = str(ctime)
        if title or content:
            news.append({
                "title": title or content[:80],
                "content": content,
                "time": pub_time,
                "source": "CLS Wire",
            })
    return news


def get_global_news(
    curr_date: Annotated[str, "Current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "Days to look back"] = 7,
    limit: Annotated[int, "Max articles"] = 10,
) -> str:
    """Get China/global financial news via direct HTTP (CLS + Eastmoney)."""
    start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - relativedelta(
        days=look_back_days
    )
    start_date = start_dt.strftime("%Y-%m-%d")

    all_news: list[dict] = []
    source_errors: dict[str, Exception] = {}

    # Source 1: CLS wire (财联社快讯) — official mobile telegraph page
    try:
        all_news.extend(_fetch_cls_global_news(limit))
    except Exception as e:
        source_errors["CLS"] = e

    # Source 2: Eastmoney global (东财7x24资讯) — direct HTTP
    try:
        em_url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
        em_params = {
            "client": "web",
            "biz": "web_724",
            "fastColumn": "102",
            "sortEnd": "",
            "pageSize": str(limit),
            "req_trace": str(uuid.uuid4()),
        }
        em_headers = {"User-Agent": _UA, "Referer": "https://kuaixun.eastmoney.com/"}
        r_em = _em_get(em_url, params=em_params, headers=em_headers, timeout=10)
        d_em = r_em.json()
        for item in d_em.get("data", {}).get("fastNewsList", []):
            title = item.get("title", "")
            summary = item.get("summary", "")[:200]
            pub_time = item.get("showTime", "")
            all_news.append({
                "title": title,
                "content": summary,
                "time": pub_time,
                "source": "Eastmoney Global",
            })
    except Exception as e:
        source_errors["Eastmoney"] = e

    if not all_news:
        logger.warning(
            "Global news sources unavailable for %s: %s",
            curr_date,
            "; ".join(f"{name}={error}" for name, error in source_errors.items()),
        )
        return f"No global news found for {curr_date}"

    for name, error in source_errors.items():
        logger.info("%s global news unavailable; another source succeeded: %s", name, error)

    # Deduplicate by title
    seen: set[str] = set()
    unique: list[dict] = []
    for n in all_news:
        if n["title"] not in seen:
            seen.add(n["title"])
            unique.append(n)

    news_str = ""
    for n in unique[:limit]:
        news_str += f"### {n['title']} (source: {n['source']})\n"
        if n.get("content"):
            snippet = (
                n["content"][:300] + "..."
                if len(n["content"]) > 300
                else n["content"]
            )
            news_str += f"{snippet}\n"
        news_str += "\n"

    return (
        f"## China & Global Market News, from {start_date} to {curr_date}:\n\n"
        + news_str
    )


# ---- 9. get_insider_transactions ----


def get_insider_transactions(
    ticker: Annotated[str, "A-stock code"],
) -> str:
    """Get shareholder/insider activity via mootdx F10.

    Note: A-stock insider transaction data differs from US markets.
    Uses mootdx F10 shareholder research as the closest equivalent.
    """
    code = _normalize_ticker(ticker)

    try:
        text = _call_mootdx("F10", symbol=code, name="股东研究")

        if not text or not text.strip():
            return f"No insider/shareholder data found for A-stock '{code}'"

        header = f"# Shareholder Research for {code} (A-stock)\n"
        header += "# Note: A-stock equivalent of insider transactions\n"
        header += "# Data source: mootdx F10\n"
        header += (
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        import re

        sec4_hits = list(re.finditer(r"\r?\n【4\.股东变化】\r?\n", text))
        if sec4_hits:
            sec4_pos = sec4_hits[-1].start()
            before_sec4 = text[:sec4_pos]
            sec4_text = text[sec4_pos:]
            cut_at = 2000
            if len(sec4_text) > cut_at:
                sec4_text = (
                    sec4_text[:cut_at]
                    + "\n\n(... older shareholder history omitted, "
                    f"{len(text) - sec4_pos - cut_at} chars truncated ...)"
                )
            text = before_sec4 + sec4_text

        return header + text

    except Exception as e:
        return f"Error retrieving insider/shareholder data for {code}: {str(e)}"


# ---- 10. get_profit_forecast ----


def get_profit_forecast(
    ticker: Annotated[str, "A-stock code"],
    curr_date: Annotated[str, "current date (unused, for interface compat)"] = None,
) -> str:
    """Get consensus EPS forecasts with forward valuation (同花顺 direct HTTP)."""
    code = _normalize_ticker(ticker)

    try:
        df = _ths_eps_forecast(code)

        if df is None or df.empty:
            return f"No analyst coverage found for A-stock '{code}'"

        lines = [
            f"# Consensus EPS Forecast for {code} (A-stock)",
            "# Source: 同花顺 analyst consensus (direct HTTP)",
            f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        eps_by_year = {}
        for _, row in df.iterrows():
            year = str(row.iloc[0]) if len(row) > 0 else ""
            count_val = row.iloc[1] if len(row) > 1 else 0
            mean_eps_val = row.iloc[3] if len(row) > 3 else 0
            min_eps_val = row.iloc[2] if len(row) > 2 else "N/A"
            max_eps_val = row.iloc[4] if len(row) > 4 else "N/A"
            try:
                count = int(count_val)
            except (ValueError, TypeError):
                count = 0
            try:
                mean_eps = float(mean_eps_val)
            except (ValueError, TypeError):
                mean_eps = 0
            lines.append(
                f"FY{year}: EPS={mean_eps} (range {min_eps_val}~{max_eps_val}), "
                f"analysts={count}"
            )
            if count < 3:
                lines.append("  Warning: low coverage (<3 analysts)")
            eps_by_year[year] = mean_eps

        # Forward valuation
        try:
            tq = _tencent_quote([code])
            if code in tq:
                price = tq[code]["price"]
                pe_ttm = tq[code]["pe_ttm"]
                lines.append(f"\nCurrent: price={price}, PE(TTM)={pe_ttm}")

                years_sorted = sorted(eps_by_year.keys())
                if years_sorted and eps_by_year.get(years_sorted[0], 0) > 0:
                    eps_cur = eps_by_year[years_sorted[0]]
                    fwd_pe = price / eps_cur
                    lines.append(
                        f"Forward PE (FY{years_sorted[0]}): {fwd_pe:.1f}x"
                    )
                    if (
                        len(years_sorted) >= 2
                        and eps_by_year.get(years_sorted[1], 0) > 0
                    ):
                        eps_next = eps_by_year[years_sorted[1]]
                        cagr = eps_next / eps_cur - 1
                        if cagr > 0:
                            peg = fwd_pe / (cagr * 100)
                            lines.append(
                                f"PEG: {peg:.2f} (CAGR={cagr * 100:.0f}%)"
                            )
                            if fwd_pe > 30:
                                digest = math.log(fwd_pe / 30) / math.log(
                                    1 + cagr
                                )
                                lines.append(
                                    f"PE Digestion to 30x: {digest:.1f} years"
                                )
                        else:
                            lines.append(
                                f"EPS declining ({cagr * 100:.0f}%), "
                                f"PEG not applicable"
                            )
        except Exception as e:
            logger.warning("Forward PE calc failed for %s: %s", code, e)

        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving profit forecast for {code}: {str(e)}"


# ---- 11. get_hot_stocks ----


def get_hot_stocks(
    curr_date: Annotated[str, "Date YYYY-MM-DD, empty string for today"] = "",
) -> str:
    """Get strong stocks with topic attribution from 同花顺 editorial team.

    Returns stocks that hit limit-up with human-curated reason tags
    explaining WHY they surged (e.g. '算力租赁+AI政务').
    """
    import requests

    if not curr_date or curr_date.strip() == "":
        curr_date = datetime.now().strftime("%Y-%m-%d")

    try:
        url = (
            f"http://zx.10jqka.com.cn/event/api/getharden/"
            f"date/{curr_date}/orderby/date/orderway/desc/charset/GBK/"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "Chrome/117.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        if data.get("errocode", 0) != 0:
            return f"同花顺 API error: {data.get('errormsg', 'unknown')}"

        rows = data.get("data") or []
        if not rows:
            return (
                f"No hot stocks data for {curr_date} "
                f"(may be non-trading day or data not yet available)"
            )

        lines = [
            f"# Hot Stocks with Topic Attribution ({curr_date})",
            "# Source: 同花顺 editorial (human-curated reason tags)",
            f"# Total: {len(rows)} stocks",
            "",
        ]

        from collections import Counter

        all_tags: list[str] = []

        for row in rows:
            code = row.get("code", "")
            name = row.get("name", "")
            reason = row.get("reason", "")
            zhangfu = row.get("zhangfu", "")
            huanshou = row.get("huanshou", "")
            chengjiaoe = row.get("chengjiaoe", "")
            dde = row.get("ddejingliang", "")

            lines.append(
                f"{code} {name}: +{zhangfu}% "
                f"换手{huanshou}% 成交额{chengjiaoe} "
                f"大单净量{dde} | {reason}"
            )

            if reason:
                tags = [t.strip() for t in str(reason).split("+") if t.strip()]
                all_tags.extend(tags)

        if all_tags:
            cnt = Counter(all_tags)
            lines.append("\n## Theme Frequency (top 15)")
            for tag, n in cnt.most_common(15):
                lines.append(f"  {tag}: {n} stocks")

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching hot stocks for {curr_date}: {str(e)}"


# ---- 12. get_northbound_flow ----


def _northbound_cache_path() -> str:
    """Path to local CSV cache for northbound daily close snapshots."""
    from .config import get_config

    config = get_config()
    cache_dir = config.get(
        "data_cache_dir", os.path.expanduser("~/.tradingagents/cache")
    )
    os.makedirs(cache_dir, exist_ok=True)
    # v1 could persist totals assembled from unequal HGT/SGT series.  Use a
    # versioned cache so those invalid snapshots are never reused.
    return os.path.join(cache_dir, "northbound_daily_v2.csv")


def _save_northbound_snapshot(date_str: str, hgt: float, sgt: float) -> None:
    """Append today's northbound close to local CSV cache (dedup by date)."""
    import csv

    path = _northbound_cache_path()
    existing: dict[str, tuple[str, str]] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    existing[row[0]] = (row[1], row[2])
    existing[date_str] = (f"{hgt:.2f}", f"{sgt:.2f}")
    sorted_dates = sorted(existing.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "hgt", "sgt"])
        for d in sorted_dates:
            writer.writerow([d, existing[d][0], existing[d][1]])


def _load_northbound_history(n: int = 20) -> list[tuple[str, float, float]]:
    """Load last N days of northbound close data from local cache."""
    import csv

    try:
        path = _northbound_cache_path()
    except OSError as exc:
        logger.info("Northbound cache directory unavailable: %s", exc)
        return []
    if not os.path.exists(path):
        return []
    rows: list[tuple[str, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                try:
                    rows.append((row[0], float(row[1]), float(row[2])))
                except ValueError:
                    continue
    return rows[-n:]


def get_northbound_flow(
    curr_date: Annotated[str, "Date YYYY-MM-DD"],
    include_history: Annotated[
        bool, "Include historical daily data (last 20 trading days)"
    ] = False,
) -> str:
    """Get northbound capital flow (沪深股通) from 同花顺 hsgtApi.

    Realtime: minute-level cumulative net buying for HGT(沪股通) + SGT(深股通).
    History: self-cached daily close snapshots (upstream APIs stopped updating
    northbound history since 2024-08).
    """
    hsgt_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "Chrome/117.0.0.0 Safari/537.36"
        ),
        "Host": "data.hexin.cn",
        "Referer": "https://data.hexin.cn/",
    }

    lines = [
        f"# Northbound Capital Flow ({curr_date})",
        "# Source: 同花顺 hsgtApi (沪深股通) + local cache",
        "",
    ]

    today_str = datetime.now().strftime("%Y-%m-%d")
    if curr_date != today_str:
        matching = [row for row in _load_northbound_history(200) if row[0] == curr_date]
        if matching:
            _, hgt_close, sgt_close = matching[-1]
            total = hgt_close + sgt_close
            lines.extend(
                [
                    "## Validated Historical Snapshot (亿元)",
                    f"HGT(沪股通)={hgt_close:.2f}亿 ",
                    f"SGT(深股通)={sgt_close:.2f}亿 ",
                    f"Total={total:.2f}亿",
                ]
            )
        else:
            lines.append(
                "[数据缺失: 北向资金] 上游接口只提供当日盘中序列，"
                f"本地无 {curr_date} 的已验证快照；为防止前视偏差，不返回当前值。"
            )
        return "\n".join(lines)

    hgt_close = 0.0
    sgt_close = 0.0
    got_realtime = False

    try:
        url_rt = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"
        r = _requests.get(url_rt, headers=hsgt_headers, timeout=10)
        d = r.json()

        times = d.get("time", [])
        hgt = d.get("hgt", [])
        sgt = d.get("sgt", [])

        if times:
            n = len(times)

            def validated_series(values) -> list[float]:
                if not isinstance(values, list) or len(values) != n:
                    return []
                try:
                    parsed = [float(value) for value in values]
                except (TypeError, ValueError):
                    return []
                return parsed if all(math.isfinite(value) for value in parsed) else []

            hgt_values = validated_series(hgt)
            sgt_values = validated_series(sgt)
            lines.append("## Realtime (cumulative net buying, 亿元)")
            start_idx = max(0, n - 10)
            for i in range(start_idx, n):
                t = times[i]
                h = f"{hgt_values[i]:.2f}" if hgt_values else "N/A"
                s = f"{sgt_values[i]:.2f}" if sgt_values else "N/A"
                lines.append(f"  {t}: HGT={h} SGT={s}")

            if not hgt_values:
                lines.append(
                    f"[数据质量警告] HGT 序列不完整 ({len(hgt)}/{n})。"
                )
            if not sgt_values:
                lines.append(
                    f"[数据质量警告] SGT 序列不完整 ({len(sgt)}/{n})。"
                )

            if hgt_values and sgt_values:
                hgt_close = hgt_values[-1]
                sgt_close = sgt_values[-1]
                total = hgt_close + sgt_close
                lines.append(
                    f"\nClose: HGT(沪股通)={hgt_close:.2f}亿 "
                    f"SGT(深股通)={sgt_close:.2f}亿 "
                    f"Total={total:.2f}亿"
                )
                if total > 0:
                    lines.append("Signal: Net northbound INFLOW (bullish)")
                elif total < 0:
                    lines.append("Signal: Net northbound OUTFLOW (bearish)")
                got_realtime = True
            else:
                available = []
                if hgt_values:
                    available.append(f"HGT={hgt_values[-1]:.2f}亿")
                if sgt_values:
                    available.append(f"SGT={sgt_values[-1]:.2f}亿")
                if available:
                    lines.append("Close (partial): " + " ".join(available))
                lines.append(
                    "[数据缺失: 北向资金合计] 沪深股通序列未同时完整返回，"
                    "不计算合计、不生成方向信号，也不写入缓存。"
                )
        else:
            lines.append("No realtime data (non-trading hours or holiday)")

        if got_realtime:
            try:
                _save_northbound_snapshot(today_str, hgt_close, sgt_close)
            except OSError as exc:
                logger.info(
                    "Northbound cache is read-only; keeping live result in memory: %s",
                    exc,
                )

        if include_history:
            history = _load_northbound_history(20)
            if history:
                lines.append("\n## Historical Daily Close (local cache, 亿元)")
                lines.append("Date       | HGT(沪股通) | SGT(深股通) | Total")
                for date, h, s in history:
                    lines.append(f"  {date}: HGT={h:.2f} SGT={s:.2f} Total={h + s:.2f}")
                avg_total = sum(h + s for _, h, s in history) / len(history)
                lines.append(
                    f"\n{len(history)}-day avg net flow: {avg_total:.2f}亿"
                )
                if got_realtime:
                    today_total = hgt_close + sgt_close
                    diff = today_total - avg_total
                    lines.append(
                        f"Today vs avg: {'+' if diff >= 0 else ''}{diff:.2f}亿 "
                        f"({'above' if diff >= 0 else 'below'} average)"
                    )
            else:
                lines.append(
                    "\n## Historical Daily: No cached data yet. "
                    "History accumulates automatically with each call."
                )

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching northbound flow: {str(e)}"


# ---------------------------------------------------------------------------
# Baidu PAE (百度股市通) helpers
# ---------------------------------------------------------------------------

_BAIDU_PAE_HEADERS = {
    "Host": "finance.pae.baidu.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) "
        "Gecko/20100101 Firefox/110.0"
    ),
    "Accept": "application/vnd.finance-web.v1+json",
    "Origin": "https://gushitong.baidu.com",
    "Referer": "https://gushitong.baidu.com/",
}


# ---- 13. get_concept_blocks ----


def get_concept_blocks(
    ticker: Annotated[str, "A-stock code (e.g. 688017)"],
) -> str:
    """Get concept/sector/region blocks that a stock belongs to (百度股市通).

    Returns industry classification (申万), concept themes, and region.
    Each block includes current day's change percentage.
    """
    import requests

    code = _normalize_ticker(ticker)

    try:
        url = (
            "https://finance.pae.baidu.com/api/getrelatedblock"
            f'?stock=[{{"code":"{code}","market":"ab","type":"stock"}}]'
            "&finClientType=pc"
        )
        r = requests.get(url, headers=_BAIDU_PAE_HEADERS, timeout=10)
        d = r.json()

        if str(d.get("ResultCode", -1)) != "0":
            return (
                f"Baidu PAE error: ResultCode={d.get('ResultCode')} "
                f"{d.get('ResultMsg', '')}"
            )

        result = d.get("Result", {})
        categories = result.get(code, [])
        if not categories:
            return f"No concept/block data for {code}"

        lines = [
            f"# Concept & Sector Blocks for {code} (A-stock)",
            "# Source: 百度股市通 (Baidu PAE)",
            f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        concept_names: list[str] = []

        for cat in categories:
            cat_name = cat.get("name", "")
            items = cat.get("list", [])
            if not items:
                continue
            lines.append(f"## {cat_name}")
            for item in items:
                name = item.get("name", "")
                ratio = item.get("ratio", "")
                desc = item.get("describe", "")
                suffix = f" ({desc})" if desc else ""
                lines.append(f"  {name}{suffix}: {ratio}")
                if cat_name == "概念":
                    concept_names.append(name)

        if concept_names:
            lines.append(f"\nConcept tags: {' / '.join(concept_names)}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching concept blocks for {code}: {str(e)}"


# ---- 14. get_fund_flow ----


def get_fund_flow(
    ticker: Annotated[str, "A-stock code"],
    curr_date: Annotated[str, "Date YYYY-MM-DD"],
    include_history: Annotated[
        bool, "Include historical daily fund flow (last 20 days)"
    ] = True,
) -> str:
    """Get individual stock fund flow from 东财 push2.

    Realtime: minute-level main/large/medium/small/super order net inflow.
    History: daily net inflow for 20 trading days (push2his).

    V0.2.7: replaced 百度 PAE (fundflow/fundsortlist, offline since 2026-05)
    with 东财 push2 fund flow API.
    """
    code = _normalize_ticker(ticker)
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    lines = [
        f"# Fund Flow for {code} (A-stock)",
        "# Source: 东财 push2 (Eastmoney)",
        f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    try:
        # Realtime minute-level fund flow
        url_rt = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
        params_rt = {
            "secid": secid, "klt": 1,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
        }
        r = _em_get(url_rt, params=params_rt, timeout=10)
        d = r.json()
        klines = d.get("data", {}).get("klines", [])

        if klines:
            lines.append(
                "## Realtime Minute Flow "
                "(主力/小单/中单/大单/超大单 净流入, 元)"
            )
            for line in klines[-10:]:
                parts = line.split(",")
                if len(parts) >= 6:
                    lines.append(
                        f"  {parts[0]}: "
                        f"主力={float(parts[1])/1e4:.0f}万 "
                        f"大单={float(parts[4])/1e4:.0f}万 "
                        f"超大单={float(parts[5])/1e4:.0f}万"
                    )

            last_parts = klines[-1].split(",")
            if len(last_parts) >= 2:
                main_net = float(last_parts[1])
                lines.append(
                    f"\nClose: 主力净流入={main_net/1e4:.0f}万元"
                )
                if main_net > 0:
                    lines.append(
                        "Signal: Net main force INFLOW (bullish)"
                    )
                elif main_net < 0:
                    lines.append(
                        "Signal: Net main force OUTFLOW (bearish)"
                    )
        else:
            lines.append(
                "No realtime fund flow (non-trading hours or holiday)"
            )

        # Historical daily fund flow (push2his)
        if include_history:
            url_hist = (
                "https://push2his.eastmoney.com"
                "/api/qt/stock/fflow/daykline/get"
            )
            params_hist = {
                "secid": secid, "lmt": 20, "klt": 101,
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56,f57",
            }
            rh = _em_get(url_hist, params=params_hist, timeout=10)
            dh = rh.json()
            hist_klines = dh.get("data", {}).get("klines", [])

            if hist_klines:
                lines.append(
                    f"\n## Historical Daily Fund Flow "
                    f"(last {len(hist_klines)} trading days)"
                )
                lines.append(
                    "Date | 主力净流入(万) | 大单(万) "
                    "| 中单(万) | 小单(万) | 超大单(万)"
                )
                for line in hist_klines:
                    parts = line.split(",")
                    if len(parts) >= 6:
                        lines.append(
                            f"  {parts[0]} "
                            f"| main={float(parts[1])/1e4:.0f} "
                            f"| large={float(parts[4])/1e4:.0f} "
                            f"| mid={float(parts[3])/1e4:.0f} "
                            f"| small={float(parts[2])/1e4:.0f} "
                            f"| super={float(parts[5])/1e4:.0f}"
                        )

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching fund flow for {code}: {str(e)}"


# ---------------------------------------------------------------------------
# 15. Dragon Tiger Board (龙虎榜)
# ---------------------------------------------------------------------------

def get_dragon_tiger_board(
    ticker: str,
    trade_date: str,
    look_back_days: int = 30,
) -> str:
    """Get dragon-tiger board (龙虎榜) appearances and seat details.

    Args:
        ticker: 6-digit A-share code, e.g. '000858'
        trade_date: YYYY-MM-DD
        look_back_days: how many days back to search (default 30)

    Returns:
        Formatted text with LHB appearances, top buyer/seller seats,
        and institutional activity.
    """
    code = safe_ticker_component(ticker)
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = end_dt - pd.Timedelta(days=look_back_days)
    start_date_str = start_dt.strftime("%Y-%m-%d")
    lines = [f"# 龙虎榜数据 | {code} | {trade_date} (近{look_back_days}日)"]

    # 1. 上榜记录 — eastmoney datacenter direct HTTP
    try:
        data = _eastmoney_datacenter(
            "RPT_DAILYBILLBOARD_DETAILSNEW",
            filter_str=(
                f"(TRADE_DATE>='{start_date_str}')"
                f"(TRADE_DATE<='{trade_date}')"
                f"(SECURITY_CODE=\"{code}\")"
            ),
            page_size=50,
            sort_columns="TRADE_DATE",
            sort_types="-1",
        )
        if not data:
            lines.append(f"\n近{look_back_days}日未上龙虎榜。")
        else:
            lines.append(f"\n## 上榜记录 ({len(data)} 次)")
            lines.append("日期 | 原因 | 净买入(万) | 换手率")
            for row in data:
                net_buy = round((row.get("BILLBOARD_NET_AMT") or 0) / 10000, 1)
                turnover = round(float(row.get("TURNOVERRATE") or 0), 2)
                lines.append(
                    f"  {str(row.get('TRADE_DATE', ''))[:10]} "
                    f"| {row.get('EXPLANATION', '')} "
                    f"| {net_buy:.0f} "
                    f"| {turnover:.2f}%"
                )
    except Exception as e:
        lines.append(f"龙虎榜列表查询失败: {e}")

    # 2. 最近上榜的买卖席位 — eastmoney datacenter direct HTTP
    try:
        if data:
            latest_date = str(data[0].get("TRADE_DATE", ""))[:10]
            lines.append(f"\n## 最近上榜席位明细 ({latest_date})")

            # 买入席位
            buy_data = _eastmoney_datacenter(
                "RPT_BILLBOARD_DAILYDETAILSBUY",
                filter_str=f"(TRADE_DATE='{latest_date}')(SECURITY_CODE=\"{code}\")",
                page_size=10,
                sort_columns="BUY",
                sort_types="-1",
            )
            if buy_data:
                lines.append("\n### 买入席位 TOP5")
                lines.append("营业部 | 买入(万) | 卖出(万) | 净额(万)")
                for row in buy_data[:5]:
                    buy_amt = round((row.get("BUY") or 0) / 10000, 1)
                    sell_amt = round((row.get("SELL") or 0) / 10000, 1)
                    net = round((row.get("NET") or 0) / 10000, 1)
                    lines.append(
                        f"  {row.get('OPERATEDEPT_NAME', '')} "
                        f"| {buy_amt:.0f} | {sell_amt:.0f} | {net:.0f}"
                    )

            # 卖出席位
            sell_data = _eastmoney_datacenter(
                "RPT_BILLBOARD_DAILYDETAILSSELL",
                filter_str=f"(TRADE_DATE='{latest_date}')(SECURITY_CODE=\"{code}\")",
                page_size=10,
                sort_columns="SELL",
                sort_types="-1",
            )
            if sell_data:
                lines.append("\n### 卖出席位 TOP5")
                lines.append("营业部 | 买入(万) | 卖出(万) | 净额(万)")
                for row in sell_data[:5]:
                    buy_amt = round((row.get("BUY") or 0) / 10000, 1)
                    sell_amt = round((row.get("SELL") or 0) / 10000, 1)
                    net = round((row.get("NET") or 0) / 10000, 1)
                    lines.append(
                        f"  {row.get('OPERATEDEPT_NAME', '')} "
                        f"| {buy_amt:.0f} | {sell_amt:.0f} | {net:.0f}"
                    )
    except Exception:
        pass

    # 3. 机构动向 — 从买卖席位明细筛选机构专用席位 (OPERATEDEPT_CODE="0")
    try:
        inst_buy = 0.0
        inst_sell = 0.0
        for detail, side in [(buy_data, "buy"), (sell_data, "sell")]:
            for row in (detail or []):
                if str(row.get("OPERATEDEPT_CODE", "")) == "0":
                    if side == "buy":
                        inst_buy += (row.get("BUY") or 0)
                    else:
                        inst_sell += (row.get("SELL") or 0)
        if inst_buy > 0 or inst_sell > 0:
            lines.append("\n## 机构动向")
            lines.append(
                f"  机构买入 {inst_buy/1e4:.0f} 万 "
                f"| 卖出 {inst_sell/1e4:.0f} 万 "
                f"| 净额 {(inst_buy - inst_sell)/1e4:.0f} 万"
            )
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 16. Lockup Expiry Calendar (限售解禁日历)
# ---------------------------------------------------------------------------

def get_lockup_expiry(
    ticker: str,
    trade_date: str,
    forward_days: int = 90,
) -> str:
    """Get lockup expiry schedule for a stock.

    Args:
        ticker: 6-digit A-share code
        trade_date: YYYY-MM-DD
        forward_days: how many days forward to check (default 90)

    Returns:
        Formatted text with historical unlock records and upcoming
        expiry calendar with impact metrics.
    """
    code = safe_ticker_component(ticker)
    lines = [f"# 限售解禁日历 | {code} | {trade_date}"]

    def format_lockup_row(row: dict) -> str:
        share_type = row.get("FREE_SHARES_TYPE") or row.get(
            "LIMITED_STOCK_TYPE", ""
        )
        shares = row.get("CURRENT_FREE_SHARES")
        if shares is None:
            shares = row.get("ABLE_FREE_SHARES") or row.get("FREE_SHARES_NUM")
        ratio = row.get("FREE_RATIO")
        market_cap = row.get("LIFT_MARKET_CAP")

        shares_text = "N/A" if shares is None else f"{float(shares):,.2f} 万股"
        ratio_text = "N/A" if ratio is None else f"{float(ratio) * 100:.4f}%"
        market_cap_text = (
            "N/A"
            if market_cap is None
            else f"{float(market_cap) / 10000:.2f} 亿元"
        )
        return (
            f"{str(row.get('FREE_DATE', ''))[:10]} "
            f"| {share_type or '类型未披露'} "
            f"| {shares_text} "
            f"| 占流通股 {ratio_text} "
            f"| 解禁市值 {market_cap_text}"
        )

    # 1. 历史解禁记录 — eastmoney datacenter direct HTTP
    try:
        history_data = _eastmoney_datacenter(
            "RPT_LIFT_STAGE",
            filter_str=f"(SECURITY_CODE=\"{code}\")",
            page_size=15,
            sort_columns="FREE_DATE",
            sort_types="-1",
        )
        if history_data:
            lines.append(f"\n## 个股解禁记录 (共 {len(history_data)} 批)")
            lines.append("解禁时间 | 类型 | 解禁数量 | 占流通股 | 解禁市值")
            for row in history_data:
                lines.append(f"  {format_lockup_row(row)}")
        else:
            lines.append("\n无历史解禁记录。")
    except Exception as e:
        lines.append(f"个股解禁查询失败: {e}")

    # 2. 未来待解禁 — eastmoney datacenter direct HTTP
    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d") + pd.Timedelta(
            days=forward_days
        )
        end_str = end_dt.strftime("%Y-%m-%d")
        upcoming_data = _eastmoney_datacenter(
            "RPT_LIFT_STAGE",
            filter_str=(
                f"(SECURITY_CODE=\"{code}\")"
                f"(FREE_DATE>='{trade_date}')"
                f"(FREE_DATE<='{end_str}')"
            ),
            page_size=20,
            sort_columns="FREE_DATE",
            sort_types="1",
        )
        if upcoming_data:
            lines.append(f"\n## 未来 {forward_days} 天待解禁")
            lines.append("解禁时间 | 类型 | 解禁数量 | 占流通股 | 解禁市值")
            for row in upcoming_data:
                lines.append(f"  {format_lockup_row(row)}")
        else:
            lines.append(f"\n未来 {forward_days} 天无待解禁。")
    except Exception as e:
        lines.append(f"解禁日历查询失败: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 17. Industry Comparison (行业横向对比)
# ---------------------------------------------------------------------------

def get_industry_comparison(
    ticker: str,
    trade_date: str,
    top_n: int = 20,
) -> str:
    """Get industry sector performance comparison.

    Args:
        ticker: 6-digit A-share code (used to identify relevant sector)
        trade_date: YYYY-MM-DD
        top_n: number of top/bottom industries to show (default 20)

    Returns:
        Formatted text with sector performance ranking, highlighting
        the sector the target stock belongs to.
    """
    code = safe_ticker_component(ticker)
    lines = [f"# 行业横向对比 | {code} | {trade_date}"]

    # 东财 push2 行业板块排名 (direct HTTP, replaces 同花顺 which has 401)
    try:
        target_industry = ""
        try:
            market_code = 1 if code.startswith("6") else 0
            info_response = _em_get(
                "https://push2.eastmoney.com/api/qt/stock/get",
                params={
                    "fltt": "2",
                    "invt": "2",
                    "fields": "f57,f58,f127",
                    "secid": f"{market_code}.{code}",
                },
                timeout=10,
            )
            target_industry = (
                info_response.json().get("data", {}) or {}
            ).get("f127", "")
        except Exception as exc:
            logger.info("Stock industry lookup unavailable for %s: %s", code, exc)

        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": "1",
            "pz": "100",
            "po": "1",
            "np": "1",
            "fid": "f3",
            "fltt": "2",
            "invt": "2",
            "fs": "m:90+t:2",
            "fields": (
                "f2,f3,f4,f6,f12,f13,f14,f62,f104,f105,"
                "f128,f136,f140,f141,f184,f207"
            ),
        }
        r = _em_get(url, params=params, timeout=15)
        d = r.json()
        items = d.get("data", {}).get("diff", [])

        if items:
            def industry_line(rank: int, item: dict) -> str:
                amount = item.get("f6")
                main_flow = item.get("f62")
                amount_yi = (
                    "N/A" if amount in (None, "-") else f"{float(amount) / 1e8:.2f}亿"
                )
                flow_yi = (
                    "N/A"
                    if main_flow in (None, "-")
                    else f"{float(main_flow) / 1e8:.2f}亿"
                )
                return (
                    f"  {rank}. {item.get('f14', '')} "
                    f"| {item.get('f3', 0)}% "
                    f"| {amount_yi} "
                    f"| {flow_yi} "
                    f"| {item.get('f184', 'N/A')}% "
                    f"| {item.get('f104', 0)}/{item.get('f105', 0)} "
                    f"| {item.get('f140', '')}"
                )

            lines.append(
                f"\n## 全行业表现 (东财 {len(items)} 个行业)"
            )
            lines.append(
                "排名 | 行业 | 涨跌幅 | 成交额 | 主力净流入 | 主力净占比 | 上涨/下跌 | 领涨股"
            )
            display_indices = list(range(min(top_n, len(items))))
            bottom_start = max(top_n, len(items) - top_n)
            display_indices.extend(range(bottom_start, len(items)))
            for position, index in enumerate(display_indices):
                if position == min(top_n, len(items)) and bottom_start > top_n:
                    lines.append("  ...")
                lines.append(industry_line(index + 1, items[index]))

            target_matches = [
                (index, item)
                for index, item in enumerate(items)
                if target_industry and item.get("f14") == target_industry
            ]
            if target_matches:
                target_index, target_item = target_matches[0]
                lines.append(f"\n## 标的所属行业：{target_industry}")
                lines.append(industry_line(target_index + 1, target_item))

                board_code = target_item.get("f12")
                if board_code:
                    try:
                        peer_response = _em_get(
                            url,
                            params={
                                "pn": "1",
                                "pz": "100",
                                "po": "1",
                                "np": "1",
                                "fid": "f20",
                                "fltt": "2",
                                "invt": "2",
                                "fs": f"b:{board_code}",
                                "fields": "f2,f3,f9,f12,f14,f20,f23",
                            },
                            timeout=15,
                        )
                        peers = (
                            peer_response.json().get("data", {}).get("diff", [])
                        )
                        pe_values = [
                            float(peer.get("f9"))
                            for peer in peers
                            if peer.get("f9") not in (None, "-")
                            and float(peer.get("f9")) > 0
                        ]
                        pb_values = [
                            float(peer.get("f23"))
                            for peer in peers
                            if peer.get("f23") not in (None, "-")
                            and float(peer.get("f23")) > 0
                        ]
                        if peers:
                            lines.append("\n## 同行业估值对比（按总市值排序）")
                            if pe_values or pb_values:
                                median_pe = (
                                    f"{pd.Series(pe_values).median():.2f}x"
                                    if pe_values
                                    else "N/A"
                                )
                                median_pb = (
                                    f"{pd.Series(pb_values).median():.2f}x"
                                    if pb_values
                                    else "N/A"
                                )
                                lines.append(
                                    f"行业样本={len(peers)}，有效 PE 中位数={median_pe}，"
                                    f"有效 PB 中位数={median_pb}"
                                )
                            lines.append("代码 | 公司 | 股价 | 涨跌幅 | PE | PB | 总市值")
                            for peer in peers[:10]:
                                market_cap = peer.get("f20")
                                market_cap_text = (
                                    "N/A"
                                    if market_cap in (None, "-")
                                    else f"{float(market_cap) / 1e8:.2f}亿"
                                )
                                lines.append(
                                    f"{peer.get('f12', '')} | {peer.get('f14', '')} "
                                    f"| {peer.get('f2', 'N/A')} "
                                    f"| {peer.get('f3', 'N/A')}% "
                                    f"| {peer.get('f9', 'N/A')} "
                                    f"| {peer.get('f23', 'N/A')} "
                                    f"| {market_cap_text}"
                                )
                    except Exception as exc:
                        lines.append(f"同行估值获取失败: {exc}")
            elif target_industry:
                lines.append(
                    f"\n标的所属行业：{target_industry}；本次行业榜未返回同名板块。"
                )
        else:
            lines.append("行业数据获取为空。")
    except Exception as e:
        lines.append(f"行业对比查询失败: {e}")

    return "\n".join(lines)
