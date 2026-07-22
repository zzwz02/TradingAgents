import logging
from datetime import datetime

from dateutil.relativedelta import relativedelta
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_fundamentals,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
logger = logging.getLogger(__name__)

_PREFETCH_STOCK_LOOKBACK_DAYS = 45
_PREFETCH_INDICATOR_LOOKBACK_DAYS = 30
_PREFETCH_INDICATORS = (
    "close_10_ema",
    "close_50_sma",
    "close_200_sma",
    "macd",
    "macds",
    "macdh",
    "rsi",
    "atr",
)


def _safe_tool_invoke(tool_obj, payload: dict) -> str:
    try:
        return str(tool_obj.invoke(payload))
    except Exception as exc:
        name = getattr(tool_obj, "name", tool_obj.__class__.__name__)
        return f"[数据缺失: {name}] {type(exc).__name__}: {exc}"


def _prefetch_start_date(current_date: str) -> str:
    try:
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    except ValueError:
        return current_date
    return (current_dt - relativedelta(days=_PREFETCH_STOCK_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )


def _build_prefetched_market_data(ticker: str, current_date: str) -> str:
    """Fetch core technical data before the LLM runs.

    CLI-based providers do not always emit structured tool calls reliably. This
    prefetch gives the analyst concrete OHLCV and indicator data even when the
    model chooses to answer directly.
    """
    start_date = _prefetch_start_date(current_date)
    quote_data = _safe_tool_invoke(
        get_fundamentals,
        {"ticker": ticker, "curr_date": current_date},
    )
    stock_data = _safe_tool_invoke(
        get_stock_data,
        {
            "symbol": ticker,
            "start_date": start_date,
            "end_date": current_date,
        },
    )
    indicator_data = _safe_tool_invoke(
        get_indicators,
        {
            "symbol": ticker,
            "indicator": ",".join(_PREFETCH_INDICATORS),
            "curr_date": current_date,
            "look_back_days": _PREFETCH_INDICATOR_LOOKBACK_DAYS,
        },
    )

    message = (
        "Prefetched A-stock technical data for "
        f"{ticker} through {current_date}: "
        f"{len(quote_data)} quote chars, {len(stock_data)} stock chars, "
        f"{len(indicator_data)} indicator chars"
    )
    logger.info(
        "%s",
        message,
    )
    print(
        f"[A-STOCK TECHNICAL PREFETCH] {message}",
        flush=True,
    )

    return (
        "## 已预取的 A 股技术面数据\n"
        "以下数据已在技术分析师运行前由后台直连工具获取。必须优先使用这些数据；"
        "如果某段包含 [数据缺失] 或错误信息，请在报告中明确标注对应缺口，"
        "并可继续调用工具补拉。\n\n"
        "### 实时行情、换手率与股本快照\n"
        f"{quote_data}\n\n"
        "### K 线与成交量\n"
        f"{stock_data}\n\n"
        "### 技术指标\n"
        f"{indicator_data}"
    )


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)
        prefetched_market_data = _build_prefetched_market_data(ticker, current_date)

        tools = [
            get_stock_data,
            get_indicators,
            get_fundamentals,
        ]

        system_message = (
            """你是一位专注于 A 股市场的技术分析师。你的任务是从以下技术指标中选择最多 **8 个**最相关的指标，为给定的 A 股标的提供技术面分析。选择时应注重指标间的互补性，避免冗余。

⚠️ A 股市场特殊规则（分析时必须纳入考量）：
- **涨跌停制度**：主板 ±10%，科创板/创业板 ±20%，ST 股 ±5%。触及涨跌停后流动性骤降，技术指标可能失真。
- **T+1 交易制度**：当日买入次日才能卖出，短线策略的可执行性受限。
- **北向资金**：外资通过沪深港通的流入流出是重要的市场风向标，大幅流入/流出常领先于趋势转折。
- **换手率**：A 股散户占比高，换手率是判断资金活跃度和筹码松动的关键指标。
- **量价关系**：A 股「量在价先」规律显著，放量突破和缩量回调是核心交易信号。

可选技术指标（调用 get_indicators 时必须使用下列英文标识符作为参数名）：

均线类 (Moving Averages)：
- close_50_sma：50 日简单均线 - 中期趋势方向判断，动态支撑/阻力位。滞后性较强，需配合短期指标。
- close_200_sma：200 日简单均线 - 长期趋势基准，金叉/死叉战略信号。反应缓慢，适合趋势确认。
- close_10_ema：10 日指数均线 - 短期动量快速捕捉，适合活跃交易。震荡市噪音多，需配合长均线过滤。

MACD 类：
- macd：MACD 主线 - 趋势动量的核心信号，关注交叉与背离。横盘市需配合其他指标确认。
- macds：MACD 信号线 - 与主线交叉触发交易信号。单独使用易产生假信号。
- macdh：MACD 柱状图 - 动量强度可视化，提前发现顶/底背离。波动较大，需配合趋势过滤。

动量类 (Momentum)：
- rsi：RSI 相对强弱指标 - 超买(>70)/超卖(<30)判断。注意：A 股强势股 RSI 可长期维持在 60-80 区间，不能机械套用阈值。

波动率类 (Volatility)：
- boll：布林带中轨 - 20 日均线基准，价格运动的中枢参考。
- boll_ub：布林带上轨 - 价格触及时为潜在超买/突破信号。强趋势中价格可能沿上轨运行。
- boll_lb：布林带下轨 - 价格触及时为潜在超卖信号。需配合其他指标确认是否真正见底。
- atr：ATR 平均真实波幅 - 衡量波动率，用于动态止损和仓位管理。

成交量类 (Volume)：
- vwma：成交量加权均线 - 结合量价验证趋势的可靠性。注意异常放量可能扭曲结果。

操作要求：
1. **必须**先调用 get_stock_data 获取 K 线数据
2. 再调用 get_indicators 获取选定指标（参数名使用上述英文标识符，否则调用会失败）
3. 撰写详细的技术分析报告，包含具体数值和技术信号研判结论（仅供研究参考，不构成投资建议）
4. 报告末尾附 Markdown 表格汇总关键技术信号和结论

后台已在本轮运行前预取 K 线和关键指标，见下方“已预取的 A 股技术面数据”。如果预取数据完整，可以直接据此撰写报告；如需补充更多指标，再继续调用工具。

📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]：
1. 最新收盘价、日期、当日涨跌幅
2. 近 30 日累计涨跌幅
3. 近 5 日平均成交量 vs 近 20 日平均成交量（判断放量/缩量）
4. 至少 3 个技术指标的当前数值和多空信号
5. 关键支撑位和阻力位"""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}"
                    "\n\n{prefetched_market_data}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(prefetched_market_data=prefetched_market_data)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
