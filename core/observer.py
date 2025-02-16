import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from core.portfolio import Portfolio
import plotly.express as px
from utils.indicators import win_rate, annual_return, annual_volatility, drawdown


class Observer:
    def __init__(self, portfolio: Portfolio, benchmark: Optional[pd.DataFrame] = None):
        """
        观察者模块，用于监控和记录回测过程中的数据与指标。

        :param portfolio: Portfolio 对象，包含交易和持仓信息。
        :param benchmark: 基准曲线，pandas DataFrame，index 为日期，columns 为标的符号及其价格。
        """
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.results = pd.DataFrame(
            columns=["date", "total_value", "cash", "returns", "returns_pct"]
        )
        self.performance_metrics = {}
        self.drawdown_records = pd.DataFrame()

    def record(self, dt, total_value, cash):
        """
        记录每个时间步的回测结果。

        :param dt: 当前日期。
        :param total_value: 组合总价值。
        :param cash: 当前现金。
        """
        previous_value = (
            self.results["total_value"].iloc[-1]
            if not self.results.empty
            else total_value
        )
        returns = total_value - previous_value
        returns_pct = (returns / previous_value) * 100 if previous_value != 0 else 0

        new_row = pd.DataFrame(
            [
                {
                    "date": dt,
                    "total_value": total_value,
                    "cash": cash,
                    "returns": returns,
                    "returns_pct": returns_pct,
                }
            ]
        )
        self.results = pd.concat([self.results, new_row], ignore_index=True)

    def calculate_metrics(self, interval_months: int = 3):
        """
        计算主要的评价指标，包括：
        - 交易胜率
        - 年化收益率
        - 区间最大回撤（默认 3 个月）
        - 总最大回撤
        - 年化波动率
        """
        # 计算胜率
        trade_log = self.portfolio.trade_log
        self.performance_metrics["win_rate"] = win_rate(trade_log, self.portfolio)

        # 计算年化收益率
        start_value = self.results["total_value"].iloc[0]
        end_value = self.results["total_value"].iloc[-1]
        total_days = (self.results["date"].iloc[-1] - self.results["date"].iloc[0]).days
        self.performance_metrics["annual_return"] = annual_return(
            start_value, end_value, total_days
        )

        # 计算最大回撤
        values = self.results[["date", "total_value"]].copy().set_index("date")
        drawdown_result = drawdown(values, interval_months)
        self.drawdown_records = drawdown_result[0]
        (
            self.performance_metrics["max_drawdown"],
            self.performance_metrics["max_drawdown_interval"],
        ) = drawdown_result[1]

        # 计算年华波动率
        daily_returns = self.results[["date", "returns_pct"]].copy().set_index("date")
        daily_returns["returns_pct"] = daily_returns["returns_pct"] / 100
        self.performance_metrics["annual_volatility"] = annual_volatility(daily_returns)

    def plot_results(self):
        """
        绘制收益率曲线和基准曲线（如果有）。
        """
        df = self.results
        fig = px.line(
            df, x="date", y="total_value", title="Portfolio Value and Benchmark"
        )

        fig.update_xaxes(rangeslider_visible=True)
        fig.show()

    def print_metrics(self):
        """
        打印计算的评价指标。
        """
        for metric, value in self.performance_metrics.items():
            print(
                f"{metric}: {value:.2%}"
                if isinstance(value, float)
                else f"{metric}: {value}"
            )
