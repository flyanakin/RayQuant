import pandas as pd
import matplotlib.pyplot as plt
from core.portfolio import Portfolio
import plotly.express as px
from utils.indicators import win_rate, annual_return, annual_volatility, drawdown


class Observer:
    def __init__(self, portfolio: Portfolio):
        """
        观察者模块，用于监控和记录回测过程中的数据与指标。

        :param portfolio: Portfolio 对象，包含交易和持仓信息。
        """
        self.portfolio = portfolio
        self.results = pd.DataFrame(
            columns=["date", "total_value", "cash", "returns", "returns_pct"]
        )
        self.performance_metrics = {}
        self.drawdown_records = pd.DataFrame()
        # 记录benchmark数据与指标
        self.benchmark_results = []  # 每条记录为：{"date": dt, benchmark_symbol: close, ...}
        self.benchmark_metrics = {}  # 格式：{ benchmark_symbol: {"annual_return": ..., "max_drawdown": ..., "annual_volatility": ...} }

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

    def record_benchmark(self, dt, benchmark_values: dict):
        """
        记录每个时间步 benchmark 的收盘价数据

        :param dt: 当前日期
        :param benchmark_values: 字典格式 {benchmark_symbol: close_price, ...}
        """
        record = {"date": dt}
        record.update(benchmark_values)
        self.benchmark_results.append(record)

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

        # 计算年化波动率
        daily_returns = self.results[["date", "returns_pct"]].copy().set_index("date")
        daily_returns["returns_pct"] = daily_returns["returns_pct"] / 100
        self.performance_metrics["annual_volatility"] = annual_volatility(daily_returns)

    def calculate_benchmark_metrics(self, interval_months: int = 3):
        """
        针对记录的 benchmark 数据计算各 benchmark 的
        年化收益率、最大回撤、年化波动率，并写入 self.benchmark_metrics 字典中
        """
        if not self.benchmark_results:
            print("未记录到任何 benchmark 数据。")
            return

        df = pd.DataFrame(self.benchmark_results)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        # 对于每个 benchmark 列（除 date 外）
        for benchmark in df.columns:
            series = df[benchmark]
            if series.empty:
                continue
            start_value = series.iloc[0]
            end_value = series.iloc[-1]
            total_days = (series.index[-1] - series.index[0]).days
            ann_return = annual_return(start_value, end_value, total_days)

            # 利用 drawdown 函数计算最大回撤
            series_df = series.to_frame(name="close")
            dd_df, (max_dd, max_dd_interval) = drawdown(series_df, interval_months)

            # 计算年化波动率（先计算每日收益率）
            daily_returns = series.pct_change().dropna()
            daily_returns_df = pd.DataFrame({"returns_pct": daily_returns})
            ann_vol = annual_volatility(daily_returns_df)

            self.benchmark_metrics[benchmark] = {
                "annual_return": ann_return,
                "max_drawdown": max_dd,
                "annual_volatility": ann_vol,
            }

    def plot_results(self):
        """
        绘制收益率曲线和基准曲线（如果有）。
        """
        # 处理组合数据
        df_port = self.results.copy()
        df_port["date"] = pd.to_datetime(df_port["date"])
        df_port = df_port.sort_values("date").set_index("date")
        df_port = df_port[["total_value"]]

        # 处理 benchmark 数据
        df_bench = pd.DataFrame(self.benchmark_results)
        df_bench["date"] = pd.to_datetime(df_bench["date"])
        df_bench = df_bench.sort_values("date").set_index("date")

        # 合并两者（按日期对齐）
        df_all = df_port.join(df_bench, how="outer").reset_index()

        # 获取除日期外所有列
        cols_to_plot = [col for col in df_all.columns if col != "date"]
        # 将所有绘图列转换为数值型，确保类型一致
        for col in cols_to_plot:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

        fig = px.line(df_all, x="date", y=cols_to_plot, title="Portfolio Value and Benchmark")
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()

    def print_metrics(self):
        """
        打印计算的评价指标。
        """
        print("Portfolio Metrics:")
        for metric, value in self.performance_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value}")

        if self.benchmark_metrics:
            print("\nBenchmark Metrics:")
            for benchmark, metrics in self.benchmark_metrics.items():
                print(f"Benchmark {benchmark}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.2%}")
                    else:
                        print(f"  {metric}: {value}")
