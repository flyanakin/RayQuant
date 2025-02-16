import pandas as pd
import matplotlib.pyplot as plt
from core.portfolio import Portfolio
import plotly.express as px
import plotly.graph_objects as go
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

        self.initial_portfolio_value = None
        # ===== 专门存 benchmark 的相对收益(同样用 list[dict] 结构) =====
        self.benchmark_relative_results = []
        # 用于存储各 benchmark 的初始价格
        self.benchmark_initial_values = {}

    def record(self, dt, total_value, cash):
        """
        记录每个时间步的回测结果。

        :param dt: 当前日期。
        :param total_value: 组合总价值。
        :param cash: 当前现金。
        """
        # 如果是第一次记录，则记下初始净值
        if self.initial_portfolio_value is None:
            self.initial_portfolio_value = total_value

        previous_value = (
            self.results["total_value"].iloc[-1]
            if not self.results.empty
            else total_value
        )
        returns = total_value - previous_value
        returns_pct = (returns / previous_value) * 100 if previous_value != 0 else 0

        # ===== 计算组合相对收益(基于初始净值归一化，从1.0起) =====
        relative_return = total_value / self.initial_portfolio_value

        new_row = pd.DataFrame(
            [
                {
                    "date": dt,
                    "total_value": total_value,
                    "cash": cash,
                    "returns": returns,
                    "returns_pct": returns_pct,
                    "relative_return": relative_return,
                }
            ]
        )
        self.results = pd.concat([self.results, new_row], ignore_index=True)

    def record_benchmark(self, dt, benchmark_values: dict):
        """
        记录每个时间步的 benchmark 收盘价(绝对值)，并额外记录相对收益。
        :param dt: 当前日期
        :param benchmark_values: {symbol: close_price, ...}
        """
        # 1) 先保存绝对值(原先逻辑)
        record_abs = {"date": dt}
        # 2) 再保存相对收益(新增)
        record_rel = {"date": dt}

        for sym, close_price in benchmark_values.items():
            # 绝对值
            record_abs[sym] = close_price

            # 如果是第一次见到该 symbol，则记录其初始价格
            if sym not in self.benchmark_initial_values:
                self.benchmark_initial_values[sym] = close_price

            init_price = self.benchmark_initial_values[sym]
            # 防止除以0
            if init_price and init_price != 0:
                record_rel[sym] = close_price / init_price
            else:
                record_rel[sym] = float("nan")

        self.benchmark_results.append(record_abs)
        self.benchmark_relative_results.append(record_rel)

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
        daily_returns["returns_pct"] = daily_returns["returns_pct"] / 100.0
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
        只绘制“相对收益”曲线（组合与多个 benchmark），都从 1.0 起
        """
        # 1) 整理组合的相对收益
        df_port = self.results[["date", "relative_return"]].copy()
        df_port["date"] = pd.to_datetime(df_port["date"])
        df_port = df_port.sort_values("date").reset_index(drop=True)

        # 如果组合数据为空，直接返回
        if df_port.empty:
            print("No portfolio data to plot.")
            return

        # 2) 整理 benchmark 的相对收益
        df_bench_rel = pd.DataFrame(self.benchmark_relative_results)
        if not df_bench_rel.empty:
            df_bench_rel["date"] = pd.to_datetime(df_bench_rel["date"])
            df_bench_rel = df_bench_rel.sort_values("date").reset_index(drop=True)

        # 3) 合并到同一个表
        df_all = pd.merge(df_port, df_bench_rel, on="date", how="outer")
        # 将组合的列名改得直观一些
        df_all.rename(columns={"relative_return": "Portfolio"}, inplace=True)

        # 转成数值型，避免 plotly 出错
        for col in df_all.columns:
            if col != "date":
                df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

        # 4) 用 plotly.express 画线
        #    横轴：date；纵轴：组合和所有基准的相对收益，都从 1.0 起
        cols_to_plot = [c for c in df_all.columns if c != "date"]
        fig = px.line(
            df_all,
            x="date",
            y=cols_to_plot,
            title="Relative Return (Base=1)",
        )
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
