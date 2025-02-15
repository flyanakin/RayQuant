import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from core.portfolio import Portfolio


class Observer:
    def __init__(self,
                 portfolio: Portfolio,
                 benchmark: Optional[pd.DataFrame] = None):
        """
        观察者模块，用于监控和记录回测过程中的数据与指标。

        :param portfolio: Portfolio 对象，包含交易和持仓信息。
        :param benchmark: 基准曲线，pandas DataFrame，index 为日期，columns 为标的符号及其价格。
        """
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.results = pd.DataFrame(columns=['date', 'total_value', 'cash', 'returns', 'returns_pct'])
        self.performance_metrics = {}

    def record(self, dt, total_value, cash):
        """
        记录每个时间步的回测结果。

        :param dt: 当前日期。
        :param total_value: 组合总价值。
        :param cash: 当前现金。
        """
        previous_value = self.results['total_value'].iloc[-1] if not self.results.empty else total_value
        returns = total_value - previous_value
        returns_pct = (returns / previous_value) * 100 if previous_value != 0 else 0

        new_row = pd.DataFrame(
            [{'date': dt, 'total_value': total_value, 'cash': cash, 'returns': returns, 'returns_pct': returns_pct}]
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
        trade_log = self.portfolio.trade_log
        self.performance_metrics['win_rate'] = self._calculate_win_rate(trade_log)
        self.performance_metrics['annual_return'] = self._calculate_annual_return()
        self.performance_metrics['max_drawdown'], self.performance_metrics[
            'max_drawdown_interval'] = self._calculate_max_drawdown(interval_months)
        self.performance_metrics['annual_volatility'] = self._calculate_annual_volatility()

    def _calculate_win_rate(self, trade_log: pd.DataFrame) -> float:
        """
        通过向量化操作计算交易胜率，减少 for 循环，提升效率。

        :param trade_log: 包含资产、交易日期、交易数量和交易价格的 DataFrame
                          trade_qty > 0 表示买入，trade_qty < 0 表示卖出。
        :return: 交易胜率（0-1 之间的小数）。
        """
        if trade_log.empty:
            return 0.0

            # 确保 trade_qty 为数值型
        trade_log['trade_qty'] = pd.to_numeric(trade_log['trade_qty'], errors='coerce').fillna(0)

        # 计算累计持仓数量
        trade_log['cumulative_qty'] = trade_log.groupby('asset')['trade_qty'].cumsum()

        # 找到所有完整平仓交易
        closed_trades = trade_log[trade_log['cumulative_qty'] == 0].copy()
        closed_trades['trade_value'] = closed_trades['trade_qty'] * closed_trades['trade_price']

        # 计算每个资产的盈亏
        closed_trade_pnl = closed_trades.groupby('asset')['trade_value'].sum()

        # 统计盈利的交易数量和总交易数量
        win_count = (closed_trade_pnl > 0).sum()
        total_count = closed_trade_pnl.count()

        # 检查未平仓头寸
        last_trade = trade_log.groupby('asset').tail(1)  # 获取每个资产的最后一笔交易
        last_trade_with_position = last_trade[last_trade['cumulative_qty'] > 0]

        if not last_trade_with_position.empty:
            # 获取每个资产的最后价格，计算浮动盈亏
            last_trade_with_position['last_price'] = last_trade_with_position['asset'].apply(self.portfolio.get_asset_last_price)
            last_trade_with_position['floating_profit'] = (
                    (last_trade_with_position['last_price'] - last_trade_with_position['trade_price']) *
                    last_trade_with_position['cumulative_qty']
            )

            # 统计浮盈的未平仓交易
            floating_wins = (last_trade_with_position['floating_profit'] > 0).sum()
            win_count += floating_wins
            total_count += len(last_trade_with_position)

        return win_count / total_count if total_count > 0 else 0.0

    def _calculate_annual_return(self) -> float:
        """
        计算年化收益率。
        """
        start_value = self.results['total_value'].iloc[0]
        end_value = self.results['total_value'].iloc[-1]
        total_days = (self.results['date'].iloc[-1] - self.results['date'].iloc[0]).days
        annualized_return = ((end_value / start_value) ** (365 / total_days)) - 1 if start_value > 0 else 0
        return annualized_return

    def _calculate_max_drawdown(self, interval_months: int) -> tuple:
        """
        计算区间最大回撤和总最大回撤。

        :param interval_months: 区间长度，单位为月，默认 3 个月。
        :return: (最大回撤百分比, 对应区间)
        """
        max_drawdown = 0
        max_drawdown_interval = None
        # 示例实现逻辑
        for i in range(len(self.results)):
            peak = self.results['total_value'][:i + 1].max()
            drawdown = (peak - self.results['total_value'].iloc[i]) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_interval = (self.results['date'].iloc[i - interval_months], self.results['date'].iloc[i])
        return max_drawdown, max_drawdown_interval

    def _calculate_annual_volatility(self) -> float:
        """
        计算年化波动率，基于收益率的标准差。
        """
        daily_returns = self.results['returns_pct'] / 100
        annual_volatility = daily_returns.std() * (250 ** 0.5)
        return annual_volatility

    def plot_results(self):
        """
        绘制收益率曲线和基准曲线（如果有）。
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['date'], self.results['total_value'], label='Portfolio')
        if self.benchmark is not None:
            for symbol in self.benchmark.columns:
                plt.plot(self.benchmark.index, self.benchmark[symbol], label=f'Benchmark: {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Total Value')
        plt.title('Portfolio Value and Benchmark')
        plt.legend()
        plt.show()

    def print_metrics(self):
        """
        打印计算的评价指标。
        """
        for metric, value in self.performance_metrics.items():
            print(f"{metric}: {value:.2%}" if isinstance(value, float) else f"{metric}: {value}")