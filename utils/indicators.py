import numpy as np
import pandas as pd
from core.portfolio import Portfolio


def win_rate(
        trade_log: pd.DataFrame,
        portfolio: Portfolio
) -> float:
    """
    根据交易记录计算交易胜率。

    :param trade_log: DataFrame，字段要求，包含
                        asset：资产标识
                        trade_date：交易日期
                        trade_qty：成交数量，为负数是表示卖出
                        trade_price：成交价格
    :param portfolio: Portfolio 对象，用于获取最新价格。
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
        last_trade_with_position['last_price'] = last_trade_with_position['asset'].apply(
            portfolio.get_asset_last_price)
        last_trade_with_position['floating_profit'] = (
                (last_trade_with_position['last_price'] - last_trade_with_position['trade_price']) *
                last_trade_with_position['cumulative_qty']
        )

        # 统计浮盈的未平仓交易
        floating_wins = (last_trade_with_position['floating_profit'] > 0).sum()
        win_count += floating_wins
        total_count += len(last_trade_with_position)

    return win_count / total_count if total_count > 0 else 0.0


def annual_return(
        start_value: float,
        end_value: float,
        total_days: int
) -> float:
    """
    计算年化收益率。
    :param start_value: 起始价值
    :param end_value: 结束价值
    :param total_days: 总天数，根据自然年转换得到复利周期
    """
    annualized_return = ((end_value / start_value) ** (365 / total_days)) - 1 if start_value > 0 else 0
    return annualized_return


def drawdown(
        df: pd.DataFrame,
        interval_months: int
) -> tuple:
    """
    根据时间序列计算区间每个区间的回撤，并给出最大回撤值和总最大回撤的所在的区间。
    :param df: DataFrame，字段要求，包含
                index: 日期，按日
                value: 字段名不做要求，表示资产价值即可
    :param interval_months: 区间长度，单位为月，默认 3 个月。
    :return: (最大回撤百分比, 对应区间)
    """
    # 确保按照日期排序
    df = df.sort_index()

    # 假设资产价值在第一列
    asset_series = df.iloc[:, 0]

    # 获取数据起始和结束日期
    start_date = asset_series.index.min()
    end_date = asset_series.index.max()

    interval_drawdowns = {}

    current_start = start_date
    while current_start <= end_date:
        # 计算当前区间的结束日期：起始日 + interval_months 个月 - 1 天
        current_end = current_start + pd.DateOffset(months=interval_months) - pd.Timedelta(days=1)
        if current_end > end_date:
            current_end = end_date

        # 截取当前区间内的数据
        interval_data = asset_series.loc[current_start:current_end]

        # 计算回撤：先计算累计最大值，再计算回撤百分比
        if not interval_data.empty:
            cummax = interval_data.cummax()
            dd_series = (cummax - interval_data) / cummax
            max_dd = dd_series.max()
        else:
            max_dd = 0  # 如果数据为空，则回撤设为0

        # 记录区间及其最大回撤
        interval_drawdowns[(current_start, current_end)] = max_dd

        # 下一区间的起始日期为当前结束日期的下一天
        current_start = current_end + pd.Timedelta(days=1)

    # 找出整体最大回撤及其对应的区间
    max_interval = max(interval_drawdowns, key=lambda x: interval_drawdowns[x])
    overall_max_dd = interval_drawdowns[max_interval]

    return interval_drawdowns, (overall_max_dd, max_interval)


def annual_volatility(
        df: pd.DataFrame
) -> float:
    """
    计算年化波动率，基于收益率的标准差。
    :param df: DataFrame，字段要求，包含
                index: 日期，按日
                value: 字段名不做要求，表示每日回报率即可
    """
    # 确保数据按照日期排序
    df = df.sort_index()

    # 假设每日回报率在第一列
    daily_returns = df.iloc[:, 0]

    # 计算每日收益率标准差
    daily_std = daily_returns.std()

    # 使用250个交易日将日波动率年化
    annual_vol = daily_std * np.sqrt(250)

    return annual_vol
