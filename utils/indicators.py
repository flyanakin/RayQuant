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
    global _rate
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
        _rate = win_count / total_count if total_count > 0 else 0.0

    return round(_rate, 4)


def annual_return(
        start_value: float,
        end_value: float,
        total_days: int,
        df: pd.DataFrame = None,
) -> float:
    """
    计算年化收益率。
    :param start_value: 起始价值
    :param end_value: 结束价值
    :param total_days: 总天数，根据自然年转换得到复利周期
    :param df: DataFrame，总价值的时间序列，若输入df时，直接自动计算。字段要求，包含
                index: 日期，按日
                value: 字段名不做要求，表示资产价值即可
    :return
        回报率：浮点数，保留小数点后4位
    """
    if df is not None:
        df.sort_index(ascending=True)
        total_days = (df.index[-1] - df.index[0]).days
        start_value = df.iloc[0, 0]
        end_value = df.iloc[-1, 0]

    annualized_return = ((end_value / start_value) ** (365 / total_days)) - 1 if start_value > 0 else 0
    return round(annualized_return, 4)


def drawdown(
        df: pd.DataFrame,
        interval_months: int
) -> tuple[pd.DataFrame, tuple]:
    """
    根据时间序列计算每个区间内的最大回撤，并返回一个DataFrame，
    其中区间的起始日期和结束日期作为MultiIndex，同时返回整体最大回撤及其对应区间。
    :param df: DataFrame，字段要求，包含
                index: 日期，按日
                value: 字段名不做要求，表示资产价值即可
    :param interval_months: 区间长度，单位为月，默认 3 个月。
    :return: (interval_drawdowns_df, (最大回撤百分比, 对应区间))
             interval_drawdowns_df: DataFrame，MultiIndex为['start_date','end_date']，
             列为['drawdown']，表示每个区间的最大回撤百分比；
             (最大回撤百分比, 对应区间): tuple，第一个元素为所有区间中的最大回撤百分比，
             第二个元素为对应区间的 (起始日期, 结束日期)。
    """
    # 确保按照日期排序
    df = df.sort_index()

    # 假设资产价值在第一列
    asset_series = df.iloc[:, 0]

    # 获取数据起始和结束日期
    start_date = asset_series.index.min()
    end_date = asset_series.index.max()

    intervals = []

    current_start = start_date
    while current_start <= end_date:
        # 计算当前区间的结束日期：起始日 + interval_months个月 - 1天
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

        intervals.append((current_start, current_end, max_dd))

        # 下一区间的起始日期为当前结束日期的下一天
        current_start = current_end + pd.Timedelta(days=1)

        # 构造DataFrame，使用多重索引 (起始日期, 结束日期)
    interval_drawdowns_df = pd.DataFrame(intervals, columns=['start_date', 'end_date', 'drawdown'])
    interval_drawdowns_df['drawdown'] = interval_drawdowns_df['drawdown'].round(4)
    interval_drawdowns_df.set_index(['start_date', 'end_date'], inplace=True)

    # 找出整体最大回撤及其对应的区间
    max_row = interval_drawdowns_df['drawdown'].idxmax()
    overall_max_dd = interval_drawdowns_df['drawdown'].max()

    return interval_drawdowns_df, (overall_max_dd, max_row)


def annual_volatility(
        df: pd.DataFrame
) -> float:
    """
    计算年化波动率，基于收益率的标准差。
    :param df: DataFrame，字段要求，包含
                index: 日期，按日
                value: 字段名不做要求，表示每日回报率即可
    :return 浮点数，保留小数点后四位
    """
    # 确保数据按照日期排序
    df = df.sort_index()

    # 假设每日回报率在第一列
    daily_returns = df.iloc[:, 0]

    # 计算每日收益率标准差
    daily_std = daily_returns.std()

    # 使用250个交易日将日波动率年化
    annual_vol = daily_std * np.sqrt(250)

    return round(annual_vol, 4)


def kelly_criterion(
        winning_rate: float,
        winning_reward: float,
        losing_reward: float,
        losing_rate: float = None,
) -> float:
    """
    凯利公式计算，输出最佳投注比例
    :param winning_rate:
    :param losing_rate: 失败概率（通常与 winning_rate 之和为1）
    :param winning_reward:
    :param losing_reward:
    :return:
        f
    """
    if losing_rate is None:
        losing_rate = 1 - winning_rate

    # 保证losing_reward为正，因为losing_reward表示净亏损
    losing_reward = abs(losing_reward)

    if winning_reward == 0:
        return 0.0

    if losing_reward == 0:
        # 净损失为0的时候，控制投注不超过100%
        return 1.0

    f = winning_rate / losing_reward - losing_rate / winning_reward
    return round(f, 4)


def compute_future_return(df: pd.DataFrame,
                          future_days: int,
                          direction: str = 'long'
) -> pd.DataFrame:
    """
    根据输入 df 中的 price 字段计算未来收益率
    :param direction:
    :param df: index为日期，并已做升序拍排序，price列为交易价格
    :param future_days: N个交易日
    :param direction: 交易方向，做多做空
    :return
          df: 增加future_return、end_date、end_price三列，并且删除没有未来价格的行
    """
    df = df.copy()
    # 计算未来第N天的日期与价格
    df['end_date'] = df.index.to_series().shift(-future_days)
    df['end_price'] = df['price'].shift(-future_days)

    # 根据方向计算收益率：
    # 做多时收益率 = (未来价格 / 当前价格 - 1)
    # 做空时收益率 = (当前价格 / 未来价格 - 1)
    if direction == 'long':
        df['future_return'] = df['end_price'] / df['price'] - 1
    elif direction == 'short':
        df['future_return'] = df['price'] / df['end_price'] - 1
    else:
        raise ValueError("direction 参数必须为 'long' 或 'short'")

    df.dropna(inplace=True)
    return df
