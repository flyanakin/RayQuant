import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# 假设你要测试的函数位于 portfolio_metrics.py 中
from utils.indicators import win_rate, annual_return, drawdown, annual_volatility, kelly_criterion


# 用于测试的 DummyPortfolio
class DummyPortfolio:
    def __init__(self, price_dict):
        self.price_dict = price_dict

    def get_asset_last_price(self, asset):
        return self.price_dict.get(asset, 0)


def test_win_rate_empty():
    """测试空交易记录，返回 0.0"""
    empty_df = pd.DataFrame(columns=['asset', 'trade_date', 'trade_qty', 'trade_price'])
    portfolio = DummyPortfolio({})
    assert win_rate(empty_df, portfolio) == 0.0


def test_win_rate_mixed_trades():
    """
    测试包含闭仓和平仓中的浮动盈亏的情况：
    - 资产 A：买入后卖出（闭仓），最后一笔交易卖出 100 单位，交易值 = -100*8 = -800（亏损）
    - 资产 B：先卖出开空后买回（闭仓），最后一笔交易买入 100 单位，交易值 = 100*8 = 800（盈利）
    - 资产 C：只有一笔买入，持仓未平，浮动盈亏 = (portfolio.last_price - trade_price)*100，
             如果 portfolio 返回 12，则浮动盈亏 = (12-10)*100 = 200（盈利）
    最终 win_count = 2, total_count = 3，胜率 = 2/3。
    """
    data = [
        {"asset": "A", "trade_date": "2025-01-01", "trade_qty": 100, "trade_price": 10},
        {"asset": "A", "trade_date": "2025-01-02", "trade_qty": -100, "trade_price": 8},
        {"asset": "B", "trade_date": "2025-01-01", "trade_qty": -100, "trade_price": 10},
        {"asset": "B", "trade_date": "2025-01-02", "trade_qty": 100, "trade_price": 8},
        {"asset": "C", "trade_date": "2025-01-03", "trade_qty": 100, "trade_price": 10},
    ]
    trade_log = pd.DataFrame(data)
    # DummyPortfolio 对于资产 C 返回最后价格 12；对于 A、B 不会用到
    portfolio = DummyPortfolio({"C": 12})
    rate = win_rate(trade_log, portfolio)
    assert pytest.approx(rate, rel=1e-3) == 2 / 3


def test_annual_return():
    """测试年化收益率计算"""
    # 当总天数为 365 天时，收益翻倍应为 100%（即收益率 1.0）
    ret = annual_return(100, 200, 365)
    assert pytest.approx(ret, rel=1e-3) == 1.0

    # 当总天数为 730 天时，年化收益率 = 2^(365/730)-1 = sqrt(2)-1
    ret2 = annual_return(100, 200, 730)
    expected = (2 ** (365 / 730)) - 1  # sqrt(2)-1
    assert pytest.approx(ret2, rel=1e-3) == expected


def test_annual_return_with_dataframe():
    # 构造一个包含起始和结束日期的 DataFrame
    df = pd.DataFrame(
        {"asset_value": [100, 200]},
        index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
    )
    # 根据日期计算总天数（注意这里使用 index 计算日期差）
    total_days = (df.index[-1] - df.index[0]).days
    # 根据公式计算预期的年化收益率
    expected_return = round((200 / 100) ** (365 / total_days) - 1, 4)

    # 调用函数时传入 df 参数，其他参数的值可以随意传入，因为函数内部会重新赋值
    result = annual_return(start_value=0, end_value=0, total_days=0, df=df)

    # 断言返回结果与预期一致
    assert result == expected_return


def test_drawdown():
    """
    测试回撤计算：
    构造一个时间序列，数值如下：
        日期：2025-01-01 至 2025-01-10
        资产价值：100, 110, 105, 120, 115, 130, 125, 140, 135, 150
    计算过程：
        对于每个数据点，先计算累积最大值，再计算回撤比例。
        此序列中最大回撤出现在 2025-01-03 时，回撤比例 ≈ (110-105)/110 ≈ 0.04545
    因为数据区间不足 1 个月，所以整体只有一个区间，区间为 (2025-01-01, 2025-01-10)。
    """
    dates = pd.date_range(start="2025-01-01", periods=10, freq="D")
    values = [100, 110, 105, 120, 115, 130, 125, 140, 135, 150]
    df = pd.DataFrame(values, index=dates, columns=["value"])

    # 调用修改后的drawdown函数，返回DataFrame和整体最大回撤及其所在区间
    interval_drawdowns_df, (overall_max_dd, max_interval) = drawdown(df, interval_months=1)

    # 检查整体最大回撤值是否正确
    assert pytest.approx(overall_max_dd, rel=1e-3) == 0.0455

    # 检查整体最大回撤所在的区间
    expected_interval = (dates[0], dates[-1])
    assert max_interval == expected_interval

    # 检查返回的DataFrame的索引为MultiIndex，且只有一个区间
    assert isinstance(interval_drawdowns_df.index, pd.MultiIndex)
    assert len(interval_drawdowns_df) == 1
    assert expected_interval in interval_drawdowns_df.index

    # 检查DataFrame中对应区间的回撤值
    dd_value = interval_drawdowns_df.loc[expected_interval, 'drawdown']
    assert pytest.approx(dd_value, rel=1e-3) == 0.0455


def test_annual_volatility():
    """
    测试年化波动率：
    构造一个表示每日收益率的序列，计算标准差后年化（使用 250 个交易日）。
    例如，对于序列 [0.01, -0.02, 0.015, -0.005, 0.02]，
    使用样本标准差（ddof=1）计算，再乘以 sqrt(250) 得到年化波动率。
    """
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    returns = [0.01, -0.02, 0.015, -0.005, 0.02]
    df = pd.DataFrame(returns, index=dates, columns=["return"])
    daily_std = np.std(returns, ddof=1)
    expected_annual_vol = daily_std * np.sqrt(250)
    vol = annual_volatility(df)
    assert pytest.approx(vol, rel=1e-3) == expected_annual_vol


def test_zero_winning_reward():
    """
    测试当 winning_reward 为 0 时，函数返回 0.0。
    """
    result = kelly_criterion(0.5, 0, 1)
    assert result == 0.0

def test_zero_losing_reward():
    """
    测试当 losing_reward 为 0 时，函数返回 1.0。
    注意：即使传入的 losing_reward 为 0，内部会取 abs(0) 后仍为 0。
    """
    result = kelly_criterion(0.5, 2, 0)
    assert result == 1.0

def test_negative_losing_reward():
    """
    测试当传入负的 losing_reward 时，内部会取绝对值，
    与正常正值的 losing_reward 结果相同。
    例如：winning_rate=0.5, winning_reward=2, losing_reward=-1，
    结果应为 0.25。
    """
    result = kelly_criterion(0.5, 2, -1)
    assert result == 0.25