# test_portfolio.py
import pytest
import pandas as pd
from core.portfolio import Portfolio


def test_portfolio_buy_and_sell():
    portfolio = Portfolio(initial_cash=10000)

    # 1) 买入测试
    portfolio.buy(
        asset="600000.SH", trade_date="2023-01-02", trade_qty=100, trade_price=10.0
    )
    # 期望：花费100*10=1000，剩余现金=9000
    assert portfolio.cash == 9000
    # 持仓应该有1条记录
    assert len(portfolio.asset) == 1
    row = portfolio.asset.iloc[0]
    assert row["asset"] == "600000.SH"
    assert row["quantity"] == 100
    assert row["cost_price"] == 10.0

    # 交易日志
    assert len(portfolio.trade_log) == 1
    log_row = portfolio.trade_log.iloc[0]
    assert log_row["asset"] == "600000.SH"
    assert log_row["trade_date"] == "2023-01-02"
    assert log_row["trade_qty"] == 100
    assert log_row["trade_price"] == 10.0

    # 2) 卖出测试（部分卖出）
    portfolio.sell(
        asset="600000.SH", trade_date="2023-01-03", trade_qty=50, trade_price=12.0
    )
    # 卖出进账 = 50*12=600, 现金 -> 9000+600=9600
    assert portfolio.cash == 9600
    # 剩余仓位
    row = portfolio.asset.iloc[0]
    assert row["quantity"] == 50  # 剩余50

    # 交易日志应再加一条
    assert len(portfolio.trade_log) == 2
    sell_log = portfolio.trade_log.iloc[1]
    assert sell_log["trade_date"] == "2023-01-03"
    assert sell_log["trade_qty"] == -50
    assert sell_log["trade_price"] == 12.0

    # 3) 卖出剩余全部仓位
    portfolio.sell(
        asset="600000.SH", trade_date="2023-01-04", trade_qty=50, trade_price=8.0
    )
    # 全部卖出 -> 剩余仓位应被删除
    assert len(portfolio.asset) == 0


def test_portfolio_total_value():
    # 测试 total_value 的正确性
    portfolio = Portfolio(initial_cash=10000)
    portfolio.buy(
        asset="600000.SH", trade_date="2023-01-02", trade_qty=100, trade_price=10.0
    )

    # 构造一个行情表(只有一行)
    data = pd.DataFrame({"close": [12.0]}, index=["2023-01-03"])

    # 组合总市值 = 现金(9000) + 持仓市值(100*12=1200)=10200
    val = portfolio.total_value("2023-01-03", data)
    assert val == 10200
