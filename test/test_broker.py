import pytest
import pandas as pd
from core.broker import Broker
from core.portfolio import Portfolio


def test_broker_execute_orders():
    broker = Broker()
    portfolio = Portfolio(initial_cash=10000)
    # 构造订单表
    orders_df = pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "asset": "600000.SH",
                "side": "BUY",
                "quantity": 200,
            },
            {
                "date": "2023-01-03",
                "asset": "600000.SH",
                "side": "SELL",
                "quantity": 100,
            },
        ]
    )

    # 构造行情(简化只有一只标的)
    data = pd.DataFrame({"close": [10.0, 12.0]}, index=["2023-01-02", "2023-01-03"])

    broker.execute_orders(orders_df, data, portfolio)

    # 检查执行结果
    # 第一天买入: 200*10=2000 -> cash=8000, quantity=200
    # 第二天卖出: 100*12=1200 -> cash=8000+1200=9200, quantity=100
    assert portfolio.cash == 9200
    asset_row = portfolio.asset.iloc[0]
    assert asset_row["asset"] == "600000.SH"
    assert asset_row["quantity"] == 100
    # 交易日志2条
    assert len(portfolio.trade_log) == 2
