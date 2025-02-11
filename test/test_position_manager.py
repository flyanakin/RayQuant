import pytest
import pandas as pd
from datetime import datetime

# 导入待测试的类和方法
from core.strategy import Signal
from core.portfolio import Portfolio
from core.position_manager import EqualWeightPositionManager, get_min_lot


# 构造一个简单的 DummyDatahub，因为 transform_signals_to_orders 接口中需要 data 参数，但本例中未使用
class DummyDatahub:
    pass


# 统一使用一个固定的时间，便于测试结果判断
@pytest.fixture
def current_time():
    return pd.Timestamp("2023-01-01")


@pytest.fixture
def datahub():
    return DummyDatahub()


# 构造一个仅包含卖出情形的组合
@pytest.fixture
def portfolio_with_sell():
    portfolio = Portfolio(initial_cash=500000)
    # 设置持仓：假设持有 '688001.SH' 1000 股，成本价为 50
    portfolio.asset = pd.DataFrame(
        [{"asset": "688001.SH", "quantity": 1000, "cost_price": 50}]
    )
    return portfolio


# 构造一个用于买入情形的组合（无持仓，仅有现金）
@pytest.fixture
def portfolio_buy():
    return Portfolio(initial_cash=1000000)


def create_signal(
    df_data: dict, index_tuples: list, index_names: list, current_time: pd.Timestamp
) -> Signal:
    """
    辅助函数：构造包含多层索引的 Signal 对象
    """
    index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
    df = pd.DataFrame(df_data, index=index)
    return Signal(df, current_time=current_time)


def test_sell_order(portfolio_with_sell, current_time, datahub):
    """
    测试卖出信号：
      - 组合中持有资产 '688001.SH' 数量为 1000，
      - 信号中对 '688001.SH' 发出 SELL 信号，
      - 预期生成一笔卖出订单，数量为 1000。
    """
    # 构造信号，索引包含 trade_date 和 symbol
    index_tuples = [(current_time, "688001.SH")]
    index_names = ["trade_date", "symbol"]
    df_data = {"close": [55], "signal": ["SELL"]}
    signal = create_signal(df_data, index_tuples, index_names, current_time)

    manager = EqualWeightPositionManager()
    order_obj = manager.transform_signals_to_orders(
        signals=signal,
        portfolio=portfolio_with_sell,
        data=datahub,
        current_time=current_time,
    )
    orders_df = order_obj.get()

    # 应生成一笔卖出订单
    assert len(orders_df) == 1
    order = orders_df.iloc[0]
    assert order["side"] == "SELL"
    assert order["asset"] == "688001.SH"
    assert order["quantity"] == 1000


def test_buy_order_sufficient_cash(portfolio_buy, current_time, datahub):
    """
    测试买入信号（充足资金）：
      - 组合现金为 1000000，
      - 信号中对 '688001.SH' 发出 BUY 信号（close=50），
      - 对于 '688001.SH'，根据 get_min_lot 判断最小手数为 200，
        allocated_cash = 1000000, raw_qty = 1000000/50 = 20000，
        向下取整到 200 的整数倍后订单数量应为 20000。
    """
    index_tuples = [(current_time, "688001.SH")]
    index_names = ["trade_date", "symbol"]
    close_price = 50
    df_data = {"close": [close_price], "signal": ["BUY"]}
    signal = create_signal(df_data, index_tuples, index_names, current_time)

    manager = EqualWeightPositionManager()
    order_obj = manager.transform_signals_to_orders(
        signals=signal, portfolio=portfolio_buy, data=datahub, current_time=current_time
    )
    orders_df = order_obj.get()

    assert len(orders_df) == 1
    order = orders_df.iloc[0]
    assert order["side"] == "BUY"
    assert order["asset"] == "688001.SH"
    # allocated_cash = 1000000, raw_qty = 1000000/50 = 20000
    # 向下取整到 200 的倍数：int(20000//200)*200 = 20000
    assert order["quantity"] == 20000


def test_buy_order_insufficient_cash(current_time, datahub):
    """
    测试买入信号（资金不足）：
      - 组合现金为 100，
      - 信号中对 '123456' 发出 BUY 信号（close=50），
      - 对于 '123456'，get_min_lot 返回 100（因为不以 '688' 开头），
        allocated_cash = 100, raw_qty = 100/50 = 2，
        向下取整到 100 的倍数后得到 0，预期不生成订单。
    """
    portfolio = Portfolio(initial_cash=100)
    index_tuples = [(current_time, "123456")]
    index_names = ["trade_date", "symbol"]
    close_price = 50
    df_data = {"close": [close_price], "signal": ["BUY"]}
    signal = create_signal(df_data, index_tuples, index_names, current_time)

    manager = EqualWeightPositionManager()
    order_obj = manager.transform_signals_to_orders(
        signals=signal, portfolio=portfolio, data=datahub, current_time=current_time
    )
    orders_df = order_obj.get()

    # 未满足最小交易单位，不应生成任何订单
    assert orders_df.empty


def test_multiple_buy_signals(portfolio_buy, current_time, datahub):
    """
    测试多个买入信号：
      - 组合现金为 1000000，
      - 信号中包含两个 BUY 信号：'688001.SH' 和 '000002.SH'，
      - 对于 '688001.SH'，get_min_lot 返回 200，close=50，allocated_cash=500000，
            raw_qty = 500000/50 = 10000，订单数量 = 10000；
      - 对于 '000002.SH'，get_min_lot 返回 100，close=20，allocated_cash=500000，
            raw_qty = 500000/20 = 25000，订单数量 = 25000；
      - 最终应生成两个订单，各自订单数量符合计算规则。
    """
    index_tuples = [(current_time, "688001.SH"), (current_time, "000002.SH")]
    index_names = ["trade_date", "symbol"]
    df_data = {"close": [50, 20], "signal": ["BUY", "BUY"]}
    signal = create_signal(df_data, index_tuples, index_names, current_time)

    manager = EqualWeightPositionManager()
    order_obj = manager.transform_signals_to_orders(
        signals=signal, portfolio=portfolio_buy, data=datahub, current_time=current_time
    )
    orders_df = order_obj.get()

    # 应生成两个订单
    assert len(orders_df) == 2

    # 对于 '688001.SH'
    order_688 = orders_df[orders_df["asset"] == "688001.SH"].iloc[0]
    allocated_cash = portfolio_buy.cash / 2  # 每个买入标的分配 500000
    raw_qty_688 = allocated_cash / 50  # 500000/50 = 10000
    # 向下取整到 200 的倍数：int(10000//200)*200 = 10000
    assert order_688["quantity"] == 10000

    # 对于 '000002.SH'
    order_000002 = orders_df[orders_df["asset"] == "000002.SH"].iloc[0]
    raw_qty_000002 = allocated_cash / 20  # 500000/20 = 25000
    # 向下取整到 100 的倍数：int(25000//100)*100 = 25000
    print(orders_df)
    assert order_000002["quantity"] == 25000
