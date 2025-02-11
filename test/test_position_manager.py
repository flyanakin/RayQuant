import pytest
import pandas as pd
from datetime import datetime

# 导入待测试的类和方法
from core.strategy import Signal
from core.portfolio import Portfolio
from core.position_manager import EqualWeightPositionManager, get_min_lot
from core.datahub import Datahub


class TestDatahub(Datahub):
    """
    用于测试的 Datahub 实现，直接从内存中加载数据，不依赖外部文件。
    可通过设置 custom_bar_df 属性来指定测试数据集。
    """
    def load_bar_data(self):
        # 如果外部设置了 custom_bar_df 则直接使用，否则提供默认数据
        if hasattr(self, "custom_bar_df"):
            self.bar_df = self.custom_bar_df
        else:
            data = [
                {"trade_date": pd.Timestamp("2023-01-01"), "symbol": "688001.SH", "close": 55},
                {"trade_date": pd.Timestamp("2023-01-01"), "symbol": "000002.SH", "close": 20},
                {"trade_date": pd.Timestamp("2023-01-01"), "symbol": "123456", "close": 50},
            ]
            df = pd.DataFrame(data)
            self.bar_df = df.set_index(["trade_date", "symbol"])

    def load_fundamental_data(self):
        self.fundamental_df = pd.DataFrame()

    def load_info_data(self):
        self.info_df = pd.DataFrame()


# 固定的当前时间，方便测试判断
@pytest.fixture
def current_time():
    return pd.Timestamp("2023-01-01")


@pytest.fixture
def datahub():
    """
    构造一个测试用的 Datahub 实例，初始数据集包含：
      - '688001.SH'，close 值为 55
      - '000002.SH'，close 值为 20
      - '123456'，close 值为 50
    """
    data = [
        {"trade_date": pd.Timestamp("2023-01-01"), "symbol": "688001.SH", "close": 55},
        {"trade_date": pd.Timestamp("2023-01-01"), "symbol": "000002.SH", "close": 20},
        {"trade_date": pd.Timestamp("2023-01-01"), "symbol": "123456", "close": 50},
    ]
    df = pd.DataFrame(data).set_index(["trade_date", "symbol"])
    hub = TestDatahub(data_dict={"bar": {"path": "", "col_mapping": {"trade_date": "trade_date", "symbol": "symbol"}}})
    hub.custom_bar_df = df
    hub.load_bar_data()
    return hub


@pytest.fixture
def portfolio_with_sell():
    """
    构造仅用于卖出测试的组合：持有 '688001.SH' 1000 股，成本价为 50
    """
    portfolio = Portfolio(initial_cash=500000)
    portfolio.asset = pd.DataFrame(
        [{"asset": "688001.SH", "quantity": 1000, "cost_price": 50}]
    )
    return portfolio


@pytest.fixture
def portfolio_buy():
    """
    构造仅用于买入测试的组合：仅有现金 1000000，无持仓
    """
    return Portfolio(initial_cash=1000000)


def create_signal(df_data: dict, index_tuples: list, index_names: list, current_time: pd.Timestamp) -> Signal:
    """
    辅助函数：构造包含多层索引的 Signal 对象
    """
    index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
    df = pd.DataFrame(df_data, index=index)
    return Signal(df, current_time=current_time)


def test_sell_order(portfolio_with_sell, current_time, datahub):
    """
    测试卖出信号：
      - 组合中持有 '688001.SH' 1000 股，
      - 信号发出 SELL，
      - 预期生成卖出订单，数量为 1000。
    """
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

    # 校验订单：应生成一笔卖出订单，数量与持仓一致
    assert len(orders_df) == 1
    order = orders_df.iloc[0]
    assert order["side"] == "SELL"
    assert order["asset"] == "688001.SH"
    assert order["quantity"] == 1000


def test_buy_order_sufficient_cash(portfolio_buy, current_time, datahub):
    """
    测试买入信号（充足资金）：
      - 组合现金为 1000000，
      - 信号发出 BUY，对 '688001.SH'，close=50，
      - get_min_lot 返回 200，
        raw_qty = 1000000/50 = 20000，
        向下取整到200的整数倍后，订单数量应为 20000。
    """
    index_tuples = [(current_time, "688001.SH")]
    index_names = ["trade_date", "symbol"]
    close_price = 50
    df_data = {"close": [close_price], "signal": ["BUY"]}
    signal = create_signal(df_data, index_tuples, index_names, current_time)

    # 修改 datahub 数据，使 '688001.SH' 的 close 值为 50（覆盖初始测试数据）
    data = [{"trade_date": current_time, "symbol": "688001.SH", "close": 50}]
    df = pd.DataFrame(data).set_index(["trade_date", "symbol"])
    datahub.custom_bar_df = df
    datahub.load_bar_data()

    manager = EqualWeightPositionManager()
    order_obj = manager.transform_signals_to_orders(
        signals=signal,
        portfolio=portfolio_buy,
        data=datahub,
        current_time=current_time,
    )
    orders_df = order_obj.get()

    assert len(orders_df) == 1
    order = orders_df.iloc[0]
    assert order["side"] == "BUY"
    assert order["asset"] == "688001.SH"
    assert order["quantity"] == 20000


def test_buy_order_insufficient_cash(current_time, datahub):
    """
    测试买入信号（资金不足）：
      - 组合现金为 100，
      - 信号发出 BUY，对 '123456'，close=50，
      - get_min_lot 返回 100，
        raw_qty = 100/50 = 2，向下取整后为 0，不生成订单。
    """
    portfolio = Portfolio(initial_cash=100)
    index_tuples = [(current_time, "123456")]
    index_names = ["trade_date", "symbol"]
    close_price = 50
    df_data = {"close": [close_price], "signal": ["BUY"]}
    signal = create_signal(df_data, index_tuples, index_names, current_time)

    # 修改 datahub 数据，使 '123456' 的 close 值为 50
    data = [{"trade_date": current_time, "symbol": "123456", "close": 50}]
    df = pd.DataFrame(data).set_index(["trade_date", "symbol"])
    datahub.custom_bar_df = df
    datahub.load_bar_data()

    manager = EqualWeightPositionManager()
    order_obj = manager.transform_signals_to_orders(
        signals=signal,
        portfolio=portfolio,
        data=datahub,
        current_time=current_time,
    )
    orders_df = order_obj.get()

    # 由于资金不足且未满足最小交易单位，订单列表应为空
    assert orders_df.empty


def test_multiple_buy_signals(portfolio_buy, current_time, datahub):
    """
    测试多个买入信号：
      - 组合现金为 1000000，
      - 两个 BUY 信号：'688001.SH' 和 '000002.SH'，
      - 对于 '688001.SH'：get_min_lot 返回 200，close=50，
            allocated_cash = 500000，raw_qty = 500000/50 = 10000，
            订单数量应为 10000；
      - 对于 '000002.SH'：get_min_lot 返回 100，close=20，
            allocated_cash = 500000，raw_qty = 500000/20 = 25000，
            订单数量应为 25000；
      - 最终生成两个订单，各自数量符合预期。
    """
    index_tuples = [(current_time, "688001.SH"), (current_time, "000002.SH")]
    index_names = ["trade_date", "symbol"]
    df_data = {"close": [50, 20], "signal": ["BUY", "BUY"]}
    signal = create_signal(df_data, index_tuples, index_names, current_time)

    # 修改 datahub 数据，使 '688001.SH' 的 close 为 50，'000002.SH' 的 close 为 20
    data = [
        {"trade_date": current_time, "symbol": "688001.SH", "close": 50},
        {"trade_date": current_time, "symbol": "000002.SH", "close": 20},
    ]
    df = pd.DataFrame(data).set_index(["trade_date", "symbol"])
    datahub.custom_bar_df = df
    datahub.load_bar_data()

    manager = EqualWeightPositionManager()
    order_obj = manager.transform_signals_to_orders(
        signals=signal,
        portfolio=portfolio_buy,
        data=datahub,
        current_time=current_time,
    )
    orders_df = order_obj.get()

    assert len(orders_df) == 2

    # 校验 '688001.SH' 订单
    order_688 = orders_df[orders_df["asset"] == "688001.SH"].iloc[0]
    allocated_cash = portfolio_buy.cash / 2  # 每只标的分配500000
    raw_qty_688 = allocated_cash / 50         # 500000/50 = 10000
    assert order_688["quantity"] == 10000

    # 校验 '000002.SH' 订单
    order_000002 = orders_df[orders_df["asset"] == "000002.SH"].iloc[0]
    raw_qty_000002 = allocated_cash / 20        # 500000/20 = 25000
    assert order_000002["quantity"] == 25000