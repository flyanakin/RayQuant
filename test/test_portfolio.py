import pytest
import pandas as pd
from core.portfolio import Portfolio
from core.broker import Order


# Dummy Datahub，用于测试 get_asset_value 和 total_value
class DummyDatahub:
    def get_bars(self, current_date):
        """
        模拟在给定日期能拿到的行情数据。
        这里简单返回一个包含 close=150 的 DataFrame，
        方便测试时让持仓“看上去”都有价格可用。
        """
        # 你也可以根据 current_date 做一些判断，决定返回不同的价格
        return pd.DataFrame([{'close': 150}])

# Dummy Datahub，始终返回空 DataFrame，用于测试当行情数据缺失时的逻辑
class DummyEmptyDatahub:
    def get_bars(self, current_date):
        """
        模拟在给定日期拿不到任何行情数据（空 DataFrame），
        用于测试 fallback 到 current_price 的逻辑。
        """
        return pd.DataFrame()


def test_buy_new_asset():
    """测试首次买入一个新标的"""
    portfolio = Portfolio(initial_cash=10000)
    order_data = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [50],
        'trade_price': [100]
    })
    order = Order(order_data)
    portfolio.buy(order)

    # 检查现金余额：10000 - 50*100 = 5000
    assert portfolio.cash == 5000

    # 检查持仓记录
    assert not portfolio.asset.empty
    row = portfolio.asset.iloc[0]
    assert row['asset'] == 'A'
    assert row['quantity'] == 50
    assert row['cost_price'] == 100
    assert row['current_price'] == 100

    # 检查交易日志
    assert not portfolio.trade_log.empty
    trade_log_row = portfolio.trade_log.iloc[0]
    assert trade_log_row['asset'] == 'A'
    assert trade_log_row['trade_date'] == pd.Timestamp('2023-01-01')
    assert trade_log_row['trade_qty'] == 50
    assert trade_log_row['trade_price'] == 100


def test_buy_existing_asset():
    """测试对同一标的多次买入，检查数量和加权成本价更新"""
    portfolio = Portfolio(initial_cash=20000)
    # 第一次买入：50 股，单价 100
    order_data1 = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [50],
        'trade_price': [100]
    })
    order1 = Order(order_data1)
    portfolio.buy(order1)

    # 第二次买入：再买入 50 股，单价 200
    order_data2 = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-02')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [50],
        'trade_price': [200]
    })
    order2 = Order(order_data2)
    portfolio.buy(order2)

    # 现金：20000 - (50*100 + 50*200) = 20000 - 15000 = 5000
    assert portfolio.cash == 5000

    # 持仓：总数量 100 股，加权成本价 = (50*100 + 50*200)/100 = 150
    row = portfolio.asset.iloc[0]
    assert row['asset'] == 'A'
    assert row['quantity'] == 100
    assert row['cost_price'] == 150
    assert row['current_price'] == 150

    # 交易日志应有两条记录
    assert len(portfolio.trade_log) == 2


def test_buy_insufficient_cash():
    """测试资金不足时买入应抛出异常"""
    portfolio = Portfolio(initial_cash=1000)
    # 试图买入 20 股，每股 100，总成本 2000，超过资金
    order_data = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [20],
        'trade_price': [100]
    })
    order = Order(order_data)
    with pytest.raises(ValueError, match="现金不足"):
        portfolio.buy(order)


def test_sell_partial():
    """测试部分卖出"""
    portfolio = Portfolio(initial_cash=5000)
    # 先买入 50 股，每股 100，使现金变为 5000 - 5000 = 0
    order_data_buy = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [50],
        'trade_price': [100]
    })
    order_buy = Order(order_data_buy)
    portfolio.buy(order_buy)

    # 部分卖出 20 股，每股 110，获得现金 20*110 = 2200
    order_data_sell = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-02')],
        'asset': ['A'],
        'side': ['SELL'],
        'quantity': [20],
        'trade_price': [110]
    })
    order_sell = Order(order_data_sell)
    portfolio.sell(order_sell)

    # 检查现金余额
    assert portfolio.cash == 2200

    # 持仓剩余 30 股，且当前价格更新为 110
    row = portfolio.asset.iloc[0]
    assert row['quantity'] == 30
    assert row['current_price'] == 110

    # 交易日志应有两条记录，其中卖出记录数量为负
    assert len(portfolio.trade_log) == 2
    sell_log = portfolio.trade_log.iloc[1]
    assert sell_log['trade_qty'] == -20


def test_sell_all():
    """测试全仓卖出时持仓记录被移除"""
    portfolio = Portfolio(initial_cash=10000)
    # 买入 30 股，每股 100，现金变为 10000 - 3000 = 7000
    order_data_buy = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [30],
        'trade_price': [100]
    })
    order_buy = Order(order_data_buy)
    portfolio.buy(order_buy)

    # 全部卖出 30 股，每股 120，获得现金 30*120 = 3600
    order_data_sell = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-02')],
        'asset': ['A'],
        'side': ['SELL'],
        'quantity': [30],
        'trade_price': [120]
    })
    order_sell = Order(order_data_sell)
    portfolio.sell(order_sell)

    # 现金：7000 + 3600 = 10600
    assert portfolio.cash == 10600
    # 持仓应已清空
    assert portfolio.asset.empty
    # 检查交易日志中卖出记录数量为 -30
    sell_log = portfolio.trade_log.iloc[1]
    assert sell_log['trade_qty'] == -30


def test_sell_nonexistent_asset():
    """测试卖出未持有标的时应抛出异常"""
    portfolio = Portfolio(initial_cash=10000)
    order_data_sell = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['B'],  # 标的不在持仓中
        'side': ['SELL'],
        'quantity': [10],
        'trade_price': [100]
    })
    order_sell = Order(order_data_sell)
    with pytest.raises(ValueError, match="持仓中不存在标的"):
        portfolio.sell(order_sell)


def test_sell_insufficient_quantity():
    """测试卖出数量超过持仓时应抛出异常"""
    portfolio = Portfolio(initial_cash=10000)
    # 买入 10 股
    order_data_buy = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [10],
        'trade_price': [100]
    })
    order_buy = Order(order_data_buy)
    portfolio.buy(order_buy)

    # 尝试卖出 20 股，超过持仓数量
    order_data_sell = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-02')],
        'asset': ['A'],
        'side': ['SELL'],
        'quantity': [20],
        'trade_price': [100]
    })
    order_sell = Order(order_data_sell)
    with pytest.raises(ValueError, match="持仓数量不足"):
        portfolio.sell(order_sell)


def test_get_asset_value():
    """测试基于行情数据计算持仓市值"""
    portfolio = Portfolio(initial_cash=10000)
    # 买入 10 股，每股 100
    order_data_buy = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [10],
        'trade_price': [100]
    })
    order_buy = Order(order_data_buy)
    portfolio.buy(order_buy)

    dummy = DummyDatahub()
    # Dummy 返回 close=150，则市值应为 10*150 = 1500
    asset_value = portfolio.get_asset_value(pd.Timestamp('2023-01-01'), dummy)
    assert asset_value == 1500


def test_get_asset_value_with_empty_bar():
    """测试当行情数据缺失时使用持仓记录中的 current_price"""
    portfolio = Portfolio(initial_cash=10000)
    # 买入 10 股，每股 100
    order_data_buy = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [10],
        'trade_price': [100]
    })
    order_buy = Order(order_data_buy)
    portfolio.buy(order_buy)

    dummy_empty = DummyEmptyDatahub()
    # 此时 get_bar 返回空，市值应为 10 * current_price，即 10*100 = 1000
    asset_value = portfolio.get_asset_value(pd.Timestamp('2023-01-01'), dummy_empty)
    assert asset_value == 1000


def test_total_value():
    """测试组合总价值的计算：现金 + 持仓市值"""
    portfolio = Portfolio(initial_cash=10000)
    # 买入 20 股，每股 100，花费 2000，剩余现金 8000
    order_data_buy = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01')],
        'asset': ['A'],
        'side': ['BUY'],
        'quantity': [20],
        'trade_price': [100]
    })
    order_buy = Order(order_data_buy)
    portfolio.buy(order_buy)

    dummy = DummyDatahub()
    # 使用 dummy，持仓市值 = 20*150 = 3000，总价值 = 8000 + 3000 = 11000
    total_val = portfolio.total_value(pd.Timestamp('2023-01-01'), dummy)
    assert total_val == 11000