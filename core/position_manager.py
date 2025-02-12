from abc import ABC, abstractmethod
import pandas as pd
from core.strategy import Signal
from core.portfolio import Portfolio
from core.broker import Order
from core.datahub import Datahub
import time


def get_min_lot(asset: str) -> int:
    """
    根据资产代码判断最小交易手数。

    假设：
      - 如果 asset 为688开头，则认为是 A 股或科创板，最小交易单位为200股；
      - 否则最小交易单位为100股。
    """
    if asset.isdigit() and asset.startswith('688'):
        return 200
    else:
        return 100


class PositionManager(ABC):
    """
    抽象基类：所有PositionSizer都必须实现 transform_signals_to_orders() 方法
    """

    @abstractmethod
    def transform_signals_to_orders(self,
                                    signals: Signal,
                                    portfolio: Portfolio,
                                    data: Datahub,
                                    current_time: pd.Timestamp,
                                    **kwargs) -> Order:
        """
        参数:
            signals: 必须包含 [asset, signal], index=日期 (或其他结构，子类可再细化)
            portfolio: 用于查询当前资金和持仓
            data: 获取数据
            current_time: 当前回测的时间点（pd.Timestamp），用于防止引入未来数据。
            **kwargs: 子类可能需要的其他参数(如风控、波动率、胜率等)

        返回:
            订单DataFrame, columns=[date, asset, side, quantity]
        """
        pass


class EqualWeightPositionManager(PositionManager):
    """
    简单仓位管理器实现：
      - 对于买入信号，使用全仓现金平均分配给每个买入标的，
        按照信号中的收盘价计算可以买入的股数（必须满足最小手数要求）。
      - 对于卖出信号，卖出当前持仓中该标的的所有份额。
      - 交易规则：A股、科创板最少1手200股，其他最少1手100股。
    """

    def transform_signals_to_orders(self,
                                    signals: Signal,
                                    portfolio: Portfolio,
                                    data: Datahub,
                                    current_time: pd.Timestamp,
                                    **kwargs) -> Order:
        """
        根据交易信号转换为订单：
          - 卖出信号：对于标记为 'SELL' 的信号，检查当前持仓，若持有则卖出所有份额；
          - 买入信号：对于标记为 'BUY' 的信号，使用 portfolio.cash 平均分配给每个买入标的，
            根据该标的的 close 价格计算可以买入的股数，同时要求订单数量必须是最小手数的整数倍，
            否则不生成该订单。
        """
        orders_list = []
        df_signals = signals.get()
        current_prices = data.get_bars(current_date=current_time)

        # 确保 'signal' 列为大写字符串，方便比较
        df_signals['signal'] = df_signals['signal'].astype(str).str.upper()

        # --- 处理卖出信号 ---
        sell_signals = df_signals[df_signals['signal'] == 'SELL']
        if not sell_signals.empty:
            sell_symbols = sell_signals.index.get_level_values('symbol').unique()
            # 只查询当前持仓中需要卖出的标的
            relevant_holdings = portfolio.asset[portfolio.asset['asset'].isin(sell_symbols)]
            for symbol in sell_symbols:
                holding = relevant_holdings[relevant_holdings['asset'] == symbol]
                if not holding.empty:
                    held_qty = holding['quantity'].values[0]
                    if held_qty > 0:
                        trade_price = current_prices.loc[(current_time, symbol), 'close']
                        orders_list.append({
                            "date": current_time,
                            "asset": symbol,
                            "side": "SELL",
                            "quantity": held_qty,
                            "trade_price": trade_price
                        })

        # --- 处理买入信号 ---
        # TODO: buy order信号处理可以考虑做进步抽象
        buy_signals = df_signals[df_signals['signal'] == 'BUY']
        # 开始处理买入信号的时间
        if not buy_signals.empty:
            # 一次性获取所有买入信号的价格和最小手数
            buy_symbols = buy_signals.index.get_level_values('symbol').unique()
            min_lots = {symbol: get_min_lot(symbol) for symbol in buy_symbols}
            allocated_cash = portfolio.cash / len(buy_signals)
            for symbol in buy_symbols:
                close_price = current_prices.loc[(current_time, symbol), 'close']
                min_lot = min_lots[symbol]
                raw_qty = allocated_cash / close_price
                order_qty = int(raw_qty // min_lot) * min_lot
                if order_qty >= min_lot:
                    orders_list.append({
                        "date": current_time,
                        "asset": symbol,
                        "side": "BUY",
                        "quantity": order_qty,
                        "trade_price": close_price
                    })

        # 构造订单 DataFrame
        orders_df = pd.DataFrame(orders_list, columns=['date', 'asset', 'side', 'quantity', 'trade_price'])

        return Order(orders_df)

