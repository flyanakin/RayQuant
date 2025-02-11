from abc import ABC, abstractmethod
import pandas as pd
from core.strategy import Signal
from core.portfolio import Portfolio
from core.broker import Order
from core.datahub import Datahub


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
        current_prices = data.get_bar(current_date=current_time)

        # 确保 'signal' 列为大写字符串，方便比较
        df_signals['signal'] = df_signals['signal'].astype(str).str.upper()

        # --- 处理卖出信号 ---
        sell_signals = df_signals[df_signals['signal'] == 'SELL']
        for idx, row in sell_signals.iterrows():
            # 从多层索引中提取标的代码，这里假设索引包含 'trade_date' 和 'symbol'
            if isinstance(idx, tuple):
                index_names = df_signals.index.names
                symbol = idx[index_names.index('symbol')]
            else:
                symbol = idx

            # 检查 portfolio.asset 中是否持有该标的
            holding = portfolio.asset[portfolio.asset['asset'] == symbol]
            if not holding.empty:
                held_qty = holding.iloc[0]['quantity']
                if held_qty > 0:
                    trade_price = current_prices.loc[(current_time, symbol), 'close'] # 这里默认了用收盘价立刻买入
                    orders_list.append({
                        "date": current_time,
                        "asset": symbol,
                        "side": "SELL",
                        "quantity": held_qty,
                        "trade_price": trade_price
                    })

        # --- 处理买入信号 ---
        buy_signals = df_signals[df_signals['signal'] == 'BUY']
        num_buy = len(buy_signals)
        if num_buy > 0 and portfolio.cash > 0:
            # 平均分配给每个买入标的的现金
            allocated_cash = portfolio.cash / num_buy
            for idx, row in buy_signals.iterrows():
                if isinstance(idx, tuple):
                    index_names = df_signals.index.names
                    symbol = idx[index_names.index('symbol')]
                else:
                    symbol = idx

                close_price = current_prices.loc[(current_time, symbol), 'close']
                min_lot = get_min_lot(symbol)
                # 计算使用 allocated_cash 能买入的最大股数
                raw_qty = allocated_cash / close_price
                # 向下取整到最近的整数手（即 min_lot 的整数倍）
                order_qty = int(raw_qty // min_lot) * min_lot
                # 只有满足最小交易手数要求才生成订单
                if order_qty >= min_lot:
                    orders_list.append({
                        "date": current_time,
                        "asset": symbol,
                        "side": "BUY",
                        "quantity": order_qty,
                        "trade_price": close_price
                    })

        # 构造订单 DataFrame，必须包含 ['date', 'asset', 'side', 'quantity'] 列
        orders_df = pd.DataFrame(orders_list, columns=['date', 'asset', 'side', 'quantity', 'trade_price'])
        return Order(orders_df)

