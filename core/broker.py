import pandas as pd
from core.portfolio import Portfolio


class Broker:
    """
    Broker: 执行订单的模块。
    简化假设: 对于每一个 date, asset, side, quantity, 我们直接用当日收盘价成交
    """

    def __init__(self):
        pass

    def execute_orders(
        self, orders: pd.DataFrame, data: pd.DataFrame, portfolio: Portfolio
    ):
        """
        :param orders: 包含 [date, asset, side, quantity] 列
        :param data: 行情数据(假设 index=日期, col=['asset','close'] )
                     或者是只针对单一资产, 这里留给你灵活处理
        :param portfolio: 组合对象
        """
        # 按日期排序, 逐条执行
        orders = orders.sort_values("date")
        for _, row in orders.iterrows():
            date = row["date"]
            asset = row["asset"]
            side = row["side"]
            qty = row["quantity"]

            # 从行情数据中获取当日收盘价 (简化假设)
            # 若你在 data 中有多资产，可以先过滤, 这里只演示单资产场景
            if isinstance(data, pd.DataFrame):
                # 假设 data 中 index=日期, 并且有 'close' 列
                close_price = data.loc[date, "close"]
            else:
                raise ValueError("data format not supported in this minimal example.")

            if side.upper() == "BUY":
                portfolio.buy(asset, date, qty, close_price)
            elif side.upper() == "SELL":
                portfolio.sell(asset, date, qty, close_price)
            else:
                raise ValueError(f"Unsupported side {side}")
