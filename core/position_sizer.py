# position_sizer.py
import pandas as pd
import math


class PositionSizer:
    """
    按“百分比头寸”来生成订单的版本示例：
      - BUY: 用剩余现金(100%) 来买入资产
      - SELL: 卖出全部持仓
      - 无现金或无持仓时跳过
    """

    def __init__(self):
        pass

    def transform_signals_to_orders(self,
                                    signals: pd.DataFrame,
                                    portfolio,
                                    data: pd.DataFrame) -> pd.DataFrame:
        """
        :param signals: 必须包含 [asset, signal], index=日期
        :param portfolio: 用于查询当前资金和持仓
        :param data: 行情DataFrame, index=日期, 至少包含 'close'
        :return: 订单DataFrame, [date, asset, side, quantity]
        """
        orders_list = []

        for idx, row in signals.iterrows():
            asset_name = row["asset"]
            signal = row["signal"]

            # 跳过无信号
            if pd.isna(signal):
                continue

                # 获取当日收盘价(若没有就跳过)
            if idx not in data.index:
                continue
            price = data.loc[idx, 'close']

            if signal.upper() == "BUY":
                # 1) 计算能买多少股？（假设用全部剩余现金）
                # 注意是 int() 向下取整，以确保不会超出资金
                shares_to_buy = int(portfolio.cash // price)

                if shares_to_buy <= 0:
                    # 资金不足，买不了任何一股 => 跳过
                    continue

                orders_list.append({
                    "date": idx,
                    "asset": asset_name,
                    "side": "BUY",
                    "quantity": shares_to_buy,
                })

            elif signal.upper() == "SELL":
                # 2) 先获取当前持仓
                position_info = portfolio.asset[portfolio.asset["asset"] == asset_name]
                if position_info.empty:
                    # 没有持仓则跳过
                    continue
                current_quantity = position_info.iloc[0]["quantity"]
                if current_quantity <= 0:
                    continue

                # 默认卖出全部持仓
                orders_list.append({
                    "date": idx,
                    "asset": asset_name,
                    "side": "SELL",
                    "quantity": current_quantity,
                })

            else:
                # 其他信号(HOLD等) => 跳过
                continue

        return pd.DataFrame(orders_list)