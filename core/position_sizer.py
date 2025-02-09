# position_sizer.py
import pandas as pd

class PositionSizer:
    def __init__(self):
        pass

    def transform_signals_to_orders(self, signals: pd.DataFrame, portfolio, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param signals: 必须包含 [asset, signal], index=日期
        :param portfolio: 用于查询当前持仓
        :return: 订单DataFrame, [date, asset, side, quantity]
        """
        orders_list = []
        for idx, row in signals.iterrows():
            asset_name = row["asset"]
            signal = row["signal"]
            if pd.isna(signal):
                continue  # 无信号跳过

            # 判断是否科创板(688) -> lot_size=200，否则=100
            lot_size = 200 if asset_name.startswith('688') else 100

            if signal.upper() == "BUY":
                # 2) 取当日价格
                if idx in data.index:
                    price = data.loc[idx, 'close']
                else:
                    continue

                cost_to_buy = lot_size * price
                print(f"Buying {asset_name} at {price}, cost: {cost_to_buy}")
                # 3) 判断资金够不够
                if cost_to_buy > portfolio.cash:
                    # 资金不足，就跳过或者只买部分
                    # （以下以"跳过"为例, 不下单）
                    continue
                side = "BUY"
                quantity = lot_size

            elif signal.upper() == "SELL":
                side = "SELL"
                # 先获取当前持仓
                position_info = portfolio.asset[portfolio.asset["asset"] == asset_name]
                if position_info.empty:
                    # 没有持仓则跳过，不生成订单
                    continue
                current_quantity = position_info.iloc[0]["quantity"]
                # 这里简单假设每次想卖1手，如果剩余持仓不足1手，就只卖剩余持仓
                sell_lot = min(current_quantity, lot_size)
                if sell_lot <= 0:
                    continue
                quantity = sell_lot
            else:
                continue

            orders_list.append({
                "date": idx,
                "asset": asset_name,
                "side": side,
                "quantity": quantity,
            })

        return pd.DataFrame(orders_list)