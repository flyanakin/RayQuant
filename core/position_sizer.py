from abc import ABC, abstractmethod
import pandas as pd

class PositionSizer(ABC):
    """
    抽象基类：所有PositionSizer都必须实现 transform_signals_to_orders() 方法
    """

    @abstractmethod
    def transform_signals_to_orders(self,
                                    signals: pd.DataFrame,
                                    portfolio,
                                    data: pd.DataFrame,
                                    **kwargs) -> pd.DataFrame:
        """
        参数:
            signals: 必须包含 [asset, signal], index=日期 (或其他结构，子类可再细化)
            portfolio: 用于查询当前资金和持仓
            data: 行情DataFrame, index=日期, 至少包含 'close'
            **kwargs: 子类可能需要的其他参数(如风控、波动率、胜率等)

        返回:
            订单DataFrame, columns=[date, asset, side, quantity], 由Broker执行
        """
        pass


class FullCashPositionSizer(PositionSizer):
    """
    使用全部可用资金买入资产；卖出时清空所有持仓。
    """

    def transform_signals_to_orders(self,
                                    signals: pd.DataFrame,
                                    portfolio,
                                    data: pd.DataFrame,
                                    **kwargs) -> pd.DataFrame:
        orders_list = []
        for idx, row in signals.iterrows():
            asset_name = row["asset"]
            signal = row["signal"]
            if pd.isna(signal):
                continue

            if idx not in data.index:
                continue
            price = data.loc[idx, 'close']

            if signal.upper() == "BUY":
                shares_to_buy = int(portfolio.cash // price)
                if shares_to_buy > 0:
                    orders_list.append({
                        "date": idx,
                        "asset": asset_name,
                        "side": "BUY",
                        "quantity": shares_to_buy,
                    })

            elif signal.upper() == "SELL":
                position_info = portfolio.asset[portfolio.asset["asset"] == asset_name]
                if position_info.empty:
                    continue
                current_quantity = position_info.iloc[0]["quantity"]
                if current_quantity > 0:
                    orders_list.append({
                        "date": idx,
                        "asset": asset_name,
                        "side": "SELL",
                        "quantity": current_quantity,
                    })
            else:
                continue
        return pd.DataFrame(orders_list)