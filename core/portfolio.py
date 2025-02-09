from typing import List
import pandas as pd


class Portfolio:
    """
    管理资金和持仓信息。
    由三部分构成:
      1) self.cash (float): 剩余可用资金
      2) self.asset (pd.DataFrame): 当前持仓, 包含 [asset, quantity, cost_price]
      3) self.trade_log (pd.DataFrame): 交易记录, 包含 [asset, trade_date, trade_qty, trade_price]
    """

    def __init__(self, initial_cash: float = 10000000.0):
        """
        初始化组合:
          :param initial_cash: 初始资金
        """
        self.cash = initial_cash

        # 当前持仓信息。若要支持多标的，可直接多行
        self.asset = pd.DataFrame(columns=['asset', 'quantity', 'cost_price'])
        # 交易日志
        self.trade_log = pd.DataFrame(columns=['asset', 'trade_date', 'trade_qty', 'trade_price'])

    def buy(self, asset: str, trade_date, trade_qty: int, trade_price: float):
        """
        买入操作: 更新仓位信息, 扣减现金, 记录交易日志
        :param asset: 标的
        :param trade_date: 成交日期
        :param trade_qty: 买入数量
        :param trade_price: 买入价格
        """
        cost = trade_qty * trade_price
        if cost > self.cash:
            # 资金不足就不允许买入(或自行处理买入量)
            raise ValueError("Not enough cash to complete BUY order.")

        # 扣减现金
        self.cash -= cost

        # 更新持仓
        # 如果此标的已存在, 更新数量和加权成本
        mask = (self.asset['asset'] == asset)
        if not mask.any():
            # 新增行
            new_row = {
                'asset': asset,
                'quantity': trade_qty,
                'cost_price': trade_price
            }
            self.asset = pd.concat([self.asset, pd.DataFrame([new_row])], ignore_index=True)
        else:
            current_quantity = self.asset.loc[mask, 'quantity'].values[0]
            current_cost_price = self.asset.loc[mask, 'cost_price'].values[0]

            new_quantity = current_quantity + trade_qty
            # 加权成本价 = (旧持仓 * 旧成本 + 新买入数量 * 买入价) / (总数量)
            new_cost_price = (
                current_quantity * current_cost_price + trade_qty * trade_price
            ) / new_quantity

            self.asset.loc[mask, 'quantity'] = new_quantity
            self.asset.loc[mask, 'cost_price'] = new_cost_price

        # 记录交易日志
        new_trade = {
            'asset': asset,
            'trade_date': trade_date,
            'trade_qty': trade_qty,
            'trade_price': trade_price
        }
        self.trade_log = pd.concat([self.trade_log, pd.DataFrame([new_trade])], ignore_index=True)

    def sell(self, asset: str, trade_date, trade_qty: int, trade_price: float):
        """
        卖出操作: 更新仓位信息, 增加现金, 记录交易日志
        :param asset: 标的
        :param trade_date: 成交日期
        :param trade_qty: 卖出数量
        :param trade_price: 卖出价格
        """
        mask = (self.asset['asset'] == asset)
        if not mask.any():
            raise ValueError(f"No position to SELL for asset: {asset}")

        current_quantity = self.asset.loc[mask, 'quantity'].values[0]
        if trade_qty > current_quantity:
            raise ValueError("Not enough shares to sell.")

        # 卖出获得资金
        revenue = trade_qty * trade_price
        self.cash += revenue

        # 更新持仓
        new_quantity = current_quantity - trade_qty
        if new_quantity == 0:
            # 清仓后删掉这一行
            self.asset = self.asset.loc[~mask]
        else:
            self.asset.loc[mask, 'quantity'] = new_quantity
            # cost_price 原则上不变 (剩余持仓维持原成本)

        # 记录交易日志
        new_trade = {
            'asset': asset,
            'trade_date': trade_date,
            'trade_qty': -trade_qty,  # 卖出记为负数
            'trade_price': trade_price
        }
        self.trade_log = pd.concat([self.trade_log, pd.DataFrame([new_trade])], ignore_index=True)

    def get_asset_value(self, date, price_lookup: pd.DataFrame) -> float:
        """
        计算当前持仓的市值 = ∑(quantity * 最新价格)
        :param date: 当前日期
        :param price_lookup: 一个行情表, index 为日期, columns 包含 'close'
        :return: 持仓市值
        """
        total_val = 0.0
        for _, row in self.asset.iterrows():
            asset_symbol = row['asset']
            quantity = row['quantity']
            # 简化假设: 同一个 asset_symbol 的 price_lookup
            # 这里演示只针对一个标的, 若多标的, 需更复杂处理
            if date in price_lookup.index:
                latest_price = price_lookup.loc[date, 'close']
            else:
                # 若当前日没有价格(休市等), 可能用前一日价格, 这里简单返回0
                latest_price = 0
            total_val += quantity * latest_price
        return total_val

    def total_value(self, date, price_lookup: pd.DataFrame) -> float:
        """
        返回组合总价值 = 现金 + 持仓市值
        """
        return self.cash + self.get_asset_value(date, price_lookup)