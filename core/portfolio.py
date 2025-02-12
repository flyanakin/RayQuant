import pandas as pd
from core.broker import Order
from core.datahub import Datahub


class Portfolio:
    """
    管理资金和持仓信息。
    由三部分构成:
      1) self.cash (float): 剩余可用资金
      2) self.asset (pd.DataFrame): 当前持仓, 包含 [asset, quantity, cost_price]
        - asset: 标的名称或代码
        - quantity: 持仓数量(注意是股数而不是手数)
        - cost_price: 加权成本价
        - current_price: 当前市场价格
      3) self.trade_log (pd.DataFrame): 交易记录, 包含 [asset, trade_date, trade_qty, trade_price]
        - trade_qty: 买入为正, 卖出为负
    """

    def __init__(self,
                 initial_cash:
                 float = 1000000.0):
        """
        初始化组合:
          :param initial_cash: 初始资金
        """
        self.cash = initial_cash

        # 当前持仓信息。若要支持多标的，可直接多行
        self.asset = pd.DataFrame(columns=['asset', 'quantity', 'cost_price', 'current_price'])
        # 交易日志
        self.trade_log = pd.DataFrame(columns=['asset', 'trade_date', 'trade_qty', 'trade_price'])

    def buy(self,
            order: Order
            ):
        """
        买入操作: 更新仓位信息, 扣减现金，忽略交易成本，买入成功时current_price=cost_price 记录交易日志
        :param order: 订单
        """
        # 遍历订单中每一条记录
        for _, row in order.get().iterrows():
            asset = row['asset']
            quantity = row['quantity']
            trade_date = row['date']
            if 'trade_price' in row:
                trade_price = row['trade_price']
            else:
                raise ValueError("买入订单缺少交易价格字段 'trade_price'。")

            total_cost = quantity * trade_price
            if self.cash < total_cost:
                raise ValueError(f"现金不足，无法买入 {asset}，需要 {total_cost}，当前现金 {self.cash}。")

            # 扣减现金
            self.cash -= total_cost

            # 更新持仓记录
            mask = self.asset['asset'] == asset
            if self.asset[mask].empty:
                # 新增持仓记录
                new_record = pd.DataFrame({
                    'asset': [asset],
                    'quantity': [quantity],
                    'cost_price': [trade_price],  # 买入时成本价为成交价
                    'current_price': [trade_price]  # 初始当前价格设为买入价格
                }, index=[0])
                self.asset = pd.concat([self.asset, new_record], ignore_index=True)
            else:
                # 更新已有持仓
                idx_existing = self.asset[mask].index[0]
                old_quantity = self.asset.at[idx_existing, 'quantity']
                old_cost_price = self.asset.at[idx_existing, 'cost_price']
                new_quantity = old_quantity + quantity
                # 计算加权平均成本价
                new_cost_price = (old_quantity * old_cost_price + quantity * trade_price) / new_quantity
                self.asset.at[idx_existing, 'quantity'] = new_quantity
                self.asset.at[idx_existing, 'cost_price'] = new_cost_price
                # 买入成功时将当前价格更新为新加权成本价
                self.asset.at[idx_existing, 'current_price'] = new_cost_price

            # 记录交易日志（买入交易数量为正）
            new_record = pd.DataFrame({
                'asset': [asset],
                'trade_date': [trade_date],
                'trade_qty': [quantity],
                'trade_price': [trade_price]
            },index=[0])
            self.trade_log = pd.concat([self.trade_log, new_record], ignore_index=True)

    def sell(self, order: Order):
        """
        卖出操作: 更新仓位信息, 增加现金, 记录交易日志
        :param order: 订单
        """
        for _, row in order.get().iterrows():
            asset = row['asset']
            quantity = row['quantity']
            trade_date = row['date']
            if 'trade_price' in row:
                trade_price = row['trade_price']
            else:
                raise ValueError("卖出订单缺少交易价格字段 'trade_price'。")

            # 检查是否持有该标的及持仓数量是否足够
            mask = self.asset['asset'] == asset
            if self.asset[mask].empty:
                raise ValueError(f"持仓中不存在标的 {asset}，无法卖出。")
            idx_existing = self.asset[mask].index[0]
            current_quantity = self.asset.at[idx_existing, 'quantity']
            if quantity > current_quantity:
                raise ValueError(f"持仓数量不足，标的 {asset} 当前持仓 {current_quantity}，尝试卖出 {quantity}。")

            # 卖出获得的现金
            self.cash += quantity * trade_price

            # 更新持仓：卖出数量扣除，如果剩余为0则移除记录，否则更新数量
            new_quantity = current_quantity - quantity
            if new_quantity == 0:
                self.asset = self.asset.drop(idx_existing).reset_index(drop=True)
            else:
                self.asset.at[idx_existing, 'quantity'] = new_quantity
                # 卖出后，更新持仓中的当前价格为卖出价格（此处也可以选择不更新，根据实际需求调整）
                self.asset.at[idx_existing, 'current_price'] = trade_price

            # 记录交易日志，卖出时交易数量记为负
            new_record = pd.DataFrame({
                'asset': [asset],
                'trade_date': [trade_date],
                'trade_qty': [-quantity],
                'trade_price': [trade_price]
            }, index=[0])
            self.trade_log = pd.concat([self.trade_log, new_record], ignore_index=True)

    def get_asset_value(self,
                        current_date: pd.Timestamp,
                        data: Datahub
                        ) -> float:
        """
        计算当前持仓的市值 = ∑(quantity * 最新价格)
        :param current_date: 当前日期
        :param data: 一个行情表, index 为日期, columns 包含 'close'
        :return: 持仓市值
        """
        total_value = 0.0
        # TODO:去掉循环
        df_bar = data.get_bars(current_date=current_date)
        for _, row in self.asset.iterrows():
            asset = row['asset']
            quantity = row['quantity']
            # 获取指定标的在当前日期的行情数据
            if df_bar.empty:
                # 若无法获取行情数据，则退而求其次，使用持仓记录中的 current_price
                price = row['current_price']
            else:
                # 假定返回的 DataFrame 中 'close' 为最新价格
                price = df_bar.iloc[0]['close']
            total_value += quantity * price
        return total_value

    def total_value(self,
                    current_date: pd.Timestamp,
                    data: Datahub
                    ) -> float:
        """
        返回组合总价值 = 现金 + 持仓市值
        """
        asset_value = self.get_asset_value(current_date, data)
        return self.cash + asset_value
