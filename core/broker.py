import pandas as pd
from core.portfolio import Portfolio


class Broker:
    """
    Broker: 执行订单的模块。
    假设 order 中的 'quantity' 是 “手数”，
    Broker 会乘以 lot_size 把它转成真实股数后再调用 Portfolio。
    """
    def __init__(self, lot_size=100):
        """
        :param lot_size: 每手多少股(默认为100)
        """
        self.lot_size = lot_size

    def execute_orders(self, orders: pd.DataFrame, data: pd.DataFrame, portfolio: Portfolio):
        """
        :param orders: 必须包含 [date, asset, side, quantity] 列
                       其中 'quantity' 表示“手数”。
        :param data: 行情表(至少含 'close')
        :param portfolio: 用于更新资金和持仓。其内 quantity 一律用“股数”。
        """
        orders = orders.sort_values('date')
        for _, row in orders.iterrows():
            date = row['date']
            asset = row['asset']
            side = row['side'].upper()
            lot_qty = row['quantity']  # 这是“手数”

            if date not in data.index:
                continue
            close_price = data.loc[date, 'close']

            # 把手数转换成股数
            share_qty = lot_qty * self.lot_size
            cost = share_qty * close_price

            if side == 'BUY':
                # 如果资金不足，就做“部分买”或跳过
                if cost > portfolio.cash:
                    max_shares = int(portfolio.cash // close_price)
                    if max_shares <= 0:
                        # 一股都买不起
                        continue
                    else:
                        # 按照股数去结算可能买多少手
                        lot_can_buy = max_shares // self.lot_size
                        if lot_can_buy <= 0:
                            continue
                        share_qty = lot_can_buy * self.lot_size

                # 到这里 share_qty 就是要买的真实股数
                portfolio.buy(asset, date, share_qty, close_price)

            elif side == 'SELL':
                # 若持仓不足 => 部分卖
                if share_qty > 0:
                    portfolio.sell(asset, date, share_qty, close_price)