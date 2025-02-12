import pandas as pd


class Order:
    """
    Order 数据封装类，用于包装订单 DataFrame，并对数据结构进行验证
    """
    REQUIRED_COLUMNS = {'date', 'asset', 'side', 'quantity', 'trade_price'}

    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self.df = df
        self.__print_orders()

    def _validate(self, df: pd.DataFrame):
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Order 数据缺少必需的列: {missing_cols}")

    def get(self):
        return self.df

    def __print_orders(self):
        if self.df.shape[0] > 0:
            print(f"产生订单：{self.df}")
        else:
            pass

    def __repr__(self):
        return f"<Order: {self.df.shape[0]} 条记录>"


class Broker:
    """
    Broker: 交易所
    """
    pass
