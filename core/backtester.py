import pandas as pd
from core.portfolio import Portfolio
from core.position_sizer import PositionSizer
from core.broker import Broker


class BackTester:
    def __init__(
        self,
        data: pd.DataFrame,
        position_sizer: PositionSizer,
        broker: Broker,
        portfolio: Portfolio,
    ):
        """
        :param data: 行情数据, index=日期, columns包含'close'等
        :param position_sizer: 封装下单数量规则
        :param broker: 执行撮合模块
        :param portfolio: 组合对象
        """
        self.data = data
        self.position_sizer = position_sizer
        self.broker = broker
        self.portfolio = portfolio

    def run_backtest(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        通过外部传入 signals, 然后:
          1) position_sizer 将 signals 转成 orders
          2) broker 执行订单 -> 更新 portfolio
          3) 记录每日组合净值
        :param signals: 策略已生成并处理好的信号 DataFrame(index=日期, 列包含['asset','signal']等)
        :return: 回测结果, 包含每日total_value
        """
        # 1) 将信号转成订单
        orders = self.position_sizer.transform_signals_to_orders(signals)

        # 2) Broker 执行订单
        self.broker.execute_orders(orders, self.data, self.portfolio)

        # 3) 统计每日组合净值
        daily_values = []
        for date in self.data.index:
            total_val = self.portfolio.total_value(date, self.data)
            daily_values.append({"date": date, "total_value": total_val})

        result_df = pd.DataFrame(daily_values).set_index("date")
        return result_df
