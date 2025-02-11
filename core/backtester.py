# backtester.py
import pandas as pd
from core.portfolio import Portfolio
from core.position_sizer import PositionSizer
from core.broker import Broker
# 关键：仅依赖抽象基类 Strategy，而非具体的 MovingAverageStrategy
from core.strategy import Strategy
from core.datahub import Datahub


class BackTester:
    def __init__(
            self,
            strategy: Strategy,           # <-- 依赖抽象基类，任何子类都可传进来
            position_sizer: PositionSizer,
            broker: Broker,
            portfolio: Portfolio,
            hub: Datahub
    ):
        """
        :param strategy: 任意符合 Strategy 接口的对象
        :param position_sizer: 将信号转成订单
        :param broker: 负责撮合交易
        :param portfolio: 组合信息
        :param hub: 行情(用于撮合和计算净值)
        """
        self.strategy = strategy
        self.position_sizer = position_sizer
        self.broker = broker
        self.portfolio = portfolio
        self.hub = hub

    def run_backtest(self) -> pd.DataFrame:
        """
        核心流程:
          1) 用 strategy.generate_signals() 生成交易信号
          2) 用 position_sizer 把信号转为订单
          3) broker 执行订单，更新 portfolio
          4) 计算并记录每日组合净值
        :return: 回测结果 DataFrame(index=日期, columns=['total_value'])
        """
        # 1) 策略信号
        signals = self.strategy.generate_signals()

        # 2) 转换为订单
        orders = self.position_sizer.transform_signals_to_orders(signals, self.portfolio, self.hub.get_bar())

        # 3) Broker 执行订单
        self.broker.execute_orders(orders, self.hub.get_bar(), self.portfolio)

        # 4) 统计每日净值
        daily_values = []
        for date in self.hub.get_bar().index:
            total_val = self.portfolio.total_value(date, self.hub.get_bar())
            daily_values.append({'date': date, 'total_value': total_val})

        result_df = pd.DataFrame(daily_values).set_index('date')
        return result_df
