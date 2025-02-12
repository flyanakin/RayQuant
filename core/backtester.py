# backtester.py
import pandas as pd
from core.portfolio import Portfolio
from core.position_manager import PositionManager
from core.broker import Broker
from core.strategy import Strategy
from core.datahub import Datahub
import time


class BackTester:
    def __init__(
            self,
            data: Datahub,
            strategy: Strategy,           # <-- 依赖抽象基类，任何子类都可传进来
            position_manager: PositionManager,
            portfolio: Portfolio,
            broker: Broker = None,
    ):
        """
        :param strategy: 任意符合 Strategy 接口的对象
        :param position_manager: 将信号转成订单
        :param portfolio: 组合信息
        :param data: 读取数据
        :param broker: 负责撮合交易
        """
        self.strategy = strategy
        self.position_manager = position_manager
        self.broker = broker
        self.portfolio = portfolio
        self.data = data

    def run_backtest_without_broker(self) -> pd.DataFrame:
        # 运行回测
        daily_values = []
        for date_data in self.data.timeseries_iterator():
            #print(f"date: {date_data}")
            #print(f"type: {type(date_data)}")
            start_time = time.time()  # 开始计时
            dt, data = date_data
            signals = self.strategy.generate_signals(dt)
            transform_time = time.time()  # 生成信号后的时间

            orders = self.position_manager.transform_signals_to_orders(
                signals=signals,
                portfolio=self.portfolio,
                data=self.data,
                current_time=dt,
            )
            orders_time = time.time()  # 转换信号为订单后的时间

            self.portfolio.buy(orders)
            self.portfolio.sell(orders)
            trading_time = time.time()  # 买卖操作后的时间

            total_val = self.portfolio.total_value(current_date=dt, data=self.data)
            daily_values.append({'date': dt, 'total_value': total_val})
            valuation_time = time.time()  # 计算总价值后的时间
            print(f"当前日期:{dt}, 组合总价值:{total_val}")

            # 输出每个步骤的执行时间
            #print(f"生成信号耗时: {transform_time - start_time:.6f}秒")
            #print(f"转换订单耗时: {orders_time - transform_time:.6f}秒")
            #print(f"执行交易耗时: {trading_time - orders_time:.6f}秒")
            #print(f"计算总价值耗时: {valuation_time - trading_time:.6f}秒")

        result_df = pd.DataFrame(daily_values).set_index('date')
        return result_df

    def run_backtest(self) -> pd.DataFrame:
        """
        核心流程:
          1) 用 strategy.generate_signals() 生成交易信号
          2) 用 position_sizer 把信号转为订单
          3) broker 执行订单，更新 portfolio
          4) 计算并记录每日组合净值
        :return: 回测结果 DataFrame(index=日期, columns=['total_value'])
        """
        if self.broker is None:
            return self.run_backtest_without_broker()
        else:
            raise NotImplementedError("Broker 尚未实现")
