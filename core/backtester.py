# backtester.py
import pandas as pd
from core.portfolio import Portfolio
from core.position_manager import PositionManager
from core.broker import Broker, Order
from core.strategy import Strategy
from core.datahub import Datahub
from core.observer import Observer
import time


class BackTester:
    def __init__(
            self,
            data: Datahub,
            strategy: Strategy,           # <-- 依赖抽象基类，任何子类都可传进来
            position_manager: PositionManager,
            portfolio: Portfolio,
            broker: Broker = None,
            start_date: pd.Timestamp = None,
            end_date: pd.Timestamp = None,
            benchmarks: list[str] = None,
            symbols: list[str] = None,
    ):
        """
        :param strategy: 任意符合 Strategy 接口的对象
        :param position_manager: 将信号转成订单
        :param portfolio: 组合信息
        :param data: 读取数据
        :param broker: 负责撮合交易
        :param start_date: 回测开始日期，不填时读取所有时序数据
        :param end_date: 回测结束日期，不填时读取所有时序数据
        """
        self.strategy = strategy
        self.position_manager = position_manager
        self.broker = broker
        self.portfolio = portfolio
        self.data = data
        self.observer = Observer(self.portfolio)
        self.start_date = start_date
        self.end_date = end_date
        self.benchmarks = benchmarks
        self.symbols = symbols

    def run_backtest_without_broker(self) -> Observer:
        # 运行回测
        start_time = time.time()  # 开始计时
        for date_data in self.data.timeseries_iterator():
            dt, data = date_data
            signals = self.strategy.generate_signals(dt)
            orders = self.position_manager.transform_signals_to_orders(
                signals=signals,
                portfolio=self.portfolio,
                data=self.data,
                current_time=dt,
            )

            # 这部分逻辑，如果后续有可能放到broker中
            for index, row in orders.df.iterrows():
                if row['side'] == 'BUY':
                    buy_order_df = orders.df[(orders.df['side'] == 'BUY') & (orders.df['asset'] == row['asset'])]
                    buy_order = Order(buy_order_df)
                    self.portfolio.buy(buy_order)
                elif row['side'] == 'SELL':
                    sell_order_df = orders.df[(orders.df['side'] == 'SELL') & (orders.df['asset'] == row['asset'])]
                    sell_order = Order(sell_order_df)
                    self.portfolio.sell(sell_order)

            total_val = round(self.portfolio.total_value(current_date=dt, data=self.data))
            cash = round(self.portfolio.cash)
            self.observer.record(dt, total_val, cash)
            print(f"当前日期:{dt}")

        self.observer.calculate_metrics()
        end_time = time.time()

        print(f"回测总耗时: {end_time - start_time:.2f}秒")
        return self.observer

    def run_backtest(self) -> Observer:
        """
        核心流程:
          1) 用 strategy.generate_signals() 生成交易信号
          2) 用 position_sizer 把信号转为订单
          3) broker 执行订单，更新 portfolio
          4) 计算并记录每日组合净值
        :return: 回测结果 Observer
        """
        if self.start_date is None or self.end_date is None:
            self.data.load_all_data()
        else:
            self.data.load_all_data(start_date=self.start_date,
                                    end_date=self.end_date,
                                    benchmarks=self.benchmarks,
                                    symbols=self.symbols
                                    )

        if self.broker is None:
            return self.run_backtest_without_broker()
        else:
            raise NotImplementedError("Broker 尚未实现")
