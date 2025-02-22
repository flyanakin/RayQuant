import pandas as pd
from core.portfolio import Portfolio
from core.position_manager import PositionManager
from core.broker import Broker, Order
from core.strategy import Strategy
from core.datahub import Datahub
from core.observer import Observer
from core.timeline import Timeline
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
        self.start_date = start_date
        self.end_date = end_date
        self.benchmarks = benchmarks
        self.symbols = symbols
        self.observer = Observer(self.portfolio)

    def run_backtest_without_broker(self) -> Observer:
        # 运行回测
        start_time = time.time()  # 开始计时
        timeline = Timeline(bar_df=self.data.bar_df)
        for dt in timeline.timeseries_iterator():
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

            # 记录 benchmark 数据：利用 Datahub.get_bars 接口获取当日 benchmark 数据（仅提取收盘价）
            benchmark_bars = self.data.get_bars(current_date=dt, benchmark=True)
            if not benchmark_bars.empty:
                # 如果返回的是 MultiIndex，则先重置索引
                if isinstance(benchmark_bars.index, pd.MultiIndex):
                    benchmark_bars = benchmark_bars.reset_index()
                # 从数据中提取每个 benchmark 的收盘价（假设列名为 "close"）
                if "close" in benchmark_bars.columns:
                    # 假设数据中每个 benchmark 对应一行，可利用 groupby 防止重复
                    benchmark_values = benchmark_bars.groupby("symbol")["close"].first().to_dict()
                else:
                    benchmark_values = {}
                self.observer.record_benchmark(dt, benchmark_values)

            print(f"当前日期:{dt}")

        self.observer.calculate_metrics()
        self.observer.calculate_benchmark_metrics()
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
