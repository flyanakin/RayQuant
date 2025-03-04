from abc import ABC, abstractmethod
import warnings
import pandas as pd
import numpy as np
from core.datahub import Datahub
from utils.technical_process import moving_average
import time


class Signal:
    """
    Signal 数据封装类，用于包装信号 DataFrame，并对数据结构进行验证
    """
    # 定义必须包含的列和索引层级（可以根据需要调整）
    REQUIRED_COLUMNS = {'close', 'signal'}
    REQUIRED_INDEX = {'trade_date', 'symbol'}

    def __init__(self, df: pd.DataFrame, current_time: pd.Timestamp):
        self._validate(df, current_time)
        self.df = df

    def _validate(self, df: pd.DataFrame, current_time: pd.Timestamp):
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        missing_index = self.REQUIRED_INDEX - set(df.index.names)
        if missing_cols:
            raise ValueError(f"Signal 数据缺少必需的列: {missing_cols}")
        if missing_index:
            raise ValueError(f"Signal 数据缺少必需的索引: {missing_index}")
        # 检查索引中的日期是否存在未来数据
        trade_dates = df.index.get_level_values('trade_date')
        if (trade_dates > current_time).any():
            warnings.warn("Signal 数据包含未来数据", UserWarning)

    def get(self):
        return self.df

    def __repr__(self):
        return f"<Signal: {self.df.shape[0]} 条记录>"


class Strategy(ABC):
    """
    策略基类
    Description:
      - 任何继承本类的策略，都需要实现 generate_signals()，并返回一个包含特定列的 DataFrame.
    """

    REQUIRED_COLUMNS = {'signal'}
    REQUIRED_INDEX = {'trade_date', 'symbol'}

    @abstractmethod
    def generate_signals(self,
                         current_time: pd.Timestamp,
                         **kwargs) -> Signal:
        """
        必须由子类实现的方法，用于生成交易信号。

        参数:
            current_time: 当前回测的时间点（pd.Timestamp），用于防止引入未来数据。
            **kwargs: 策略参数（如均线窗口、偏离阈值等）

        Returns:
            Signal: 信号对象，包含信号 DataFrame
                Signal.get():
                - index: 多层索引，其中必须包含 'trade_date' 和 'symbol'
                - columns: 至少包含 ['asset','signal'] 两列:
                    * 'asset': 标的名称或代码
                    * 'signal': 'BUY'/'SELL'/'HOLD'等指示
                - 其他列可选，如 'message', 'indicator_value'...
        """
        pass


class MovingAverageStrategy(Strategy):
    """
    均线策略示例：当向下偏离长周期均线到一定程度时买入，向上偏离短周期均线到一定程度时卖出。
    """
    def __init__(self,
                 hub: Datahub,
                 indicator: str = 'close',
                 ma_buy: int = 720,
                 ma_sell: int = 180,
                 buy_bias: float = -0.3,
                 sell_bias: float = 0.15
                 ):
        self.hub = hub
        self.indicator = indicator
        self.ma_buy = ma_buy
        self.ma_sell = ma_sell
        self.buy_bias = buy_bias
        self.sell_bias = sell_bias

    def generate_signals(self, current_time: pd.Timestamp, **kwargs) -> Signal:
        """
        使用父类的签名：def generate_signals(self, data: pd.DataFrame, **kwargs)
        通过 kwargs 获取策略所需的具体参数。
        :param current_time:
        """

        # 1) 从 kwargs 中获取所需参数，若不存在则设默认值
        indicator = self.indicator
        ma_buy = self.ma_buy
        ma_sell = self.ma_sell
        buy_bias = self.buy_bias
        sell_bias = self.sell_bias

        window = max(ma_buy, ma_sell)
        start_date = current_time - pd.Timedelta(days=window - 1)

        data = self.hub.get_bars(start_date=start_date, end_date=current_time).copy()

        # 2) 计算长短均线
        # 将指标列转换为 NumPy 数组（建议提前保证数据类型合适，比如 float32 或 float64）
        indicator_values = data[indicator].values.astype(np.float64)
        data['buy_ma'] = moving_average(indicator_values, ma_buy)
        data['sell_ma'] = moving_average(indicator_values, ma_sell)

        # 3) 计算偏离
        data['buy_bias_val'] = (data[indicator] - data['buy_ma']) / data['buy_ma']
        data['sell_bias_val'] = (data[indicator] - data['sell_ma']) / data['sell_ma']

        # 4) 生成 signal
        data['signal'] = np.where(
            data['sell_bias_val'] > sell_bias, 'SELL',
            np.where(data['buy_bias_val'] < buy_bias, 'BUY', None)
        )

        # 5) 整理输出
        full_output = data[['signal', indicator, 'close', 'buy_ma', 'sell_ma', 'buy_bias_val', 'sell_bias_val']].copy()
        # 只保留current_time的信号
        output = full_output.xs(
                    key=current_time,
                    level='trade_date',
                    drop_level=False
                )

        return Signal(output, current_time)
