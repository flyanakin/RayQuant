from abc import ABC, abstractmethod
import warnings
import pandas as pd
from core.datahub import Datahub


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
                         **kwargs) -> pd.DataFrame:
        """
        必须由子类实现的方法，用于生成交易信号。

        参数:
            current_time: 当前回测的时间点（pd.Timestamp），用于防止引入未来数据。
            **kwargs: 策略参数（如均线窗口、偏离阈值等）

        Returns:
            pd.DataFrame:
                - index: 多层索引，其中必须包含 'trade_date' 和 'symbol'
                - columns: 至少包含 ['asset','signal'] 两列:
                    * 'asset': 标的名称或代码
                    * 'signal': 'BUY'/'SELL'/'HOLD'等指示
                - 其他列可选，如 'message', 'indicator_value'...
        """
        pass

    def validate_signals_format(self,
                                signals: pd.DataFrame,
                                current_time: pd.Timestamp,
                                ):
        """
        检查返回的 DataFrame 是否符合必要的格式，并且确保信号不包含未来的数据。

        参数:
            signals: 策略生成的信号 DataFrame
            current_time: 当前回测时间（pd.Timestamp），所有信号的 trade_date 均应 <= current_time
        """
        missing_cols = self.REQUIRED_COLUMNS - set(signals.columns)
        missing_index = self.REQUIRED_INDEX - set(signals.index.names)
        if missing_cols:
            raise ValueError(f"Strategy signals missing required columns: {missing_cols}")
        if missing_index:
            raise ValueError(f"Strategy signals missing required index: {missing_index}")

        # 检查是否包含未来数据: 取出索引中 trade_date 这一层进行比较
        trade_dates = signals.index.get_level_values('trade_date')
        if (trade_dates > current_time).any():
            warnings.warn("策略信号中包含未来数据，请检查策略逻辑是否存在前瞻性偏差。", UserWarning)


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

    def generate_signals(
            self,
            current_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        使用父类的签名：def generate_signals(self, data: pd.DataFrame, **kwargs)
        通过 kwargs 获取策略所需的具体参数。
        """

        # 1) 从 kwargs 中获取所需参数，若不存在则设默认值
        indicator = self.indicator
        ma_buy = self.ma_buy
        ma_sell = self.ma_sell
        buy_bias = self.buy_bias
        sell_bias = self.sell_bias

        data = self.hub.get_bar(end_date=current_time)

        # 2) 计算长短均线
        data['buy_ma'] = data[indicator].rolling(window=ma_buy).mean()
        data['sell_ma'] = data[indicator].rolling(window=ma_sell).mean()

        # 3) 计算偏离
        data['buy_bias_val'] = (data[indicator] - data['buy_ma']) / data['buy_ma']
        data['sell_bias_val'] = (data[indicator] - data['sell_ma']) / data['sell_ma']

        # 4) 生成 signal
        data['signal'] = None
        data.loc[data['buy_bias_val'] < buy_bias, 'signal'] = 'BUY'
        data.loc[data['sell_bias_val'] > sell_bias, 'signal'] = 'SELL'
        print(data.columns)

        # 5) 整理输出
        output = pd.DataFrame({
            'signal': data['signal'],
            'indicator': data[indicator],  # 可选
            'buy_ma': data['buy_ma'],
            'sell_ma': data['sell_ma'],
            'buy_bias_val': data['buy_bias_val'],
            'sell_bias_val': data['sell_bias_val']
        }, index=data.index)

        # 6) 校验格式
        self.validate_signals_format(output, current_time)
        return output
