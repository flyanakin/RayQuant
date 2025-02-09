# strategy.py
from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    策略基类
    Description:
      - 任何继承本类的策略，都需要实现 generate_signals()，并返回一个包含特定列的 DataFrame.
    """

    REQUIRED_COLUMNS = {'asset', 'signal'}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        必须由子类实现的方法，用于生成交易信号。

        参数:
            data (pd.DataFrame): 策略所需的行情或因子数据(子类自行定义结构)
            **kwargs: 策略参数（如均线窗口、偏离阈值等）

        Returns:
            pd.DataFrame:
                - index: 一般为交易日期(可空)
                - columns: 至少包含 ['asset','signal'] 两列:
                    * 'asset': 标的名称或代码
                    * 'signal': 'BUY'/'SELL'/'HOLD'等指示
                - 其他列可选，如 'message', 'indicator_value'...
        """
        pass

    def validate_signals_format(self, signals: pd.DataFrame):
        """
        检查返回的 DataFrame 是否符合必要的列格式。
        """
        missing_cols = self.REQUIRED_COLUMNS - set(signals.columns)
        if missing_cols:
            raise ValueError(f"Strategy signals missing required columns: {missing_cols}")


class MovingAverageStrategy(Strategy):
    """
    均线策略示例：当向下偏离长周期均线到一定程度时买入，向上偏离短周期均线到一定程度时卖出。
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def generate_signals(self, **kwargs) -> pd.DataFrame:
        """
        使用父类的签名：def generate_signals(self, data: pd.DataFrame, **kwargs)
        通过 kwargs 获取策略所需的具体参数。
        """

        # 1) 从 kwargs 中获取所需参数，若不存在则设默认值
        indicator = kwargs.get('indicator', 'close')
        asset_col = kwargs.get('asset_col', 'asset')
        ma_buy = kwargs.get('ma_buy', 720)
        ma_sell = kwargs.get('ma_sell', 180)
        buy_bias = kwargs.get('buy_bias', -0.3)
        sell_bias = kwargs.get('sell_bias', 0.15)

        data = self.data

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

        # 5) 整理输出
        output = pd.DataFrame({
            'asset': data[asset_col],
            'signal': data['signal'],
            'indicator': data[indicator],  # 可选
            'buy_ma': data['buy_ma'],
            'sell_ma': data['sell_ma'],
            'buy_bias_val': data['buy_bias_val'],
            'sell_bias_val': data['sell_bias_val']
        }, index=data.index)

        # 6) 校验格式
        self.validate_signals_format(output)
        return output