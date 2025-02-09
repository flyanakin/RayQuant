from typing import Dict, Any, List
import pandas as pd
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    策略基类
    Description: 策略是量化交易的核心，策略的业务语义是 满足一定条件时，产生交易信号。
                 如：沪深300成交量向下偏离720日均线20%时，产生买入信号。在本Quant中，策略是高度可编程可扩展的，框架只规范信号格式
    """
    @abstractmethod
    def generate_signals(self, **kwargs) -> pd.DataFrame:
        """
        必须由子类实现的方法，用于生成的交易信号。

        参数:
            data (pd.DataFrame):
                不做限制，由子类自行定义数据格式

        Returns:
            pd.DataFrame:
                需要包含时间、标的、买卖信号（BUY/SELL）、用户可以定义其他信息如附加的message
        """
        pass


class MovingAverageStrategy(Strategy):
    """
    均线策略
    Description: 向下偏离均线到一定程度时买入，向上偏离均线到一定程度时卖出
    """
    def __init__(
            self,
            data: pd.DataFrame,
    ) -> None:
        """
        :param data: 标的的时序数据，这里默认data中index为时间，包含'asset_name', 诸如close这样的指标列
        """
        self.data = data

    def generate_signals(
            self,
            indicator: str,
            asset_name: str,
            ma_buy: int,
            ma_sell: int,
            buy_bias: float,
            sell_bias: float,
    ) -> pd.DataFrame:
        """
        生成交易信号
        :param indicator: 指标名称，根据哪个指标的均线进行计算
        :param asset_name: 标的名称
        :param ma_buy: 买入参考的均线
        :param ma_sell: 卖出参考的均线
        :param buy_bias: 买入偏离均线的百分比
        :param sell_bias: 卖出偏离均线的百分比
        :return:
        """
        # 计算均线
        self.data['buy_ma'] = self.data[indicator].rolling(window=ma_buy).mean()
        self.data['sell_ma'] = self.data[indicator].rolling(window=ma_sell).mean()
        # 计算偏离均线的百分比
        self.data['buy_bias'] = (self.data[indicator] - self.data['buy_ma']) / self.data['buy_ma']
        self.data['sell_bias'] = (self.data[indicator] - self.data['sell_ma']) / self.data['sell_ma']
        # 生成交易信号
        self.data.loc[self.data['buy_bias'] < buy_bias, 'signal'] = 'BUY'
        self.data.loc[self.data['sell_bias'] > sell_bias, 'signal'] = 'SELL'
        return self.data[[indicator, asset_name, 'buy_ma', 'sell_ma', 'buy_bias', 'sell_bias', 'signal']]
