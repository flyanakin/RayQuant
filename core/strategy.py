from typing import Dict, Any, List
import pandas as pd
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    策略基类
    Description: 策略是量化交易的核心，策略的业务语义是 满足一定条件时，产生交易信号。
                 如：沪深300成交量向下偏离720日均线20%时，产生买入信号。在本Quant中，策略是高度可编程可扩展的，框架只规范信号格式
    """

    def __init__(self, params: Dict[str] = None) -> None:
        self.params = params

    @abstractmethod
    def generate_signals(self, data=None) -> pd.DataFrame:
        """
        必须由子类实现的方法，用于生成的交易信号。

        参数:
            data (pd.DataFrame):
                不做限制，由子类自行定义数据格式

        Returns:
            pd.DataFrame:
                需要包含时间、标的、买卖信号（'BUY'/'SELL'）、用户可以定义其他信息如附加的message
        """
        pass
