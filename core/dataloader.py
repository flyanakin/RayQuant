import pandas as pd
from typing import Dict, Any, List


class DataLoader:
    """
    读取数据，支持时序数据读取和非时序数据读取，因为整个框架都是基于pandas，理论上不使用Dataloader也可以，这里核心是做格式的对齐
    - line数据：策略引擎的核心，回测的时候，引擎会根据时间进行循环计算，通常就是 时间-价格 的序列，如日线数据
    - non-line数据：策略引擎的辅助数据，不需要根据时间进行循环计算
    """
    def __init__(self):
        data_dict = {}

    def load_line_data(self,
                       data_path: str,
                       asset_col: str = 'asset_name',
                       date_col: str = 'trade_date',
                       ) -> pd.DataFrame:
        """
        读取时序数据
        """
        return pd.read_csv(data_path)