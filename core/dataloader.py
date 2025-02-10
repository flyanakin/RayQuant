import pandas as pd
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import os


class DataLoader(ABC):
    """
    读取数据，支持时序数据读取和非时序数据读取
    包含：
      - 时序数据（timeseries_df）
      - 财报数据（fundamental_df）
      - 元数据（meta_df）
    可以根据需要重写底层“读取”逻辑来适配不同存储方式。
    """

    def __init__(
        self,
        time_col: str = "trade_date",  # 数据表中的日期列列名
        symbol_col: str = "asset",  # 数据表中的股票标识列列名
        timeseries_path: str = None,
        fundamental_path: str = None,
        meta_path: str = None,
    ):
        """
        :param time_col: 时序数据的时间列名
        :param symbol_col: 股票唯一标识列名
        :param timeseries_path: 时序数据文件默认路径，默认单文件
        :param fundamental_path: 财报数据文件默认路径，默认单文件
        :param meta_path: Meta 数据文件默认路径，默认单文件
        """

        self.time_col = time_col
        self.symbol_col = symbol_col

        self.timeseries_path = timeseries_path
        self.fundamental_path = fundamental_path
        self.meta_path = meta_path

        # 下面是内部 DataFrame 存储
        self.timeseries_df = None  # 用于存放时序数据
        self.fundamental_df = None  # 用于存放财报数据
        self.meta_df = None  # 用于存放元数据

    @abstractmethod
    def load_timeseries_data(self):
        """
        抽象方法：读取时序数据并存入 self.timeseries_df。
        子类必须实现。
        """
        pass

    @abstractmethod
    def load_fundamental_data(self):
        """
        抽象方法：读取财报数据并存入 self.fundamental_df。
        子类必须实现。
        """
        pass

    @abstractmethod
    def load_meta_data(self):
        """
        抽象方法：读取元数据并存入 self.meta_df。
        子类必须实现。
        """
        pass

    def load_all_data(self):
        """
        一键加载所有数据。
        这里不是抽象方法，因为基类可直接调用子类实现的抽象方法。
        """
        self.load_timeseries_data()
        self.load_fundamental_data()
        self.load_meta_data()

    def get_main_timeline(self):
        """
        非抽象方法，可直接在基类中实现。
        """
        if self.timeseries_df is None or self.timeseries_df.empty:
            return []
        # timeseries_df 索引 = (date, symbol)
        unique_dates = self.timeseries_df.index.get_level_values(0).unique()
        return unique_dates

    def get_data_by_date(self, current_date):
        """
        在特定日期，返回所有标的的时序数据快照(行索引= symbol)。
        """
        if self.timeseries_df is None:
            raise ValueError(
                "Timeseries data not loaded. Call load_timeseries_data() first."
            )

        try:
            df_slice = self.timeseries_df.xs(current_date, level=0)
            return df_slice
        except KeyError:
            return pd.DataFrame()

    def timeseries_iterator(self):
        """
        非抽象方法，用于回测引擎迭代。
        """
        timeline = self.get_main_timeline()
        for dt in timeline:
            yield dt, self.get_data_by_date(dt)

    def get_fundamental(self,
                        current_date,
                        start_date=None,
                        end_date=None
                        ) -> pd.DataFrame:
        """
        在特定日期区间，返回所有标的的财务数据快照(行索引= symbol)。
        """
        pass


class CSVDataLoader(DataLoader):
    """
    从按年拆分的低频CSV中加载数据，并提供按日期迭代的接口
    一个具体子类，必须实现3个抽象方法：
      - load_timeseries_data()
      - load_fundamental_data()
      - load_meta_data()
    否则会报错。
    """

    def load_timeseries_data(self):
        if not os.path.exists(self.timeseries_path):
            print(f"[WARN] Timeseries file not found: {self.timeseries_path}")
            self.timeseries_df = pd.DataFrame()
            return

        df = pd.read_csv(self.timeseries_path)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df.set_index([self.time_col, self.symbol_col], inplace=True)
        df.sort_index(inplace=True)
        self.timeseries_df = df

    def load_fundamental_data(self):
        if not os.path.exists(self.fundamental_path):
            print(f"[WARN] Fundamental file not found: {self.fundamental_path}")
            self.fundamental_df = pd.DataFrame()
            return
        df = pd.read_csv(self.fundamental_path)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df.set_index([self.symbol_col, self.time_col], inplace=True)
        df.sort_index(inplace=True)
        self.fundamental_df = df

    def load_meta_data(self):
        if not os.path.exists(self.meta_path):
            print(f"[WARN] Meta file not found: {self.meta_path}")
            self.meta_df = pd.DataFrame()
            return
        df = pd.read_csv(self.meta_path)
        df.set_index(self.symbol_col, inplace=True)
        df.sort_index(inplace=True)
        self.meta_df = df
