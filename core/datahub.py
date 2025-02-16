import pandas as pd
from typing import Dict, Any
from abc import ABC, abstractmethod
import os


class Datahub(ABC):
    """
    读取数据，支持时序数据读取和非时序数据读取
    包含：
      - 时序数据（bar_df），如日线/周线，最细颗粒度到日级别
      - 财报数据（fundamental_df）
      - 元数据（info_df），如股票名称/行业之类
    可以根据需要重写底层“读取”逻辑来适配不同存储方式。
    """

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, Any]],
    ):
        """
        :param data_dict: 数据字典，包含数据类型和路径、字段映射，如
               {
                    "bar":{
                        "path": "data/bar.csv",
                        "col_mapping": {
                            "trade_date": "date",
                            "symbol": "ts_code",
                        },
                        {
                        "path": "data/benchmark.csv",
                        "col_mapping": {
                            "trade_date": "date",
                            "symbol": "ts_code",
                        "tags": ["benchmark"]
                        }
                    },
               }
        """

        self.data_dict = data_dict

        # 下面是内部 DataFrame 存储
        self.bar_df = None  # 用于存放时序数据
        self.fundamental_df = None  # 用于存放财报数据
        self.info_df = None  # 用于存放元数据
        self.load_all_data()

    @abstractmethod
    def load_bar_data(self):
        """
        抽象方法：读取时序数据并存入 self.bar_df。
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
    def load_info_data(self):
        """
        抽象方法：读取元数据并存入 self.info_df。
        子类必须实现。
        """
        pass

    def load_all_data(self):
        """
        一键加载所有数据。
        这里不是抽象方法，因为基类可直接调用子类实现的抽象方法。
        """
        if 'bar' not in self.data_dict:
            raise ValueError("bar data not found in data_dict.")
        if 'info' in self.data_dict:
            self.load_info_data()
        if 'fundamental' in self.data_dict:
            self.load_fundamental_data()
        self.load_bar_data()

    def get_main_timeline(self):
        """
        非抽象方法，可直接在基类中实现。
        """
        if self.bar_df is None or self.bar_df.empty:
            return []
        # bar_df 索引 = (trade_date, symbol)
        unique_dates = self.bar_df.index.get_level_values(0).unique()
        return unique_dates

    def get_data_by_date(self, current_date):
        """
        在特定日期，返回所有标的的时序数据快照(行索引= symbol)。
        """
        if self.bar_df is None:
            raise ValueError(
                "Timeseries data not loaded. Call load_bar_data() first."
            )

        try:
            df_slice = self.bar_df.xs(current_date, level=0)
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

    def get_bars(self,
                 current_date: pd.Timestamp = None,
                 start_date: pd.Timestamp = None,
                 end_date: pd.Timestamp = None,
                 symbol: str = None,
                 query: str = None) -> pd.DataFrame:
        """
        在特定日期区间，返回所有标的的行情数据快照(行索引= symbol)。
        :param current_date: 当前或特定的单一日期
        :param start_date: 起始日期
        :param end_date: 结束日期
        :param symbol: 标的的唯一标识
        :param query: 其他查询条件，预留参数
        :return: 符合条件的行情数据
        """
        # 验证日期有效性
        if current_date and not isinstance(current_date, pd.Timestamp):
            raise ValueError("current_date must be a pd.Timestamp object.")
        if start_date and not isinstance(start_date, pd.Timestamp):
            raise ValueError("start_date must be a pd.Timestamp object.")
        if end_date and not isinstance(end_date, pd.Timestamp):
            raise ValueError("end_date must be a pd.Timestamp object.")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date should not be later than end_date.")

        if query:
            return self.bar_df.query(query)

            # 构造日期切片
        if current_date:
            date_slice = slice(current_date, current_date)
        elif end_date:
            date_slice = slice(None, end_date)
        elif start_date:
            date_slice = slice(start_date, None)
        elif start_date and end_date:
            date_slice = slice(start_date, end_date)
        else:
            date_slice = slice(None, None)

        try:
            if symbol:
                # 索引顺序为 (trade_date, symbol)
                return self.bar_df.loc[date_slice].query(f'symbol == "{symbol}"')
            else:
                return self.bar_df.loc[date_slice]
        except KeyError:
            # 没有满足条件的数据时返回空DataFrame
            return pd.DataFrame()


class LocalDataHub(Datahub):
    """
    从按年拆分的低频CSV中加载数据，并提供按日期迭代的接口
    一个具体子类，必须实现3个抽象方法：
      - load_timeseries_data()
      - load_fundamental_data()
      - load_meta_data()
    否则会报错。
    """
    def load_bar_data(self):
        path = self.data_dict['bar']['path']
        if not os.path.exists(path):
            print(f"[WARN] Timeseries file not found: {path}")
            self.bar_df = pd.DataFrame()
            return

        # 命名标准化
        mapping = {value: key for key, value in self.data_dict['bar']['col_mapping'].items()}
        df = pd.read_csv(path)
        df.rename(columns=mapping, inplace=True)

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index(['trade_date', 'symbol'], inplace=True)
        df.sort_index(inplace=True, ascending=True)
        self.bar_df = df

    def load_fundamental_data(self):
        pass

    def load_info_data(self):
        path = self.data_dict['info']['path']
        if not os.path.exists(path):
            print(f"[WARN] file not found: {path}")
            self.bar_df = pd.DataFrame()
            return

        # 命名标准化
        mapping = {value: key for key, value in self.data_dict['info']['col_mapping'].items()}
        df = pd.read_csv(path)
        df.rename(columns=mapping, inplace=True)

        df.set_index(['symbol'], inplace=True)
        df.sort_index(inplace=True, ascending=True)
        self.info_df = df
