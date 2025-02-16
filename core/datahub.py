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
               data_dict = {
                    "bar": {
                         "daily": [
                             {
                                 "path": "data/daily1.csv",
                                 "col_mapping": {
                                     "trade_date": "trade_date",  # 原始文件字段名 : 标准字段名
                                     "ts_code": "symbol",
                                     "open_price": "open",
                                     # 其他字段映射……
                                 }
                             },
                             {
                                 "path": "data/daily2.csv",
                                 "col_mapping": { ... }
                             },
                         ],
                         "benchmark": [
                             {
                                 "path": "data/benchmark.csv",
                                 "col_mapping": {
                                     "trade_date": "trade_date",
                                     "ts_code": "symbol",
                                     "close_price": "close",
                                     # ……
                                 }
                             }
                         ]
                    },
                    "info": {
                         "path": "data/info.csv",
                         "col_mapping": {
                             "股票代码": "symbol",
                             "股票名称": "name",
                             # ……
                         }
                    },
                    "fundamental": {
                         "path": "data/fundamental.csv",
                         "col_mapping": { ... }
                    }
                }
        """

        self.data_dict = data_dict

        # 下面是内部 DataFrame 存储
        self.bar_df = None  # 用于存放日线时序数据
        self.benchmark_df = None  # 用于存放benchmark时序数据
        self.fundamental_df = None  # 用于存放财报数据
        self.info_df = None  # 用于存放元数据

    @abstractmethod
    def load_bar_data(
            self,
            start_date: pd.Timestamp = None,
            end_date: pd.Timestamp = None,
            symbols: list[str] = None,
            benchmarks: list[str] = None,
    ):
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

    def load_all_data(
            self,
            start_date: pd.Timestamp = None,
            end_date: pd.Timestamp = None,
            benchmarks: list[str] = None,
            symbols: list[str] = None,
    ):
        """
        一键加载所有数据。
        这里不是抽象方法，因为基类可直接调用子类实现的抽象方法。
        """
        if ('bar' not in self.data_dict or
                'daily' not in self.data_dict['bar'] or
                'benchmark' not in self.data_dict['bar']):
            raise ValueError("data_dict must contain 'bar' with both 'daily' and 'benchmark' data.")

        if 'info' in self.data_dict:
            self.load_info_data()
        if 'fundamental' in self.data_dict:
            self.load_fundamental_data()
        self.load_bar_data(start_date=start_date, end_date=end_date, symbols=symbols, benchmarks=benchmarks)

    def get_main_timeline(self):
        """
        非抽象方法，可直接在基类中实现。
        """
        if (self.bar_df is None or self.bar_df.empty or
                self.benchmark_df is None or self.benchmark_df.empty):
            return []

        # 获取所有 daily 数据中的日期并集
        daily_union = self.bar_df.index.get_level_values(0).unique()
        # 获取 benchmark 数据中的所有日期
        benchmark_dates = self.benchmark_df.index.get_level_values(0).unique()
        # 主时间线为 daily 并集与 benchmark 日期的交集
        main_timeline = daily_union.intersection(benchmark_dates).sort_values()

        # 针对 daily 数据的每个 symbol 检查主时间线中缺失的日期
        daily_symbols = self.bar_df.index.get_level_values(1).unique()
        for sym in daily_symbols:
            # 获取该 symbol 在 daily 数据中出现的所有日期
            sym_dates = self.bar_df.loc[(slice(None), sym), :].index.get_level_values(0).unique()
            # 缺失的日期：在主时间线中，但该 symbol 未出现的数据日期
            missing_dates = main_timeline.difference(sym_dates)
            if not missing_dates.empty:
                print(f"[INFO] Daily 数据中 symbol '{sym}' 缺失日期: {sorted(missing_dates)}")

        # 针对 benchmark 数据的每个 symbol 检查主时间线中缺失的日期
        benchmark_symbols = self.benchmark_df.index.get_level_values(1).unique()
        for sym in benchmark_symbols:
            sym_dates = self.benchmark_df.loc[(slice(None), sym), :].index.get_level_values(0).unique()
            missing_dates = main_timeline.difference(sym_dates)
            if not missing_dates.empty:
                print(f"[INFO] Benchmark 数据中 symbol '{sym}' 缺失日期: {sorted(missing_dates)}")

        return main_timeline

    def get_data_by_date(self, current_date):
        """
        在特定日期，返回所有标的的时序数据快照(行索引= symbol)。
        """
        if self.bar_df is None:
            raise ValueError(
                "Timeseries data not loaded. Call load_bar_data() first."
            )

        try:
            return self.bar_df.xs(current_date, level=0)
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
                 query: str = None,
                 benchmark: bool = False,
        ) -> pd.DataFrame:
        """
        在特定日期区间，返回所有标的的行情数据快照(行索引= symbol)。
        :param current_date: 当前或特定的单一日期
        :param start_date: 起始日期
        :param end_date: 结束日期
        :param symbol: 标的的唯一标识
        :param query: 其他查询条件，预留参数
        :param benchmark: 如果为True则获取benchmark的数据
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

        if benchmark:
            df = self.benchmark_df
        else:
            df = self.bar_df

        if query:
            return df.query(query)

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
                return df.loc[date_slice].query(f'symbol == "{symbol}"').copy()
            else:
                return df.loc[date_slice].copy()
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

    def _load_csv_file(self,
                       file_info: dict,
                       start_date: pd.Timestamp = None,
                       end_date: pd.Timestamp = None,
                       apply_date_filter: bool = False,
                       symbol_filter: list[str] = None
                ) -> pd.DataFrame:
        """
        通用的 CSV 数据加载方法：
          - 检查文件存在性，不存在时返回空 DataFrame，并打印警告信息；
          - 重命名字段，转换 trade_date 为 pd.Timestamp；
          - 根据需要对日期区间进行过滤（仅当 apply_date_filter=True 且 start_date 与 end_date 均不为空时）；
          - 检查是否存在 'symbol' 字段，最后设置索引 (trade_date, symbol) 并排序。
        """
        path = file_info.get('path')
        if not os.path.exists(path):
            print(f"[WARN] 文件不存在: {path}")
            return pd.DataFrame()

        # 构造映射：原始字段名 -> 标准字段名
        mapping = {std: orig for std, orig in file_info.get('col_mapping', {}).items()}
        df = pd.read_csv(path)
        df.rename(columns=mapping, inplace=True)

        if 'trade_date' not in df.columns:
            raise ValueError(f"文件 {path} 缺失 'trade_date' 字段")
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        if apply_date_filter and start_date is not None and end_date is not None:
            df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]

        if 'symbol' not in df.columns:
            raise ValueError(f"文件 {path} 缺失 'symbol' 字段")

        # 根据 symbol_filter 过滤数据（如果传入了非空列表，则只保留指定 symbol 的数据）
        if symbol_filter:
            df = df[df['symbol'].isin(symbol_filter)]
        df.set_index(['trade_date', 'symbol'], inplace=True)
        df.sort_index(inplace=True)
        return df

    def load_bar_data(self,
                      start_date: pd.Timestamp = None,
                      end_date: pd.Timestamp = None,
                      benchmarks: list[str] = None,
                      symbols: list[str] = None
                      ):
        """
        加载 daily 与 benchmark 数据，daily 数据可根据 start_date 与 end_date 进行过滤，
        benchmark 数据保持全量。
        """
        # 读取 daily 数据
        daily_files = self.data_dict['bar']['daily']
        daily_dfs = []
        for file_info in daily_files:
            df = self._load_csv_file(file_info,
                                     start_date=start_date,
                                     end_date=end_date,
                                     apply_date_filter=True,
                                     symbol_filter=symbols
                                     )
            if not df.empty:
                daily_dfs.append(df)
        self.bar_df = pd.concat(daily_dfs) if daily_dfs else pd.DataFrame()

        # 读取 benchmark 数据（不做日期过滤）
        benchmark_files = self.data_dict['bar']['benchmark']
        benchmark_dfs = []
        for file_info in benchmark_files:
            df = self._load_csv_file(
                file_info,
                start_date=start_date,
                end_date=end_date,
                apply_date_filter=True,
                symbol_filter=benchmarks
            )
            if not df.empty:
                benchmark_dfs.append(df)
        self.benchmark_df = pd.concat(benchmark_dfs) if benchmark_dfs else pd.DataFrame()

    def load_fundamental_data(self):
        pass

    def load_info_data(self):
        path = self.data_dict['info']['path']
        if not os.path.exists(path):
            print(f"[WARN] file not found: {path}")
            self.bar_df = pd.DataFrame()
            return

        # 命名标准化
        mapping = {key: value for key, value in self.data_dict['info']['col_mapping'].items()}
        df = pd.read_csv(path)
        df.rename(columns=mapping, inplace=True)

        df.set_index(['symbol'], inplace=True)
        df.sort_index(inplace=True, ascending=True)
        self.info_df = df
