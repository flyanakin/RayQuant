import pandas as pd


class Timeline:
    def __init__(
            self,
            bar_df: pd.DataFrame,
            how: str = 'union',
            fillna: str = 'forward'
    ):
        """
        :param bar_df: 已完成标准化处理的bar数据，trade_date和symbol为多重index，
        :param how: 时间线合并策略，默认为union，则所有数据集的trade_date取并集
        :param fillna: 时间线空值处理方法，需要修改self.data_hub.bar_df的数据
                    'ignore': 不处理
                    'forward': 用上一个非空值来填充（时间升序），如果前面全是空值，则不处理。并且在bar_df增加一个字段，标识数据被处理过
        """
        self.bar_df = bar_df
        self.how = how
        self.fillna = fillna

    def get_main_timeline(
            self,
    ) -> list[pd.Timestamp]:
        """
        获取主时间轴
        :return:
               main_timeline: 时间序列
        """
        if self.bar_df is None or self.bar_df.empty:
            raise ValueError("无数据输入")

        if self.how == 'union':
            # 获取所有 daily 数据中的日期并集
            main_timeline = self.bar_df.index.get_level_values(0).unique()

            # 针对 daily 数据的每个 symbol 检查主时间线中缺失的日期
            daily_symbols = self.bar_df.index.get_level_values(1).unique()
            for sym in daily_symbols:
                # 获取该 symbol 在 daily 数据中出现的所有日期
                sym_dates = self.bar_df.loc[(slice(None), sym), :].index.get_level_values(0).unique()
                # 缺失的日期：在主时间线中，但该 symbol 未出现的数据日期
                missing_dates = main_timeline.difference(sym_dates)
                if not missing_dates.empty:
                    print(f"[INFO] Daily 数据中 symbol '{sym}' 缺失日期: {sorted(missing_dates)}")
        else:
            pass

        # TODO: 实现向前补充值

        return main_timeline

    def timeseries_iterator(self):
        """
        非抽象方法，用于回测引擎迭代。
        """
        timeline = self.get_main_timeline()
        for dt in timeline:
            yield dt
