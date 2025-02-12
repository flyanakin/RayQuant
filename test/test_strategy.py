# tests/test_strategy.py
import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from core.strategy import MovingAverageStrategy, Signal
from core.datahub import Datahub  # 如果你想mock，可以替换成fake类


# 1) 构造一个最小化的 Datahub 类（或者使用 mock 库）

class FakeDatahub(Datahub):
    """
    仅做示例：返回特定日期范围内的测试行情数据，
    index = MultiIndex(symbol, trade_date)
    """
    def __init__(self):
        # 定义一个最小化的 dummy 数据字典，确保包含 bar 数据
        dummy_data_dict = {
            "bar": {
                "path": "",  # 这里路径可以不真实，因为不会实际读取
                "col_mapping": {"trade_date": "trade_date", "symbol": "symbol"}
            }
        }
        # 调用父类构造函数
        super().__init__(dummy_data_dict)

    def load_bar_data(self):
        # 对于 FakeDatahub，我们只写一个空实现，或者简单做一些 mock
        pass

    def load_fundamental_data(self):
        pass

    def load_info_data(self):
        pass

    def get_bars(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        # 生成时间序列
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        symbol_list = ['TEST']  # 假设我们只有一个symbol做测试

        # 构造多层索引 (symbol, trade_date)
        idx = pd.MultiIndex.from_product(
            [symbol_list, dates],
            names=['symbol', 'trade_date']
        )

        # 生成 close 数据：简单地做一个递增或随机
        data = np.linspace(100, 200, len(idx))  # 从100到200等步长

        # 创建 DataFrame
        df = pd.DataFrame({
            'close': data
        }, index=idx)

        return df


def test_moving_average_strategy():
    """
    测试 MovingAverageStrategy 的信号生成逻辑
    """
    # 2) 初始化一个 FakeDatahub
    hub = FakeDatahub()

    # 3) 初始化策略对象
    #    这里可根据实际测试需求进行参数调整
    strategy = MovingAverageStrategy(
        hub=hub,
        indicator='close',  # 使用close作为指标
        ma_buy=5,
        ma_sell=3,
        buy_bias=-0.2,
        sell_bias=0.2
    )

    # 4) 准备一个测试日期
    #    注意，因为我们在 FakeDatahub 中生成的数据是 start_date -> end_date 的逐日数据
    #    假设我们要测试 2025-01-10 这一天的信号
    current_time = pd.Timestamp('2025-01-10')

    # 5) 生成信号
    signal_obj = strategy.generate_signals(current_time=current_time)

    # 6) 对返回的 Signal 对象进行断言测试
    #    - 检查返回的 signal_obj 是否为 Signal 实例
    #    - 检查其内部 DataFrame 的结构是否符合预期
    assert signal_obj is not None, "生成的 Signal 对象为空"
    df_signal = signal_obj.get()

    # 7) 检查 DataFrame 的 multi-index、列名
    #    - index应包含 ('symbol', 'trade_date')
    assert 'symbol' in df_signal.index.names, "index 中缺少 'symbol'"
    assert 'trade_date' in df_signal.index.names, "index 中缺少 'trade_date'"
    #    - columns应至少包含 'signal'、'close'
    assert 'signal' in df_signal.columns, "signal DataFrame中缺少 'signal' 列"
    assert 'close' in df_signal.columns, "signal DataFrame中缺少 'close' 列"

    # 8) 进一步检查对应日期的行是否只有一行 (因为我们只有一个symbol)
    assert len(df_signal) == 1, "测试示例中只期望1个symbol在该日出现信号"

    # 9) 你可以检查具体值，比如 signal 列生成 'BUY'/'SELL'/'None'
    #    这需要根据生成数据和策略参数来判定，下面仅给示例断言
    possible_signals = {'BUY', 'SELL', None}
    gen_signal = df_signal['signal'].iloc[0]
    assert gen_signal in possible_signals, f"信号不在预期范围内: {gen_signal}"

    # 如果你想更进一步检验数值，可以print出来看
    print(df_signal)

    # 如果想要验证不会出现未来数据，可以检查日志或捕获 warnings
    # 也可以增加更多情形的单元测试（比如异常输入、参数越界等等）


def test_signal_missing_cols():

    # 缺少'required_columns'中的'signal'
    data = pd.DataFrame({
        'close': [100, 101, 102],
    })
    # 构造多层索引，但只有 'trade_date'
    idx = pd.MultiIndex.from_arrays(
        [
            ['2025-01-10', '2025-01-10', '2025-01-10']
        ],
        names=['trade_date']
    )
    data.index = idx

    current_time = pd.Timestamp('2025-01-10')
    # 预期抛出 ValueError
    with pytest.raises(ValueError, match="缺少必需的列"):
        Signal(data, current_time)


def test_signal_missing_index():

    # 虽然包含 'signal' 和 'close' 列，但索引不完整
    data = pd.DataFrame({
        'close': [100, 101, 102],
        'signal': ['BUY', 'SELL', None]
    })
    # 只定义了一个普通索引
    data.index = [0, 1, 2]

    current_time = pd.Timestamp('2025-01-10')
    # 预期抛出 ValueError
    with pytest.raises(ValueError, match="缺少必需的索引"):
        Signal(data, current_time)