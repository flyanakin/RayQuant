import pandas as pd
import numpy as np
import pytest
from utils.analysis import risk_and_return, get_matrix
from pandas.testing import assert_frame_equal
from datetime import datetime


@pytest.fixture
def sample_data_multiple():
    """
    构造多标的的测试数据，数据采用几何增长保证每日收益率稳定，
    同时方便验证归一化收益、回撤等指标。
    """
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    # 对于几何增长，百分比收益固定：
    # symbol A: 初始值100，每日增长1%
    A = 100 * np.power(1.01, np.arange(50))
    # symbol B: 初始值200，每日增长1%
    B = 200 * np.power(1.01, np.arange(50))
    df = pd.DataFrame({'A': A, 'B': B}, index=dates)
    return df


@pytest.fixture
def sample_data_single():
    """
    构造单标的测试数据
    """
    dates = pd.date_range(start="2021-01-01", periods=60, freq="D")
    # symbol A: 初始值100，每日增长1%
    A = 100 * np.power(1.01, np.arange(60))
    df = pd.DataFrame({'A': A}, index=dates)
    return df


def test_structure(sample_data_multiple):
    """
    测试返回结果包含 'result_df', 'return_lines' 和 'drawdown_df' 三个键，
    同时验证各部分的索引和维度是否符合预期。
    """
    result = risk_and_return(sample_data_multiple, interval_months=3)

    # 检查返回字典的键
    for key in ['result_df', 'return_lines', 'drawdown_df']:
        assert key in result, f"返回结果中缺少键 {key}"

    # 检查归一化收益序列：首日的数值应均为1
    first_row = result['return_lines'].iloc[0]
    np.testing.assert_allclose(first_row.values, np.ones(len(first_row)), atol=1e-6)

    # 检查 result_df 的索引与原数据列名一致
    expected_symbols = pd.Index(sample_data_multiple.columns, name='symbol')
    pd.testing.assert_index_equal(result['result_df'].index, expected_symbols)

    # 检查 drawdown_df 的索引与原数据日期一致
    pd.testing.assert_index_equal(result['drawdown_df'].index, sample_data_multiple.index)


def test_value_types(sample_data_single):
    """
    验证单一标的时返回的指标数据类型正确，
    并且最大回撤区间为元组，包含两个日期信息。
    """
    result = risk_and_return(sample_data_single, interval_months=3)
    metrics = result['result_df'].loc['A']

    # 年化收益率、年化波动率、最大回撤均为 float 类型
    assert isinstance(metrics['annual_return'], float)
    assert isinstance(metrics['annual_volatility'], float)
    assert isinstance(metrics['max_drawdown'], float)
    # 最大回撤对应区间为 tuple，且内部应包含两个 pd.Timestamp
    assert isinstance(metrics['max_drawdown_interval'], tuple)
    assert len(metrics['max_drawdown_interval']) == 2
    start_date, end_date = metrics['max_drawdown_interval']
    assert isinstance(start_date, pd.Timestamp)
    assert isinstance(end_date, pd.Timestamp)


def test_input_not_modified(sample_data_single):
    """
    验证 risk_and_return 函数不会修改输入的 DataFrame
    """
    original = sample_data_single.copy()
    _ = risk_and_return(sample_data_single, interval_months=3)
    pd.testing.assert_frame_equal(sample_data_single, original)


def test_with_benchmarks(sample_data_multiple):
    """
    测试传入 benchmarks 参数时函数仍然正常运行，
    尽管当前实现未对 benchmarks 做特殊处理。
    """
    benchmarks = ['spx', '000300.SH']
    result = risk_and_return(sample_data_multiple, interval_months=3, benchmarks=benchmarks)
    for key in ['result_df', 'return_lines', 'drawdown_df']:
        assert key in result, f"传入 benchmarks 后返回结果中缺少键 {key}"


class TestBasicFunction:
    """测试基础功能"""

    def test_normal_case(self):
        # 构造测试数据
        df1 = pd.DataFrame({
            'trade_date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'symbol': ['A', 'A'],
            'close': [10.0, 11.0]
        })
        df2 = pd.DataFrame({
            'trade_date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'symbol': ['B', 'B'],
            'close': [20.0, 21.0]
        })

        result = get_matrix(
            dfs=[df1, df2],
            time_range=(datetime(2023, 1, 1), datetime(2023, 1, 2)),
            metric_col='close'
        )

        # 验证数据结构
        expected = pd.DataFrame(
            [[10.0, 20.0], [11.0, 21.0]],
            index=pd.date_range('2023-01-01', '2023-01-02', freq='D'),
            columns=['A', 'B'],
        )
        expected.index.name = 'trade_date'
        assert_frame_equal(result, expected, check_names=False)


class TestInvalidInputs:
    """测试异常输入"""

    def test_empty_dfs(self):
        with pytest.raises(ValueError) as excinfo:
            get_matrix([], (datetime(2023, 1, 1), datetime(2023, 1, 2)), 'close')
        assert "没有满足时间范围的数据" in str(excinfo.value)

    def test_missing_symbol_column(self):
        df = pd.DataFrame({'trade_date': [datetime(2023, 1, 1)], 'close': [10.0]})
        with pytest.raises(ValueError) as excinfo:
            get_matrix([df], (datetime(2023, 1, 1), datetime(2023, 1, 1)), 'close')
        assert "缺少'symbol'字段" in str(excinfo.value)


class TestDateTimeHandling:
    """测试日期处理"""

    def test_string_date_conversion(self):
        df = pd.DataFrame({
            'trade_date': ['2023-01-01', '2023-01-02'],
            'symbol': ['A', 'A'],
            'close': [10.0, 11.0]
        })
        result = get_matrix(
            [df],
            (datetime(2023, 1, 1), datetime(2023, 1, 2)),
            'close'
        )
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_time_filtering(self):
        df = pd.DataFrame({
            'trade_date': [datetime(2022, 12, 31), datetime(2023, 1, 1)],
            'symbol': ['A', 'A'],
            'close': [9.0, 10.0]
        })
        result = get_matrix(
            [df],
            (datetime(2023, 1, 1), datetime(2023, 1, 1)),
            'close'
        )
        assert len(result) == 1
        assert result.iloc[0, 0] == 10.0


class TestResampling:
    """测试重采样逻辑"""

    def test_weekly_resample(self):
        dates = [
            datetime(2023, 1, 31), datetime(2023, 2, 28),
            datetime(2023, 3, 31), datetime(2023, 4, 30)
        ]
        df = pd.DataFrame({
            'trade_date': dates,
            'symbol': ['A'] * 4,
            'close': [10.0, 11.0, 12.0, 13.0]
        })
        result = get_matrix(
            [df],
            time_range=(datetime(2023, 1, 1), datetime(2023, 3, 1)),
            metric_col='close',
            period='M'
        )
        expected_dates = pd.date_range('2023-01-31', '2023-02-28', freq='M')
        assert result.index.equals(expected_dates)
        assert result['A'].tolist() == [10.0, 11.0]


def test_ffill_with_warning():
    df = pd.DataFrame({
        'trade_date': [datetime(2023, 1, 2)],
        'symbol': ['A'],
        'close': [10.0]
    })
    with pytest.warns(UserWarning) as record:
        result = get_matrix(
            [df],
            (datetime(2023, 1, 1), datetime(2023, 1, 2)),
            'close'
        )
    # 检查警告内容中包含提示信息
    assert "起始处无有效数据" in str(record[0].message)
    # 因为完整索引第一行无前置数据，前向填充不会填充，故应为 NaN
    assert pd.isna(result['A'].iloc[0])
    # 后续行会前向填充，故第二行应为 10.0
    assert result['A'].iloc[1] == 10.0


class TestLogReturn:
    """测试对数收益率"""

    def test_log_return_calculation(self):
        df = pd.DataFrame({
            'trade_date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'symbol': ['A', 'A'],
            'close': [100.0, 105.0]
        })
        result = get_matrix([df], (datetime(2023, 1, 1), datetime(2023, 1, 2)),
                            'close', log_return=True)
        expected = np.log(105 / 100)
        assert np.isclose(result['A'].iloc[0], expected)
        assert len(result) == 1  # 删除首行NaN


class TestSpecialCases:
    """测试特殊参数"""

    def test_not_implemented_period(self):
        df = pd.DataFrame({
            'trade_date': [datetime(2023, 1, 1)],
            'symbol': ['A'],
            'close': [10.0]
        })
        with pytest.raises(NotImplementedError):
            get_matrix([df], (datetime(2023, 1, 1), datetime(2023, 1, 1)),
                       'close', period='N')
