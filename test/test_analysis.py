import pandas as pd
import numpy as np
import pytest
from utils.analysis import risk_and_return


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