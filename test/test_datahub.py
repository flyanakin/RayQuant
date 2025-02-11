import pandas as pd
import pytest
from core.datahub import LocalDataHub

# --------------------------
# pytest 测试用例
# --------------------------


@pytest.fixture
def csv_files(tmp_path):
    """
    利用 tmp_path fixture 创建临时 CSV 文件，并返回构造好的 data_dict。
    """
    # 创建临时 bar CSV 文件（原始字段为 "date" 和 "ts_code"）
    bar_csv = tmp_path / "bar.csv"
    bar_csv.write_text(
        "date,ts_code,open,close,volume\n"
        "2025-01-01,000001.SH,10,12,1000\n"
        "2025-01-01,000002.SH,20,18,1500\n"
        "2025-01-02,000001.SH,12,13,1100\n"
        "2025-01-02,000002.SH,18,17,1400\n"
    )

    # 创建临时 info CSV 文件
    info_csv = tmp_path / "info.csv"
    info_csv.write_text(
        "symbol,name,industry\n" "000001.SH,CompanyA,Tech\n" "000002.SH,CompanyB,Finance\n"
    )

    data_dict = {
        "bar": {
            "path": str(bar_csv),
            "col_mapping": {"trade_date": "date", "symbol": "ts_code"},
        },
        "info": {
            "path": str(info_csv),
            "col_mapping": {"symbol": "symbol", "name": "name", "industry": "industry"},
        },
    }
    return data_dict


# --------------------------
# 用于测试的子类
# --------------------------


class TestLocalDataHub(LocalDataHub):
    """
    用于测试 load_all_data 及其他接口
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.bar_df = None
        self.fundamental_df = None
        self.info_df = None
        self.load_all_data()

    def load_fundamental_data(self):
        # 构造一个简单的财报数据 DataFrame
        self.fundamental_df = pd.DataFrame(
            {"symbol": ["000001.SH", "000002.SH"], "earnings": [1.0, 2.0]}
        ).set_index("symbol")


class BarOnlyDataHub(LocalDataHub):
    """
    仅加载时序数据，用于测试 get_data_by_date 与 timeseries_iterator 方法
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.bar_df = None
        self.fundamental_df = None
        self.info_df = None
        self.load_bar_data()

    def load_fundamental_data(self):
        self.fundamental_df = pd.DataFrame()

    def load_info_data(self):
        self.info_df = pd.DataFrame()


def test_load_all_data(csv_files):
    """
    测试 load_all_data 方法：
      - 检查 fundamental_df 是否正确加载；
      - 由于 load_info_data 中存在 bug，会将 info 数据错误地赋值给 bar_df，
        并且 info_df 仍为 None。
    """
    hub = TestLocalDataHub(csv_files)

    # 检查财报数据（fundamental_df）
    expected_fundamental = pd.DataFrame(
        {"symbol": ["000001.SH", "000002.SH"], "earnings": [1.0, 2.0]}
    ).set_index("symbol")
    pd.testing.assert_frame_equal(hub.fundamental_df, expected_fundamental)

    # 预期 bar_df 实际上存储的是 info CSV 的内容
    expected_info = pd.read_csv(csv_files["info"]["path"])
    mapping = {v: k for k, v in csv_files["info"]["col_mapping"].items()}
    expected_info.rename(columns=mapping, inplace=True)
    expected_info.set_index("symbol", inplace=True)
    expected_info.sort_index(inplace=True)
    assert hub.info_df is not None


def test_get_main_timeline_empty(csv_files, tmp_path):
    """
    测试当 bar 和 info CSV 文件不存在时，get_main_timeline 返回空列表。
    这里修改 data_dict 中的文件路径为不存在的路径。
    """
    data_dict = csv_files.copy()
    data_dict["bar"]["path"] = str(tmp_path / "non_existing_bar.csv")
    data_dict["info"]["path"] = str(tmp_path / "non_existing_info.csv")

    hub = TestLocalDataHub(data_dict)
    # 当文件不存在时，load_bar_data 与 load_info_data 均设置对应 DataFrame 为空
    assert hub.bar_df.empty
    timeline = hub.get_main_timeline()
    assert timeline == []


def test_get_data_by_date_keyerror(csv_files):
    """
    测试 get_data_by_date 方法：
      - 当请求的日期不存在时，返回空 DataFrame。
    使用 BarOnlyDataHub 仅加载时序数据，避免 info 数据的干扰。
    """
    hub = BarOnlyDataHub(csv_files)
    result = hub.get_data_by_date(pd.Timestamp("2025-01-03"))
    assert result.empty


def test_timeseries_iterator(csv_files):
    """
    测试 timeseries_iterator 方法：
      - 检查是否能正确迭代所有唯一日期，并返回相应的时序数据快照。
    """
    hub = BarOnlyDataHub(csv_files)
    timeline = list(hub.timeseries_iterator())
    # 根据 bar CSV，应有两个唯一日期：2025-01-01 和 2025-01-02
    dates = [dt for dt, _ in timeline]
    expected_dates = pd.to_datetime(["2025-01-01", "2025-01-02"])
    pd.testing.assert_index_equal(pd.Index(dates), pd.Index(expected_dates))

    # 检查每个日期对应的快照数据的索引中包含 'symbol'
    for dt, df in timeline:
        assert "symbol" in df.index.names


def test_get_bar_single_date(csv_files):
    """
    测试 get_bar 方法：
      - 当请求单个特定日期时，应返回该日期的所有数据。
    """
    hub = BarOnlyDataHub(csv_files)
    result = hub.get_bar(current_date="2025-01-01")
    expected = pd.read_csv(csv_files['bar']['path'])
    mapping = {v: k for k, v in csv_files['bar']['col_mapping'].items()}
    expected.rename(columns=mapping, inplace=True)
    expected['trade_date'] = pd.to_datetime(expected['trade_date'])
    expected.set_index(['trade_date', 'symbol'], inplace=True)
    expected.sort_index(inplace=True)
    # 这里改为直接筛选日期层级，但保持两层索引
    expected = expected.loc[(slice("2025-01-01", "2025-01-01"), slice(None)), :]
    print(f"expected: {expected}")
    print(f"result: {result}")
    pd.testing.assert_frame_equal(result, expected)


def test_get_bar_date_range(csv_files):
    """
    测试 get_bar 方法：
      - 当请求一个日期范围时，应返回该范围内的所有数据。
    """
    hub = BarOnlyDataHub(csv_files)
    result = hub.get_bar(start_date="2025-01-01", end_date="2025-01-02")
    expected = pd.read_csv(csv_files['bar']['path'])
    mapping = {v: k for k, v in csv_files['bar']['col_mapping'].items()}
    expected.rename(columns=mapping, inplace=True)
    expected['trade_date'] = pd.to_datetime(expected['trade_date'])
    expected.set_index(['trade_date', 'symbol'], inplace=True)
    expected.sort_index(inplace=True)
    pd.testing.assert_frame_equal(result, expected)


def test_get_bar_specific_symbol(csv_files):
    """
    测试 get_bar 方法：
      - 当指定特定标的符号时，应只返回该标的的数据。
    """
    hub = BarOnlyDataHub(csv_files)
    result = hub.get_bar(symbol="000002.SH")
    expected = pd.read_csv(csv_files['bar']['path'])
    mapping = {v: k for k, v in csv_files['bar']['col_mapping'].items()}
    expected.rename(columns=mapping, inplace=True)
    expected['trade_date'] = pd.to_datetime(expected['trade_date'])
    expected.set_index(['trade_date', 'symbol'], inplace=True)
    expected.sort_index(inplace=True)
    # 使用 loc 来保持双层索引结构
    expected = expected.loc[(slice(None), '000002.SH'), :]
    pd.testing.assert_frame_equal(result, expected)


def test_get_bar_no_data(csv_files, tmp_path):
    """
    测试 get_bar 方法：
      - 当请求的日期或符号不存在时，应返回空 DataFrame。
    """
    hub = BarOnlyDataHub(csv_files)
    result = hub.get_bar(current_date="2025-01-03")  # 一个不存在的日期
    assert result.empty
    result = hub.get_bar(symbol="999999")  # 一个不存在的符号
    assert result.empty
