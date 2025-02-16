import os
import pandas as pd
import pytest
from datetime import datetime
from core.datahub import LocalDataHub


# ========== Fixture：构造测试所需的 CSV 文件 ==========

@pytest.fixture
def daily_csv(tmp_path):
    """
    构造一个包含两只股票、两天数据的 daily 数据 CSV 文件
    """
    data = [
        ["2021-01-01", "000001.SH", 10.0],
        ["2021-01-02", "000001.SH", 10.5],
        ["2021-01-01", "000002.SH", 20.0],
        ["2021-01-02", "000002.SH", 20.5],
    ]
    df = pd.DataFrame(data, columns=["trade_date", "ts_code", "open_price"])
    file_path = tmp_path / "daily.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def daily_missing_csv(tmp_path):
    """
    构造一个 daily 数据 CSV 文件，其中 symbol '000002' 缺少 2021-01-02 的数据
    """
    data = [
        ["2021-01-01", "000001.SH", 10.0],
        ["2021-01-02", "000001.SH", 10.5],
        ["2021-01-01", "000002.SH", 20.0],
        # 000002 缺少 2021-01-02 的数据
    ]
    df = pd.DataFrame(data, columns=["trade_date", "ts_code", "open_price"])
    file_path = tmp_path / "daily_missing.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def benchmark_csv(tmp_path):
    """
    构造 benchmark 数据 CSV 文件
    """
    data = [
        ["2021-01-01", "bench1", 100.0],
        ["2021-01-02", "bench1", 101.0],
    ]
    df = pd.DataFrame(data, columns=["trade_date", "ts_code", "close_price"])
    file_path = tmp_path / "benchmark.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def info_csv(tmp_path):
    """
    构造 info 数据 CSV 文件
    """
    data = [
        ["000001.SH", "Test Corp"],
        ["000002.SH", "Another Corp"],
    ]
    df = pd.DataFrame(data, columns=["股票代码", "股票名称"])
    file_path = tmp_path / "info.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def data_dict(daily_csv, benchmark_csv, info_csv):
    """
    构造一个完整的 data_dict，其中包含 daily、benchmark 及 info 数据的路径及字段映射
    """
    return {
        "bar": {
            "daily": [
                {
                    "path": str(daily_csv),
                    "col_mapping": {
                        "trade_date": "trade_date",
                        "ts_code": "symbol",
                        "open_price": "open"
                    }
                }
            ],
            "benchmark": [
                {
                    "path": str(benchmark_csv),
                    "col_mapping": {
                        "trade_date": "trade_date",
                        "ts_code": "symbol",
                        "close_price": "close"
                    }
                }
            ]
        },
        "info": {
            "path": str(info_csv),
            "col_mapping": {
                "股票代码": "symbol",
                "股票名称": "name"
            }
        }
        # 此处不包含 fundamental 数据
    }


@pytest.fixture
def data_dict_missing(daily_missing_csv, benchmark_csv, info_csv):
    """
    构造一个 data_dict，其中 daily 数据中存在缺失日期的情况
    """
    return {
        "bar": {
            "daily": [
                {
                    "path": str(daily_missing_csv),
                    "col_mapping": {
                        "trade_date": "trade_date",
                        "ts_code": "symbol",
                        "open_price": "open"
                    }
                }
            ],
            "benchmark": [
                {
                    "path": str(benchmark_csv),
                    "col_mapping": {
                        "trade_date": "trade_date",
                        "ts_code": "symbol",
                        "close_price": "close"
                    }
                }
            ]
        },
        "info": {
            "path": str(info_csv),
            "col_mapping": {
                "股票代码": "symbol",
                "股票名称": "name"
            }
        }
    }


# ========== 测试函数 ==========

def test_load_all_data(data_dict):
    """
    测试 load_all_data 方法：检查 daily、benchmark、info 数据是否正确加载及索引设置
    """
    hub = LocalDataHub(data_dict)
    start_date = pd.Timestamp("2021-01-01")
    end_date = pd.Timestamp("2021-01-02")
    hub.load_all_data(start_date=start_date, end_date=end_date)

    # 检查 bar_df
    assert hub.bar_df is not None
    # 应该是 MultiIndex，索引顺序为 (trade_date, symbol)
    assert isinstance(hub.bar_df.index, pd.MultiIndex)
    assert (pd.Timestamp("2021-01-01"), "000001.SH") in hub.bar_df.index
    assert (pd.Timestamp("2021-01-02"), "000002.SH") in hub.bar_df.index

    # 检查 benchmark_df
    assert hub.benchmark_df is not None
    assert isinstance(hub.benchmark_df.index, pd.MultiIndex)
    assert (pd.Timestamp("2021-01-01"), "bench1") in hub.benchmark_df.index

    # 检查 info_df：index 应该为 symbol
    assert hub.info_df is not None
    assert hub.info_df.index.name == "symbol"
    assert "name" in hub.info_df.columns


def test_get_main_timeline(data_dict, capsys):
    """
    测试 get_main_timeline 方法：返回 daily 与 benchmark 日期的交集，
    并检查当某个 symbol 缺失日期时不会产生打印提示（此测试使用完整数据）
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    timeline = hub.get_main_timeline()
    print(f"timeline:: {timeline}")
    print(f"timeline_type:: {type(timeline)}")
    expected = pd.to_datetime(["2021-01-01", "2021-01-02"])
    expected.name = "trade_date"
    pd.testing.assert_index_equal(timeline, expected)

    captured = capsys.readouterr().out
    # 对于完整数据，不应该打印任何 [INFO] 提示
    assert "[INFO]" not in captured


def test_get_main_timeline_missing_dates(data_dict_missing, capsys):
    """
    测试当 daily 数据中某个 symbol 缺失日期时，get_main_timeline 会打印出提示信息
    """
    hub = LocalDataHub(data_dict_missing)
    hub.load_all_data()
    timeline = hub.get_main_timeline()
    expected = pd.to_datetime(["2021-01-01", "2021-01-02"])
    expected.name = "trade_date"
    pd.testing.assert_index_equal(timeline, expected)

    captured = capsys.readouterr().out
    # symbol '000002' 缺失 2021-01-02 的数据，应打印提示
    assert "Daily 数据中 symbol '000002.SH' 缺失日期" in captured


def test_get_data_by_date(data_dict):
    """
    测试 get_data_by_date 方法：检查在特定日期返回的 daily 数据快照
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    snapshot = hub.get_data_by_date(pd.Timestamp("2021-01-01"))
    # daily 数据中应包含 symbol '000001' 与 '000002'
    assert "000001.SH" in snapshot.index
    assert "000002.SH" in snapshot.index


def test_timeseries_iterator(data_dict):
    """
    测试 timeseries_iterator 方法：迭代返回的日期与对应快照
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    timeline_list = list(hub.timeseries_iterator())
    expected_dates = pd.to_datetime(["2021-01-01", "2021-01-02"])
    dates = [item[0] for item in timeline_list]
    pd.testing.assert_index_equal(pd.Index(dates), pd.Index(expected_dates))


def test_get_bars_current_date(data_dict):
    """
    测试 get_bars 方法中 current_date 与 symbol 过滤的功能
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    # 指定 current_date 返回该日的所有数据
    bars = hub.get_bars(current_date=pd.Timestamp("2021-01-01"))
    # 检查 daily 数据中存在 (2021-01-01, "000001.SH")
    assert (pd.Timestamp("2021-01-01"), "000001.SH") in hub.bar_df.index

    # 再使用 symbol 过滤，只返回 '000001' 的数据
    bars_symbol = hub.get_bars(current_date=pd.Timestamp("2021-01-01"), symbol="000001.SH")
    for sym in bars_symbol.index.get_level_values(1):
        assert sym == "000001.SH"


def test_get_bars_query(data_dict):
    """
    测试 get_bars 方法中的 query 参数：过滤 open 值大于 10.0 的记录
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    bars = hub.get_bars(query="open > 10.0")
    # 根据 daily 数据，满足条件的记录 open 分别为 10.5、20.0、20.5
    assert not bars.empty
    assert all(bars['open'] > 10.0)


def test_get_bars_invalid_date(data_dict):
    """
    测试 get_bars 方法中，当传入非 pd.Timestamp 类型的日期时应抛出异常
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    with pytest.raises(ValueError):
        hub.get_bars(current_date="2021-01-01")  # 传入字符串，应该抛出异常


def test_get_bars_date_range(data_dict):
    """
    测试 get_bars 方法中使用 start_date 与 end_date 进行日期切片
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    # 传入 end_date，仅返回日期不大于 2021-01-01 的数据
    bars = hub.get_bars(end_date=pd.Timestamp("2021-01-01"))
    # 检查所有返回数据的日期都不大于 2021-01-01
    for (date, symbol) in bars.index:
        assert date <= pd.Timestamp("2021-01-01")


def test_get_bars_invalid_date_range(data_dict):
    """
    测试当 start_date 大于 end_date 时，get_bars 应抛出异常
    """
    hub = LocalDataHub(data_dict)
    hub.load_all_data()
    with pytest.raises(ValueError):
        hub.get_bars(start_date=pd.Timestamp("2021-01-02"), end_date=pd.Timestamp("2021-01-01"))


def test_missing_info_file(tmp_path, data_dict):
    """
    测试 info 数据文件不存在时，load_info_data 会打印警告信息，并设置相应状态
    """
    data_dict_modified = data_dict.copy()
    data_dict_modified["info"] = data_dict_modified["info"].copy()
    data_dict_modified["info"]["path"] = str(tmp_path / "non_existent_info.csv")

    hub = LocalDataHub(data_dict_modified)
    hub.load_info_data()
    # 根据代码逻辑，当文件不存在时，会打印警告，并将 bar_df 置为空（info_df 保持 None）
    assert hub.info_df is None


def test_missing_daily_file(tmp_path, data_dict):
    """
    测试 daily 数据文件不存在时，load_bar_data 会打印警告信息，并使 bar_df 为空
    """
    data_dict_modified = data_dict.copy()
    data_dict_modified["bar"] = data_dict_modified["bar"].copy()
    data_dict_modified["bar"]["daily"] = [{
        "path": str(tmp_path / "non_existent_daily.csv"),
        "col_mapping": {
            "trade_date": "trade_date",
            "ts_code": "symbol",
            "open_price": "open"
        }
    }]
    hub = LocalDataHub(data_dict_modified)
    hub.load_bar_data()
    assert hub.bar_df.empty


def test_load_fundamental_data(data_dict):
    """
    测试 fundamental 数据加载方法（在 LocalDataHub 中该方法为 pass），因此 fundamental_df 应保持 None
    """
    hub = LocalDataHub(data_dict)
    hub.load_fundamental_data()
    assert hub.fundamental_df is None


def test_get_data_by_date_no_data(data_dict):
    """
    测试在未加载数据时调用 get_data_by_date 应抛出异常
    """
    hub = LocalDataHub(data_dict)
    with pytest.raises(ValueError):
        hub.get_data_by_date(pd.Timestamp("2021-01-01"))