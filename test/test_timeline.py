import pandas as pd
import pytest
from core.timeline import Timeline


def test_empty_bar_df():
    # 构造空的 DataFrame 测试应抛出异常
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="无数据输入"):
        tl = Timeline(empty_df)
        tl.get_main_timeline()


def test_main_timeline_union(capsys):
    # 构造测试数据：
    # symbol 'A' 有两个日期，symbol 'B' 只有一个日期
    dates_A = pd.to_datetime(['2020-01-01', '2020-01-02'])
    dates_B = pd.to_datetime(['2020-01-01'])

    # 为 symbol 'A' 构造数据
    index_A = pd.MultiIndex.from_product([dates_A, ['A']], names=['trade_date', 'symbol'])
    df_A = pd.DataFrame({'close': [100, 101]}, index=index_A)

    # 为 symbol 'B' 构造数据
    index_B = pd.MultiIndex.from_product([dates_B, ['B']], names=['trade_date', 'symbol'])
    df_B = pd.DataFrame({'close': [200]}, index=index_B)

    # 合并数据
    df = pd.concat([df_A, df_B])

    # 初始化 Timeline，使用默认的 'union' 策略
    tl = Timeline(df)
    main_timeline = tl.get_main_timeline()

    # 立即捕获 get_main_timeline() 的打印输出
    captured = capsys.readouterr().out
    assert "symbol 'B'" in captured, f"捕获输出中没有 'symbol 'B''，捕获内容为：{captured}"
    assert "2020-01-02" in captured, f"捕获输出中没有 '2020-01-02'，捕获内容为：{captured}"

    # 预期主时间轴为日期并集：2020-01-01 和 2020-01-02
    expected_timeline = pd.DatetimeIndex(['2020-01-01', '2020-01-02'], name='trade_date')
    pd.testing.assert_index_equal(main_timeline.sort_values(), expected_timeline,
                                  obj="主时间轴不符合预期")


def test_timeseries_iterator():
    # 构造测试数据与上个测试类似
    dates_A = pd.to_datetime(['2020-01-01', '2020-01-02'])
    dates_B = pd.to_datetime(['2020-01-01'])
    index_A = pd.MultiIndex.from_product([dates_A, ['A']], names=['trade_date', 'symbol'])
    df_A = pd.DataFrame({'close': [100, 101]}, index=index_A)
    index_B = pd.MultiIndex.from_product([dates_B, ['B']], names=['trade_date', 'symbol'])
    df_B = pd.DataFrame({'close': [200]}, index=index_B)
    df = pd.concat([df_A, df_B])

    tl = Timeline(df)
    # 从迭代器中获取所有日期
    iterated_dates = list(tl.timeseries_iterator())
    expected_dates = list(pd.to_datetime(['2020-01-01', '2020-01-02']))
    assert iterated_dates == expected_dates, "迭代器返回的日期不符合预期"