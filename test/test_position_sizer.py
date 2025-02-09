import pytest
import pandas as pd
from core.position_sizer import PositionSizer


def test_position_sizer():
    # 构造示例信号: 假设有两个交易日, 两次信号
    data = {"asset": ["600000.SH", "688000.SH"], "signal": ["BUY", "SELL"]}
    # 两行数据的index当作日期
    signals_df = pd.DataFrame(data, index=["2023-01-02", "2023-01-03"])

    ps = PositionSizer()
    orders = ps.transform_signals_to_orders(signals_df)
    # 预期生成两条订单
    assert len(orders) == 2

    # 第1条
    o1 = orders.iloc[0]
    assert o1["date"] == "2023-01-02"
    assert o1["asset"] == "600000.SH"
    assert o1["side"] == "BUY"
    # 600000.SH -> 普通A股, lot_size=100
    assert o1["quantity"] == 100

    # 第2条
    o2 = orders.iloc[1]
    assert o2["date"] == "2023-01-03"
    assert o2["asset"] == "688000.SH"
    assert o2["side"] == "SELL"
    # 688开头 -> 科创板, lot_size=200(示例)
    assert o2["quantity"] == 200
