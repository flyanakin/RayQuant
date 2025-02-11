import pandas as pd
import pytest

from core.strategy import MovingAverageStrategy
from core.datahub import LocalDataHub


@pytest.fixture
def hub():
    data_dict = {
        "bar": {
            "path": "../test/dataset/index_daily_中证500.csv",
            "col_mapping": {
                "symbol": "ts_code",
            },
        },
    }

    hub = LocalDataHub(data_dict)
    return hub


def test_demo_strategy(hub):
    ma_strategy = MovingAverageStrategy(
        hub=hub,
        indicator="close",
        ma_buy=720,
        ma_sell=180,
        buy_bias=-0.3,
        sell_bias=0.15,
    )
    signals = ma_strategy.generate_signals(current_time=pd.Timestamp("2023-01-01")).get()
    print(f"\n=====交易信号=====\n{signals}")
