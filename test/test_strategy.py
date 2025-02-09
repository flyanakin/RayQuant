import pandas as pd
import pytest

from core.strategy import MovingAverageStrategy


@pytest.fixture
def index_daily():
    df = pd.read_csv('dataset/index_daily_中证500.csv')
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df.sort_values('trade_date', inplace=True)
    df.set_index('trade_date', inplace=True)
    return df


def test_demo_strategy(index_daily):
    ma_strategy = MovingAverageStrategy(data=index_daily)
    signals = ma_strategy.generate_signals(
        indicator='close',
        ma_buy=720,
        ma_sell=180,
        buy_bias=-0.3,
        sell_bias=0.15
    )
    print(f"\n=====交易信号=====\n{signals}")