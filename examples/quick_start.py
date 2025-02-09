import pandas as pd
from core.strategy import MovingAverageStrategy
from core.position_sizer import PositionSizer
from core.broker import Broker
from core.portfolio import Portfolio
from core.backtester import BackTester


def main():
    # 1) 读取数据(示例：中证500的日行情)
    df = pd.read_csv("../test/dataset/index_daily_中证500.csv")
    # 假设 CSV 中有 'ts_code', 'trade_date', 'close' 等列
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df.set_index("trade_date", inplace=True)
    df.sort_index(inplace=True)

    # 假设 CSV 中至少包含 ['ts_code','close'] 列
    # 并且 'ts_code' 代表标的名(可能都是 '000905.SH' 这样的)

    # 2) 构造公共模块
    position_sizer = PositionSizer()
    broker = Broker()
    portfolio = Portfolio(initial_cash=1000000000)

    # 3) 试用 MovingAverageStrategy
    ma_strategy = MovingAverageStrategy(
        data=df,
        indicator="close",
        asset_col="ts_code",
        ma_buy=720,
        ma_sell=180,
        buy_bias=-0.3,
        sell_bias=0.15,
    )
    backtester = BackTester(ma_strategy, position_sizer, broker, portfolio, df)
    result_ma = backtester.run_backtest()
    print("=== MA 策略结果 ===")
    print(result_ma.tail(10))

    # 4) 查看回测结果
    print("\n===== 回测每日净值 =====")
    print(result_ma.head(10))
    print("...")
    print(result_ma.tail(10))

    final_value = result_ma["total_value"].iloc[-1]
    print(f"\n回测结束时组合总价值: {final_value}")
    print("\n===== 最终持仓信息 =====")
    print(portfolio.asset)
    print("\n===== 交易日志 =====")
    print(portfolio.trade_log)


if __name__ == "__main__":
    main()
