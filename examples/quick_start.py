from core.datahub import LocalDataHub
from core.strategy import MovingAverageStrategy
from core.position_manager import EqualWeightPositionManager
from core.portfolio import Portfolio
from core.backtester import BackTester


def main():
    data_dict = {
        "bar": {
            "path": "../test/dataset/index_daily.csv",
            "col_mapping": {
                "symbol": "ts_code",
            },
        },
    }

    hub = LocalDataHub(data_dict)

    # 2) 构造公共模块
    position_manager = EqualWeightPositionManager()
    portfolio = Portfolio(initial_cash=100000000)

    # 3) 试用 MovingAverageStrategy
    ma_strategy = MovingAverageStrategy(
        hub=hub,
        indicator="close",
        ma_buy=720,
        ma_sell=720,
        buy_bias=-0.3,
        sell_bias=0.3,
    )
    backtester = BackTester(
        data=hub,
        strategy=ma_strategy,
        position_manager=position_manager,
        portfolio=portfolio,
    )
    scope = backtester.run_backtest()
    print("=== MA 策略结果 ===")
    print(scope.print_metrics())

    # 4) 查看回测结果
    print("\n===== 回测每日净值 =====")
    print(scope.results.head(10))
    print("...")
    print(scope.results.tail(10))

    final_value = scope.results["total_value"].iloc[-1]
    print(f"\n回测结束时组合总价值: {final_value}")
    print("\n===== 最终持仓信息 =====")
    print(portfolio.asset)
    print("\n===== 交易日志 =====")
    print(portfolio.trade_log)
    scope.plot_results()


if __name__ == "__main__":
    main()
