import pandas as pd
from utils.indicators import annual_return, annual_volatility, drawdown


def risk_and_return(
        df: pd.DataFrame,
        interval_months: int = 3,
        benchmarks: list[str] = None,
) -> dict:
    """
    分析数据的风险和投资回报
    :param df: 时间序列数据，index必须是日期，类型pd.Timestamp,
               每一列代表一个标的(symbol)的每日价值，列名与标的同名
    :param interval_months: 分析的周期，用于确定区间回撤，单位是月，通常取3、6、12
    :param benchmarks: benchmark的标识 如:['spx','000300.SH']，支持多个标的
    :return: 返回一个字典，包含一下key和value
        'result_df': 一个df包含所有标的的评估指标(年化收益率、年化波动率、最大回撤、最大回撤区间)
                - annual_return 年化收益率
                - annual_volatility 年化波动率
                - max_drawdown 最大回撤：百分比
                - max_drawdown_interval 最大回撤对应区间：日期区间
        'return_lines': 一个df包含所有标的的归一化收益，index为日期，所有标的在起始日的值为1，根据每日的收益率更新价值
        'drawdown_df':  回撤的序列，按每日来计算回撤的百分比
    """
    # 拷贝并保证时间序列按日期升序排列，防止影响原数据
    data = df.copy()
    data.sort_index(inplace=True)

    # 归一化收益（起始日均为1）
    return_lines = data.divide(data.iloc[0])

    # 保存各标的的指标
    metrics = []
    # 保存每日回撤序列
    drawdown_dict = {}

    # 遍历每个标的，计算指标
    for symbol in return_lines.columns:
        series = return_lines[symbol].copy()

        # 计算区间天数（用起止日期差）
        total_days = (series.index[-1] - series.index[0]).days

        # 年化收益率，利用归一化后的起始和结束价值，传入series构成的DataFrame
        ann_ret = annual_return(
            start_value=series.iloc[0],
            end_value=series.iloc[-1],
            total_days=total_days,
        )

        # 计算每日收益率（向量化，首日收益率无法计算则删除）
        daily_returns = series.pct_change().dropna()
        ann_vol = annual_volatility(daily_returns.to_frame(name='value'))

        # 计算区间最大回撤及对应区间，drawdown返回 (interval_drawdowns_df, (max_dd, (start_date, end_date)))
        dd_res = drawdown(series.to_frame(name='value'), interval_months)
        _, (max_dd, dd_interval) = dd_res

        # 计算每日回撤序列：当前值相对于历史最高值的百分比变化
        daily_dd = (series / series.cummax() - 1) * 100
        drawdown_dict[symbol] = daily_dd

        # 将计算指标存入列表
        metrics.append({
            'symbol': symbol,
            'annual_return': round(ann_ret, 4),
            'annual_volatility': round(ann_vol, 4),
            'max_drawdown': round(max_dd, 4),
            'max_drawdown_interval': dd_interval
        })

    # 将各标的指标整理成DataFrame，index为标的名称
    result_df = pd.DataFrame(metrics).set_index('symbol')
    # 将每日回撤字典构成DataFrame，index与归一化收益保持一致
    drawdown_df = pd.DataFrame(drawdown_dict, index=return_lines.index)

    return {
        'result_df': result_df,
        'return_lines': return_lines,
        'drawdown_df': drawdown_df
    }
