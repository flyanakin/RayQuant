import pandas as pd
from utils.indicators import annual_return, annual_volatility, drawdown, compute_future_return, kelly_criterion
from typing import Tuple, Dict, Optional
import re


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


def _group_cnt(total_count,
               group_cnt_range: Tuple[int, int],
               min_sample_cnt: int = 30
               ) -> int:
    """
    计算分组数量
    :param total_count: 总样本数量
    :param group_cnt_range: 分组数量范围
    :param min_sample_cnt: 每组最少的样本数
    :return:
    """
    num_groups = total_count // min_sample_cnt
    num_groups = min(max(num_groups, group_cnt_range[0]), group_cnt_range[1])
    return num_groups


def group_data(
        df: pd.DataFrame,
        group_by: str = "bias",
        group_cnt_range: Tuple[int, int] = (20, 80),
) -> Dict[str, pd.DataFrame]:
    """
    该函数会：
      根据指定字段（bias 或 indicator）进行等分分组，分组数根据样本总数计算（每组至少 30 个点），并限定在 group_cnt_range 范围内；
      分组时采用 pd.qcut，生成的分组标签为区间字符串，更具业务语义。
    :param df: index为日期，必须包含price列用于回测计算
                - price：用于计算未来收益
                - 当 group_by 为 bias 时，至少需要包含一个 ma{N}_bias 列
                - 当 group_by 为 indicator 时，需包含 indicator 列
    :param group_by:
                - 当 group_by == "bias" 时：基于 ma{N}_bias 列进行分组
                - 当 group_by == "indicator" 时：基于 indicator 列分组
    :param group_cnt_range: 分组数量的范围
    :return
        Dict: bias区间为key，value为对应的数据集 {"(-36.952, -34.26]": pd.Dataframe}
              indicator区间为key，value为对应的数据集 {"(2, 2.5]": pd.Dataframe}
    """
    if group_by == "bias":
        # 获取 ma{N}_bias 列
        bias_cols = [col for col in df.columns if re.match(r'ma\d+_bias', col)]
        if not bias_cols:
            raise ValueError("输入 DataFrame 需要至少包含一个 `ma{N}_bias` 列")
        # 选择最后一个 bias 列
        bias_col = bias_cols[-1]
        group_field = bias_col
    elif group_by == "indicator":
        # 获取 indicator 列
        group_field = 'indicator'
    else:
        raise ValueError(f"Invalid group_by value: {group_by}")

    # 根据总样本数决定分组数（每组至少 30 个点），并限定在 group_cnt_range 范围内
    total_count = len(df)
    num_groups = _group_cnt(total_count, group_cnt_range, min_sample_cnt=30)

    groups = pd.qcut(df[group_field], num_groups, duplicates='drop')
    df['group_label'] = groups

    # 构建详细数据字典，按照 Interval 的左边界从小到大排序，并转换为字符串作为 key
    group_details: Dict[str, pd.DataFrame] = {
        str(key): group.copy()
        for key, group in sorted(df.groupby('group_label', observed=False), key=lambda kv: kv[0].left)
    }

    return group_details


