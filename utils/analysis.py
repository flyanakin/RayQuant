import pandas as pd
from utils.indicators import annual_return, annual_volatility, drawdown, compute_future_return, kelly_criterion
from utils.technical_process import calculate_moving_average_bias
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


def _compute_group_metrics(
        group_detail: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算单个分组的统计指标
    :param group_detail: index为日期，需要包含以下字段
                - price: 交易价格
                - future_return: 未来的收益百分比，小数
                - end_date：卖出的日期
                - end_price: 卖出的价格
    :return:
          stats_df：包含字段
                - group_label: 分组的标识
                - winning_rate: 该分组胜率，future_return为正的占比
                - winning_reward: 胜局净收益
                - losing_rate: 该分组负率
                - losing_reward: 负局净亏损
                - f: 最佳投注比，由kelly_criterion计算得出
                - sample_cnt: 样本数
                - avg_return: 该分组的平均收益率
    """
    grouped = group_detail.groupby('group_label', observed=False)['future_return']
    stats_df = grouped.agg(
        winning_rate=lambda x: (x > 0).mean(),
        winning_reward=lambda x: x[x > 0].mean() if (x > 0).any() else 0,
        sample_cnt='count',
        avg_return='mean',
        losing_reward=lambda x: x[x < 0].mean() if (x < 0).any() else 0
    ).reset_index()
    stats_df['losing_rate'] = 1 - stats_df['winning_rate']
    # 计算 Kelly 指标
    stats_df['f'] = stats_df.apply(
        lambda row: kelly_criterion(row['winning_rate'],
                                    row['winning_reward'],
                                    row['losing_reward'],
                                    row['losing_rate']),
        axis=1
    )
    stats_df = stats_df[
        ['group_label', 'winning_rate', 'winning_reward', 'losing_rate', 'losing_reward', 'f', 'sample_cnt',
         'avg_return']]
    return stats_df


def monotonic_group_discovery(
        df: pd.DataFrame,
        min_groups: int = 3,
        direction: str = 'long'
) -> tuple[str, dict]:
    """
    单调性发现
    做多：行号越小（group_name越小），胜率越高，必须严格单调，只从第一行即胜率最高的开始向后检验，判断有连续多少行是符合胜率单调递减的
    做空：行号越大（group_name越大），胜率越小，必须严格单调，只从最后一行即胜率最低的开始向前检验，判断有连续多少行是符合胜率单调递增的
    :param df: 用户输入的dataframe，为分组统计结果，group_label列的区间会随着行数增加而增加，winning_rate表示该组的统计胜率
    :param min_groups: 至少要连续多少行（组），才能认为是具有单调性的
    :param direction: long/short，每次只判断一个方向
    :return:
        output_str: 打印信息的字符串 打印判断信息 {做多/做空} 具有单调性，有{}组数据，最低胜率{}%，最高胜率{}%, 样本数{}
        metrics: 各种指标的字典
    """
    if direction not in ['long', 'short']:
        raise ValueError("direction 参数必须为 'long' 或 'short'")

    monotonic_rows = []
    metrics = {}
    output_str = ''

    if direction == 'long':
        # 从第一行开始检查：必须满足当前行的胜率严格大于下一行的胜率
        monotonic_rows.append(df.iloc[0])
        for i in range(1, len(df)):
            # 当前行的胜率大于下一行的胜率，才加入单调序列
            if df.iloc[i - 1]['winning_rate'] > df.iloc[i]['winning_rate']:
                monotonic_rows.append(df.iloc[i])
            else:
                break
        if len(monotonic_rows) >= min_groups:
            metrics['group_cnt'] = len(monotonic_rows)
            metrics['highest_rate'] = monotonic_rows[0]['winning_rate']  # 第一行胜率最高
            metrics['interval'] = monotonic_rows[-1]['group_label']
            nums = re.findall(r"[-+]?\d*\.\d+", metrics['interval'])  # 匹配带符号的小数
            left, right = map(float, nums)
            metrics['bias_high_bound'] = right
            metrics['lowest_rate'] = monotonic_rows[-1]['winning_rate']  # 最后一行胜率最低
            metrics['sample_cnt'] = monotonic_rows[-1]['sample_cnt']  # 取最低胜率行的样本数
            metrics['f'] = monotonic_rows[-1]['f']
            output_str = "偏离均线超过{:.2f}%，做多具有单调性，有{}组数据，最低胜率{:.2f}%，最高胜率{:.2f}%，最佳投注比{:.2f}%，样本数{}，所在分组{}".format(
                metrics['bias_high_bound'] * 100,  metrics['group_cnt'], metrics['lowest_rate'] * 100,
                metrics['highest_rate'] * 100, metrics['f'] * 100, metrics['sample_cnt'], metrics['interval'])
            return output_str, metrics
        else:
            return output_str, metrics

    elif direction == 'short':
        # 做空：从最后一行开始向上检查，要求前一行的胜率必须严格小于后一行的胜率
        monotonic_rows.append(df.iloc[-1])
        for i in range(len(df) - 2, -1, -1):
            if df.iloc[i]['winning_rate'] < df.iloc[i + 1]['winning_rate']:
                # 插入到序列开头，以保证最终顺序与df一致
                monotonic_rows.insert(0, df.iloc[i])
            else:
                break
        if len(monotonic_rows) >= min_groups:
            metrics['group_cnt'] = len(monotonic_rows)
            metrics['highest_rate'] = 1 - monotonic_rows[0]['winning_rate']  # 单调区间中最高的胜率（靠近df上部）
            metrics['interval'] = monotonic_rows[-1]['group_label']
            nums = re.findall(r"[-+]?\d*\.\d+", metrics['interval'])  # 匹配带符号的小数
            left, right = map(float, nums)
            metrics['bias_low_bound'] = left
            metrics['lowest_rate'] = 1 - monotonic_rows[-1]['winning_rate']  # 最低的胜率
            metrics['sample_cnt'] = monotonic_rows[-1]['sample_cnt']
            metrics['f'] = monotonic_rows[-1]['f']
            output_str = "偏离均线超过{:.2f}%，做空具有单调性，有{}组数据，最低胜率{:.2f}%，最高胜率{:.2f}%，最佳投注比{:.2f}%，样本数{}，所在分组{}".format(
                metrics['bias_low_bound'] * 100, metrics['group_cnt'], metrics['lowest_rate'] * 100,
                metrics['highest_rate'] * 100, metrics['f'] * 100, metrics['sample_cnt'], metrics['interval'])
            return output_str, metrics
        else:
            return output_str, metrics


def indicator_ma_discovery(
        df: pd.DataFrame,
        indicators: list[str],
        mas: list[int],
) -> tuple[dict, dict]:
    """
    寻找最优的因子和均线
    :param df: index trade_date列为交易日期,
                - price: 交易价格
                - indicators列都必须在df中
    :param indicators: 指标序列，如['close', 'vol']
    :param mas: 均线序列 [5, 10, 20, ]
    :return:
    """
    group_ma_dfs = {}
    win_rate_df = {}

    for indicator in indicators:
        # 对每个指标，先初始化一个内层字典
        group_ma_dfs[indicator] = {}
        win_rate_df[indicator] = {}
        df_slice = df[['trade_date', 'price', indicator]].copy().set_index('trade_date')
        df_bias = calculate_moving_average_bias(
            df=df_slice,
            indicator_col=indicator,
            mas=mas)

        for ma in mas:
            df_bias_slice = df_bias[['indicator', 'price', 'ma' + str(ma), 'ma' + str(ma) + '_bias']].copy()
            df_bias_slice = compute_future_return(df=df_bias_slice, future_days=ma, direction='long')
            group_dfs = group_data(
                df=df_bias_slice,
                group_by="bias",
                group_cnt_range=(20, 60),
            )
            # 计算各组的指标
            stats = _compute_group_metrics(df_bias_slice)
            stats['lower_bound'] = stats['group_label'].apply(lambda x: x.left)
            stats = stats.sort_values(by='lower_bound')
            # 将 Interval 转换为字符串展示，并去掉辅助排序列
            stats['group_label'] = stats['group_label'].astype(str)
            stats.drop(columns=['lower_bound'], inplace=True)

            # 发现单调性
            monotonic, metrics = monotonic_group_discovery(stats, 3, 'long')
            output_str = f"{indicator}" + f"MA{ma}" + monotonic
            group_ma_dfs[indicator][ma] = group_dfs
            win_rate_df[indicator][ma] = stats
            if monotonic != "":
                print(output_str)

    return group_ma_dfs, win_rate_df
