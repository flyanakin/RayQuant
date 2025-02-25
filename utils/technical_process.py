import numpy as np
import pandas as pd


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """
    利用累计和计算简单移动平均（要求 arr 中无缺失值）。
    对于窗口不足 window 的前几项，返回 np.nan 保持与 pandas 行为一致。
    """
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=np.float64)
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    # 为了对齐原始数组，前 window-1 个值置为 np.nan
    return np.concatenate((np.full(window - 1, np.nan), ma))


def calculate_moving_average_bias(
        df: pd.DataFrame,
        mas: list[int],
        indicator_col: str = 'close',
) -> pd.DataFrame:
    """
    对一个df计算指标的移动均线 (MA) 和 Bias（偏离度）
    :param df: 默认 index 为时间，price列为当日交易价格
    :param indicator_col: 用于计算ma的指标列名
    :param mas: 需要计算的 MA 周期列表，如 [5, 10, 20, 60]
    :return: 结果 DataFrame，包含原始价格、指标以及对应的 'ma{N}'、'ma{N}_bias' 等列
    """
    result_df = df.copy()
    result_df['indicator'] = result_df[indicator_col]

    for ma in mas:
        ma_col = f'ma{ma}'
        bias_col = f'{ma_col}_bias'
        result_df[ma_col] = result_df['indicator'].rolling(ma, min_periods=1).mean()
        result_df[bias_col] = round((result_df['indicator'] - result_df[ma_col]) / result_df[ma_col], 4)
    result_df.drop(columns=[indicator_col], inplace=True)

    return result_df
