import numpy as np


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