# tool.py
import numpy as np
import pandas as pd
from mangol import mandel_agol as mangol
import os

# ---------------------------
# 1. 移除 outlier
# ---------------------------
def rm_outliner(df, n_sigma=3, col="FLUX"):
    mean_val = df[col].mean()
    std_val = df[col].std()
    lower, upper = mean_val - n_sigma * std_val, mean_val + n_sigma * std_val
    return df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)

# ---------------------------
# 2. 中間插值
# ---------------------------
def pad_to_npoints(binned, target_points=193):
    """
    將 binned DataFrame 補齊到 target_points。
    - 只在中間插值，不改變頭尾。
    - 假設 DataFrame 有 'phase' 與 'flux_avg' 欄位。
    """
    n_orig = len(binned)
    if n_orig >= target_points:
        return binned.copy()  # 已經足夠點，不處理

    # 原始 x 與 y（phase 與 flux）
    x_old = binned['fold_days'].to_numpy()
    y_old = binned['flux_avg'].to_numpy()

    # 保留頭尾
    x_new = np.linspace(x_old[0], x_old[-1], target_points)
    y_new = np.interp(x_new, x_old, y_old)

    return pd.DataFrame({'phase': x_new, 'flux_avg': y_new})

# -------------------------------
# 3. 幾何計算函式：投影距離 z(t)
# -------------------------------
def ha_z(t, t0, period, a_rs, iang=90):
    """計算每個時間點的 z(t) (投影中心距離 R*/R)
    t: 時間 (days)
    t0: transit 中心時間
    period: 週期 (days)
    a_rs: a/R*
    iang: 傾角 (deg)
    """
    phi = 2 * np.pi * (t - t0) / period
    z = a_rs * np.sqrt(np.sin(phi) ** 2 + (np.cos(np.radians(iang)) * np.cos(phi)) ** 2)
    z[(phi/(2*np.pi)-0.25)%1<0.5] = z.max()
    return z

def inject(period_day_begin, period_day_end):
    # 隨機生成 transit 參數
    period_day = np.random.uniform(period_day_begin, period_day_end)
    rp_rs = np.random.uniform(0.1, 0.5)
    a_rs = np.random.uniform(8, 20)
    iang = np.random.uniform(86, 90)
    # 隨機取 t0
    t0 = np.random.uniform(0.2*period_day, 0.8*period_day)

    return period_day, rp_rs, a_rs, iang, t0

def generate_signals(tic_ids, period_day_begin, period_day_end, seed=42):
    np.random.seed(seed)
    signals = {}
    for tic in tic_ids:
        # 生成 signal，但不做任何 detrend
        period_days, rp_rs, a_rs, iang, t0 = inject(period_day_begin, period_day_end)  # dummy t
        signals[tic] = {
            "period_days": period_days,
            "rp_rs": rp_rs,
            "a_rs": a_rs,
            "iang": iang,
            "t0": t0
        }
    return signals