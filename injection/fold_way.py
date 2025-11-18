import matplotlib
matplotlib.use("Agg")  # 遠端非互動式後端
from tool import rm_outliner
import numpy as np
import pandas as pd

# ---------------------------
# 單一光曲線摺疊處理
# ---------------------------

# 用 bin_minutes 設定 
def bin_lightcurve(df_lc, period_days, interval=None, bin_minutes = 20):
    #df_lc = df_lc.dropna(subset=["TIME", "FLUX"])
    df_lc = rm_outliner(df_lc, 5)
    df_lc = rm_outliner(df_lc, 3)
    df_lc["fold_mins"] = (df_lc["TIME"] * 1440) % (period_days * 1440)
    df_lc["bin_id"] = (df_lc["fold_mins"] // bin_minutes).astype(int)
    binned = (
        df_lc.groupby("bin_id")
        .agg(
            fold_mins=("fold_mins", "mean"),
            flux_avg=("FLUX", "mean"),
            count=("FLUX", "count")
        )
        .reset_index(drop=True)
    )
    binned["fold_days"] = binned["fold_mins"] / 1440.0
    if interval is not None:
        binned["interval"] = interval

    # 依相位排序，避免繪圖時亂線
    binned = binned.sort_values("fold_days").reset_index(drop=True)
    # 尾端誤差極小值處理
    if binned.iloc[-1]["count"] < binned.iloc[-2]["count"]*0.66:
        binned.loc[binned.index[-1], "flux_avg"] = 1
    
    return binned

# GPFC
def gpfc_bin(df_lc, period_days, interval=None, bin_minutes=20):
    """
    使用高解析中介分箱 (GPFC) 的光曲線摺疊方法，
    最終輸出格式與 bin_lightcurve() 相同。
    """
    n_intermediate = 4096
    n_final_bins = 256

    # Step 0: 前處理
    df_lc = rm_outliner(df_lc, 5)
    df_lc = rm_outliner(df_lc, 3)

    # Step 1: 計算相位 (分鐘)
    df_lc["phase_mins"] = (df_lc["TIME"] * 1440) % (period_days * 1440)

    # Step 2: 中介分箱
    phase_mins = df_lc["phase_mins"].to_numpy()
    flux = df_lc["FLUX"].to_numpy()

    bin_edges = np.linspace(0, period_days * 1440, n_intermediate + 1)
    digitized = np.clip(np.digitize(phase_mins, bin_edges) - 1, 0, n_intermediate - 1)

    # Step 3: 累加與計數
    sums = np.bincount(digitized, weights=flux, minlength=n_intermediate)
    counts = np.bincount(digitized, minlength=n_intermediate)
    means = sums / np.maximum(counts, 1)

    # Step 4: 合併至最終 bins
    bins_per_final = n_intermediate // n_final_bins
    flux_avg = (
        means[:bins_per_final * n_final_bins]
        .reshape(n_final_bins, bins_per_final)
        .mean(axis=1)
    )

    '''
    # 同時計算 count 與 std (以原始點對應分配)
    count_final = (
        counts[:bins_per_final * n_final_bins]
        .reshape(n_final_bins, bins_per_final)
        .sum(axis=1)
    )
    '''

    # Step 5: 計算每個最終 bin 的中心 phase
    phase_mins_bins = np.linspace(0, period_days * 1440, n_final_bins, endpoint=False)
    phase_days = phase_mins_bins / 1440.0

    # Step 6: 標準化 (保持平均, std=1)
    mean_flux = np.mean(flux_avg)
    std_flux = np.std(flux_avg)
    flux_avg = (flux_avg - mean_flux) / np.maximum(std_flux, 1e-12) + mean_flux

    # Step 7: 組成 DataFrame (與 bin_lightcurve 格式一致)
    binned = pd.DataFrame({
        "phase_mins": phase_mins_bins,
        "flux_avg": flux_avg,
        "phase_days": phase_days
    })

    if interval is not None:
        binned["interval"] = interval

    return binned
