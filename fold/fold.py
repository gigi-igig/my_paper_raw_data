from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ✅ 非互動式後端，避免 X server crash
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re

# ---------------------------
# 1. 移除 outlier
# ---------------------------
def rm_outliner(df, n_sigma=3, col="flux_norm"):
    mean_val = df[col].mean()
    std_val = df[col].std()
    lower, upper = mean_val - n_sigma * std_val, mean_val + n_sigma * std_val
    mask = (df[col] >= lower) & (df[col] <= upper)
    return df[mask].reset_index(drop=True)

# ---------------------------
# 2. 單一 phase 欄位的 binning
# ---------------------------
def bin_phase_data(df, phase_col, period_days, bin_minutes=20):
    """將指定 phase 欄位依時間分組平均，回傳包含 phase_days 與 flux_avg 的 DataFrame"""
    bin_days = bin_minutes / 1440.0
    df["phase_days"] = df[phase_col] * period_days
    df["bin_id"] = (df["phase_days"] // bin_days).astype(int)

    binned = (
        df.groupby("bin_id")
        .agg(
            phase_days=("phase_days", "mean"),
            flux_avg=("flux_norm", "mean")
        )
        .reset_index(drop=True)
    )

    # 固定 phase 在 0~period_days
    binned["phase_days"] = binned["phase_days"] % period_days
    return binned

# ---------------------------
# 3. 產生三條曲線的 binned 結果（存不同資料夾）
# ---------------------------
def generate_all_binned_data(df, period_min, period_err_min, bin_minutes=20, binned_dir=None, basename=None):
    """對 phase_p-s, phase_p, phase_p+s 各自做 bin，並可存 CSV 到 binned_dir"""
    phase_cols = ["phase_p-s", "phase_p", "phase_p+s"]
    labels = ["p-s", "p", "p+s"]
    period_days_list = [
        (period_min - period_err_min) / 1440.0,
        period_min / 1440.0,
        (period_min + period_err_min) / 1440.0
    ]

    binned_results = {}

    for phase_col, label, period_days in zip(phase_cols, labels, period_days_list):
        binned = bin_phase_data(df, phase_col, period_days, bin_minutes=bin_minutes)
        binned_results[label] = binned

        if binned_dir is not None and basename is not None:
            os.makedirs(binned_dir, exist_ok=True)
            binned_file = os.path.join(binned_dir, f"{basename}_{label}_binned.csv")
            binned.to_csv(binned_file, index=False)

    return binned_results

# ---------------------------
# 4. 畫圖（增加 marker，確保三條線都顯示）
# ---------------------------
def plot_fold_binned_comparison(binned_results, filename, bin_minutes=20):
    plt.figure(figsize=(12, 5))
    labels = ["p-s", "p", "p+s"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linewidths = [1.2, 0.8, 0.5]  # 越後面越細

    # 依序畫線：p-s -> p -> p+s
    for label, color, lw in zip(labels, colors, linewidths):
        binned = binned_results[label]
        plt.plot(
            binned["phase_days"],
            binned["flux_avg"],
            color=color,
            linewidth=lw,
            alpha=0.85,
            label=label
        )

    plt.xlabel("Phase (days)")
    plt.ylabel(f"Normalized Flux ({bin_minutes}min avg)")
    plt.title(f"{os.path.basename(filename)} Folded Light Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=250)
    plt.close()

# ---------------------------
# 5. fold function（p-s, p, p+s）
# ---------------------------
def fold(df, period_min, period_err_min):
    period_days = period_min / 1440.0
    period_days_minus = (period_min - period_err_min) / 1440.0
    period_days_plus = (period_min + period_err_min) / 1440.0

    df["phase_p-s"] = (df["TIME"] / period_days_minus) % 1
    df["phase_p"] = (df["TIME"] / period_days) % 1
    df["phase_p+s"] = (df["TIME"] / period_days_plus) % 1
    return df

# ---------------------------
# 6. 主程式：不同週期比較
# ---------------------------
def fold_different_periods(detrend_dir, period_table_file, output_dir):
    detrend_versions = ["cubic_sample", "cubic_sample_BIC", "none", "only_BIC"]

    period_table = pd.read_csv(period_table_file)
    period_table["TIC_10"] = period_table["TIC ID"].astype(str).str.zfill(10)
    all_tics = set(period_table["TIC_10"])

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for detrend_way in detrend_versions:
        print(f"\n處理版本：{detrend_way}")
        detrend_path = os.path.join(detrend_dir, detrend_way)
        output_path = os.path.join(output_dir, detrend_way)
        os.makedirs(output_path, exist_ok=True)

        available_csvs = [
            f for f in os.listdir(detrend_path)
            if f.endswith(".csv")
            and ("750" in f or "None" in f)
            and detrend_way in f
        ]

        print("available_csvs", len(available_csvs))
        for fname in tqdm(available_csvs, desc=f"{detrend_way}"):
            match = re.match(r"(\d+)_interval(.*?)_" + detrend_way + r"\.csv", fname)
            if not match:
                print("not_match")
                continue

            tic, interval = match.groups()
            TIC_10 = tic[-10:]
            if TIC_10 not in all_tics:
                continue

            file_path = os.path.join(detrend_path, fname)
            df = pd.read_csv(file_path)
            df = df.dropna(subset=["TIME", "FLUX"])
            df.rename(columns={"FLUX": "flux_norm"}, inplace=True)

            row = period_table[period_table["TIC_10"] == TIC_10].iloc[0]
            period_min = row["Period (days)"] * 1440.0
            period_err_min = row["Period error"] * 1440.0

            df = rm_outliner(df, n_sigma=5)
            df = rm_outliner(df, n_sigma=3)
            df = fold(df, period_min, period_err_min)

            basename = os.path.splitext(fname)[0]

            # CSV 存在各自 detrend 的 binned 子資料夾
            binned_dir = os.path.join(output_path, "binned")
            binned_results = generate_all_binned_data(
                df, period_min, period_err_min, bin_minutes=20,
                binned_dir=binned_dir, basename=basename
            )

            # 圖存到統一 plot 資料夾
            plot_file = os.path.join(plot_dir, f"{basename}_folded_binned_days.png")
            plot_fold_binned_comparison(binned_results, plot_file, bin_minutes=20)

            # 原始 fold 資料
            df.to_csv(os.path.join(output_path, f"{basename}_folded.csv"), index=False)

    print("\n全部版本處理完成！")

# ---------------------------
# 7. 主程式：不同 detrend 比較
# ---------------------------
def fold_different_detrend(detrend_dir, period_table_file, output_dir):
    """
    對同一個 TIC 的不同 detrend_way 結果進行摺疊並畫在同一張圖上。
    只用 period（不含 ± 誤差），bin = 20 min。
    標題中會顯示折疊週期。
    """
    detrend_versions = ["cubic_sample", "none", "linear", "cubic_sample_BIC", "linear_BIC", "only_BIC"]
    colors = {
        "cubic_sample": "#1f77b4",      # 藍
        "cubic_sample_BIC": "#ff7f0e",  # 橘
        "none": "#2ca02c",              # 綠
        "only_BIC": "#9467bd",          # 紫
        "linear": "#d62728",            # 紅
        "linear_BIC": "#dd19c3",        # 棕
    }

    # 讀取週期表
    period_table = pd.read_csv(period_table_file)
    period_table["TIC_10"] = period_table["TIC ID"].astype(str).str.zfill(10)
    all_tics = set(period_table["TIC_10"])

    # 輸出資料夾
    plot_dir = os.path.join(output_dir, "compare_detrend")
    os.makedirs(plot_dir, exist_ok=True)

    # 找出每個 detrend_way 可用的檔案
    detrend_files = {}
    for detrend_way in detrend_versions:
        detrend_path = os.path.join(detrend_dir, detrend_way)
        if not os.path.exists(detrend_path):
            continue

        available_csvs = [
            f for f in os.listdir(detrend_path)
            if f.endswith(".csv") and ("750" in f or "None" in f)
        ]
        detrend_files[detrend_way] = available_csvs

    # 收集所有可能的 TIC ID
    all_tic_ids = sorted(list(all_tics))

    # 主迴圈：每個 TIC ID
    for TIC_10 in tqdm(all_tic_ids, desc="Compare detrend versions"):
        row = period_table[period_table["TIC_10"] == TIC_10]
        if row.empty:
            continue

        period_days = row["Period (days)"].values[0]

        plt.figure(figsize=(12, 5))
        plotted_any = False

        for detrend_way in detrend_versions:
            available_csvs = detrend_files.get(detrend_way, [])
            matched_files = [f for f in available_csvs if TIC_10 in f]

            if not matched_files:
                continue

            # 取第一個 interval 檔案（例如 interval750 或 None）
            fname = matched_files[0]
            detrend_path = os.path.join(detrend_dir, detrend_way, fname)
            if not os.path.exists(detrend_path):
                continue

            df = pd.read_csv(detrend_path)
            df = df.dropna(subset=["TIME", "FLUX"]).rename(columns={"FLUX": "flux_norm"})
            df = rm_outliner(df, n_sigma=5)
            df = rm_outliner(df, n_sigma=3)

            # 只用剛剛好的週期摺疊
            df["phase_p"] = (df["TIME"] / period_days) % 1

            # bin = 20 min
            binned = bin_phase_data(df, "phase_p", period_days, bin_minutes=20)

            plt.plot(
                binned["phase_days"],
                binned["flux_avg"],
                label=detrend_way,
                color=colors.get(detrend_way, "gray"),
                linewidth=0.7,
                alpha=0.9
            )
            plotted_any = True

        if plotted_any:
            plt.xlabel("Phase (days)")
            plt.ylabel("Normalized Flux (20min avg)")
            plt.title(f"TIC {TIC_10} — Different Detrend Comparison (P = {period_days:.5f} days)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plot_file = os.path.join(plot_dir, f"TIC{TIC_10}_compare_detrend.png")
            plt.savefig(plot_file, dpi=250)
        plt.close()

    print("\n所有 TIC 的 detrend 比較圖完成！")

# ---------------------------
# 8. 執行入口
# ---------------------------
if __name__ == "__main__":
    detrend_dir = r"/data2/gigicheng/data_21/TOI/preprocess"
    period_table_file = r"/data2/gigicheng/TOI_org/exofop_tess_sector21_filter.csv"
    output_dir = r"/data2/gigicheng/data_21/TOI/folded/"
    # fold_different_periods(detrend_dir, period_table_file, output_dir)
    fold_different_detrend(detrend_dir, period_table_file, output_dir)
