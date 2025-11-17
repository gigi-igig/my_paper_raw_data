#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from tqdm import tqdm
from tool import extract_target_id, dropnan_sort, normalize_by_group
from detrend import detrend_flux
from config import detrend_methods, interval_list, no_interval_detrend, cut_tail_len, data_size
from datetime import datetime


# -------------------------------
# II. 主程式
# -------------------------------
def main(dir_path, input_csv_dir, sample_df, pic_batch=50, detrend_methods=[], cut_tail_len=0):

    tic_list = sample_df["TIC ID"].astype(str).str.zfill(10).tolist()
    csv_files = [f"{tic}_group_norm.csv" for tic in tic_list]

    results = []

    # 處理每一筆 sample
    for idx, csv_name in enumerate(tqdm(csv_files)):

        csv_path = os.path.join(input_csv_dir, csv_name)

        if not os.path.exists(csv_path):
            print(f"⚠ 找不到檔案：{csv_path}，跳過。")
            continue

        # 讀取該 TIC 的 group_norm CSV
        df = pd.read_csv(csv_path)

        time = df["TIME"].values
        flux_norm = df["FLUX"].values
        target_id = csv_name.split("_")[0]  # 例如 12345678_group_norm

        # detrend
        flux_detrended_dict = {}
        _s_bic_cache = {}

        for detrend_method in detrend_methods:
            intervals = [None] if detrend_method in no_interval_detrend else interval_list

            for interval in intervals:

                f_detrended, _s_bic_cache = detrend_flux(
                    detrend_method, time, flux_norm,
                    interval, _s_bic_cache=_s_bic_cache
                )

                f_detrended = f_detrended - np.mean(f_detrended)

                # tail cut
                if cut_tail_len > 0 and len(f_detrended) > 2 * cut_tail_len:
                    f_detrended_cut = f_detrended[cut_tail_len:-cut_tail_len]
                    time_cut = time[cut_tail_len:-cut_tail_len]
                else:
                    f_detrended_cut = f_detrended
                    time_cut = time

                key_name = f"{detrend_method}_int{interval}"
                flux_detrended_dict[key_name] = f_detrended_cut

                # 儲存 CSV
                combo_dir = f"/data2/gigicheng/data_21/raw_data/preprocess/{dir_path}/{detrend_method}"
                os.makedirs(combo_dir, exist_ok=True)

                out_csv_path = os.path.join(
                    combo_dir,
                    f"{target_id}_interval{interval}_{detrend_method}_cut{cut_tail_len}.csv"
                )

                pd.DataFrame({"TIME": time_cut, "FLUX": f_detrended_cut}).to_csv(out_csv_path, index=False)

                results.append({
                    "TIC ID": target_id,
                    "flux_mean": np.nanmean(f_detrended_cut),
                    "flux_min": np.nanmin(f_detrended_cut),
                    "flux_max": np.nanmax(f_detrended_cut),
                    "flux_std": np.nanstd(f_detrended_cut),
                    "detrend_way": detrend_method,
                    "detrend_SNR": (np.nanmean(f_detrended_cut) - np.nanmin(f_detrended_cut)) / np.nanstd(f_detrended_cut),
                    "interval": interval,
                    "cut_tail": cut_tail_len > 0,
                    "points_num": len(f_detrended_cut)
                })

    # summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(f"/data2/gigicheng/data_21/raw_data/preprocess/{dir_path}/preprocess_summary_{dir_path}.csv", index=False)
    print("\n✓ 已處理完 sample_df 的所有 TIC。")


# -------------------------------
# III. 主程式入口
# -------------------------------
if __name__ == "__main__":

    dir_path = "mean_to_0"
    input_df = pd.read_csv("/data2/gigicheng/data_21/raw_data/group_norm/group_summary_exp1.csv")

    filtered_df = input_df[input_df["org_flux_std"] < 0.001]
    sample_df = filtered_df.sample(n=data_size, random_state=42)

    input_csv_dir = "/data2/gigicheng/data_21/raw_data/group_norm/exp1"

    start_time = datetime.now()
    print(f"開始時間: {start_time}")

    main(dir_path, input_csv_dir, sample_df, pic_batch=1000, detrend_methods=detrend_methods, cut_tail_len=cut_tail_len)

    end_time = datetime.now()
    print(f"完成時間: {end_time}")
    print(f"總共耗時: {end_time - start_time}")
