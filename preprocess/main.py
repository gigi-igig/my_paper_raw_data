#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")  # 非互動式後端
import pandas as pd
from tqdm import tqdm
from tool import extract_target_id, dropnan_sort, normalize_by_group
from detrend import detrend_flux
from draw import plot_preprocess_by_interval, plot_preprocess_by_interval_html
from config import detrend_methods, interval_list, no_interval_detrend, cut_tail_len
from datetime import datetime

# -------------------------------
# II. 主程式
# -------------------------------
def main(dir_path, data_dir, data_size=1000, pic_batch=50, detrend_methods=[], cut_tail_len=0):
    """
    cut_tail_len: int, 尾端要裁掉的點數，0 表示不裁
    """
    _s_bic_cache = {}
    plot_dir = f"/data2/gigicheng/data_21/TOI/preprocess/{dir_path}/plot"
    os.makedirs(plot_dir, exist_ok=True)
    fits_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".fits")])
    if data_size != -1:
        fits_files = fits_files[:data_size]
    
    results = []

    for idx, fname in enumerate(tqdm(fits_files)):
        file_path = os.path.join(data_dir, fname)
        target_id = extract_target_id(fname)
        with fits.open(file_path) as hdul:
            time, flux = hdul[1].data['TIME'], hdul[1].data['PDCSAP_FLUX']
        time, flux = dropnan_sort(time, flux)
        time = time - np.nanmin(time)
        if len(time) < 2880:
            continue
        time, flux_norm = normalize_by_group(time, flux, dt_threshold=400/1440, flux_gap_sigma=None)

        flux_detrended_dict = {}
        for detrend_method in detrend_methods:
            intervals = [None] if detrend_method in no_interval_detrend else interval_list

            for interval in intervals:
                f_detrended, _s_bic_cache = detrend_flux(detrend_method, time, flux_norm, interval, _s_bic_cache=_s_bic_cache)
                # mean移動到0
                f_detrended = f_detrended - np.mean(f_detrended)
                # 安全裁剪 tail，time 保留原長度，輸出與統計用裁剪後的資料
                if cut_tail_len > 0 and len(f_detrended) > 2*cut_tail_len:
                    f_detrended_cut = f_detrended[cut_tail_len:-cut_tail_len]
                    time_cut = time[cut_tail_len:-cut_tail_len]
                else:
                    f_detrended_cut = f_detrended
                    time_cut = time

                key_name = f"{detrend_method}_int{interval}"
                flux_detrended_dict[key_name] = f_detrended_cut

                # 儲存 CSV
                combo_dir = f"/data2/gigicheng/data_21/TOI/preprocess/{dir_path}/{detrend_method}"
                os.makedirs(combo_dir, exist_ok=True)
                csv_path = os.path.join(combo_dir, f"{target_id}_interval{interval}_{detrend_method}_cut{cut_tail_len}.csv")
                pd.DataFrame({"TIME": time_cut, "FLUX": f_detrended_cut}).to_csv(csv_path, index=False)

                results.append({
                    "TIC ID": target_id,
                    "flux_mean": np.nanmean(f_detrended_cut),
                    "flux_min": np.nanmin(f_detrended_cut),
                    "flux_max": np.nanmax(f_detrended_cut),
                    "flux_std": np.nanstd(f_detrended_cut),
                    "detrend_way": detrend_method,
                    "detrend_SNR": (np.nanmean(f_detrended_cut)-np.nanmin(f_detrended_cut))/np.nanstd(f_detrended_cut),
                    "interval": interval,
                    "cut_tail": cut_tail_len > 0,
                    "points_num": len(f_detrended_cut)
                })

        # 畫圖
        if idx % pic_batch == 0:
            for interval in interval_list:
                plot_preprocess_by_interval(time_cut, flux_detrended_dict, plot_dir, target_id, interval)
                plot_preprocess_by_interval_html(time_cut, flux_detrended_dict, plot_dir, target_id, interval)

    # 儲存 summary CSV
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(f"/data2/gigicheng/data_21/TOI/preprocess/{dir_path}/preprocess_summary_{dir_path}.csv", index=False)
    print("\n所有 FITS 檔案前處理完成！統計 CSV 已生成。")


if __name__ == "__main__":
    dir_path = "mean_to_0"
    data_dir = "/data2/gigicheng/TOI_org/data"
    start_time = datetime.now()
    print(f"開始時間: {start_time}")

    main(dir_path, data_dir, data_size=-1, pic_batch=10, detrend_methods=detrend_methods, cut_tail_len=cut_tail_len)
    end_time = datetime.now()
    print(f"完成時間: {end_time}")
    print(f"總共耗時: {end_time - start_time}")
