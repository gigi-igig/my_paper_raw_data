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
from datetime import datetime

# -------------------------------
# II. 主程式
# -------------------------------
def main(version_dir_path, data_dir, data_size=1000):
    """
    cut_tail_len: int, 尾端要裁掉的點數，0 表示不裁
    """
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
        if len(time) < 1440*5:
            continue
        time, flux_norm = normalize_by_group(time, flux, dt_threshold=400/1440, flux_gap_sigma=None)
        # 儲存 CSV
        combo_dir = f"/data2/gigicheng/data_21/raw_data/group_norm/{version_dir_path}/"
        os.makedirs(combo_dir, exist_ok=True)
        csv_path = os.path.join(combo_dir, f"{target_id}_group_norm.csv")
        pd.DataFrame({"TIME": time, "FLUX": flux_norm}).to_csv(csv_path, index=False)

        results.append({
            "TIC ID": target_id,
            "raw_flux_std":np.nanstd(flux),
            "org_flux_std": np.nanstd(flux_norm),
            "points_num": len(flux_norm)
        })

    # 儲存 summary CSV
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(f"/data2/gigicheng/data_21/raw_data/group_norm/group_summary_{version_dir_path}.csv", index=False)
    print("\n所有 FITS 檔案前處理完成！統計 CSV 已生成。")


if __name__ == "__main__":
    version_dir_path = "exp1"
    data_dir = "/data2/gigicheng/data_21_lc/"
    start_time = datetime.now()
    print(f"開始時間: {start_time}")

    main(version_dir_path, data_dir, data_size=-1)
    end_time = datetime.now()
    print(f"完成時間: {end_time}")
    print(f"總共耗時: {end_time - start_time}")
