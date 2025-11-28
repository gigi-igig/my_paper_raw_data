from datetime import datetime
import os
import pandas as pd
import numpy as np
import sys

RAW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RAW_ROOT)
from preprocess.group_filter import group_norm
from preprocess.main import preprocess
from preprocess.config import detrend_methods, interval_list, no_interval_detrend
from injection.fold_way import bin_lightcurve
from injection.tool import pad_to_npoints
import pickle

TOI_to1 = ["0198457103", "0287225295", "0154867950",
           "0224600500", "0332064670", "0219742885"]
TOI_to1_dict = {"0198457103":0.472105419, "0287225295":0.6097554, "0154867950":0.6664948, "0224600500":0.67632351, "0332064670":0.73654625, "0219742885":0.916799171}


def main(input_dir, group_dir, version_dir_path, preprocess_dir, fold_dir, bin_minutes, target_points):
    # group
    combo_dir = os.path.join(group_dir, version_dir_path)
    for TOI_ID in TOI_to1:
        filepath = f"{input_dir}/tess2020020091053-s0021-000000{TOI_ID}-0167-s_lc.fits"
        _ = group_norm(0, TOI_ID, filepath, combo_dir)
    
    # detrend
    preprocess_path = os.path.join(preprocess_dir, version_dir_path)
    preprocess_result = []
    for TOI_ID in TOI_to1:
        csv_path = os.path.join(group_dir, f"{version_dir_path}/{TOI_ID}_group_norm.csv")
        preprocess_result.append(preprocess(csv_name=TOI_ID, csv_path=csv_path, output_path=preprocess_path))
    summary_df = pd.DataFrame(preprocess_result)
    summary_df.to_csv(f"{preprocess_path}/preprocess_summary.csv", index=False)
    print("\n已處理完 所有挑選 TOI。")
    
    # fold
    for detrend_way in detrend_methods:
        fold_path = os.path.join(fold_dir, f"{version_dir_path}/{detrend_way}")
        os.makedirs(f"{fold_path}/data", exist_ok=True)
        X_all = []
        y_all = []
        df_params_all = []
        for TOI_ID in TOI_to1:
            if detrend_way in no_interval_detrend:
                csv_path = f"{preprocess_path}/{detrend_way}/{TOI_ID}_intervalNone_{detrend_way}_cut30.csv"
                interval=None
            else:
                csv_path = f"{preprocess_path}/{detrend_way}/{TOI_ID}_interval4500_{detrend_way}_cut30.csv"
                interval=4500
            df_lc_1 = pd.read_csv(csv_path)
            binned_1 = pad_to_npoints(
                    bin_lightcurve(df_lc_1, TOI_to1_dict[TOI_ID], interval=interval, bin_minutes=bin_minutes),
                    target_points=target_points
            )
            # 儲存 CSV
            fname_1 = f"{TOI_ID}_inj_p0.csv"

            binned_1.to_csv(f"{fold_path}/data/{fname_1}", index=False)

            # CNN training 資料
            X_all.append(binned_1['flux_avg'].values)
            y_all.append(1)

            # 統計資料
            df_params_all.append({
                "TIC": TOI_ID,
                "detrend_way": detrend_way,
                "src_file": os.path.basename(csv_path),
                "file": fname_1,
                "signal": 1,
                "period_days": TOI_to1_dict[TOI_ID],
                "delta_min": 0,
                "bin_minutes": bin_minutes,
            })
        # 存 X, y, df_params
        X = np.array(X_all)[..., np.newaxis]
        y = np.array(y_all)

        with open(f"{fold_path}/X.pkl", "wb") as f:
            pickle.dump(X, f)
        with open(f"{fold_path}/y.pkl", "wb") as f:
            pickle.dump(y, f)

        pd.DataFrame(df_params_all).to_csv(f"{fold_path}/df_params.csv", index=False)

        print(f"\n完成 detrend_way: {detrend_way} ，資料存到：{fold_path}\n")


if __name__ == "__main__":
    version_dir_path = "exp1"
    root_dir = "/data2/gigicheng/data_21/raw_data/TOI_goal"
    input_dir = "/data2/gigicheng/TOI_org/data/"
    group_dir = f"{root_dir}/group_norm"
    preprocess_dir = f"{root_dir}/preprocess"
    fold_dir = f"{root_dir}/fold"
    bin_minutes = 10
    target_points = 256
    start_time = datetime.now()
    print(f"開始時間: {start_time}")
    main(input_dir, group_dir, version_dir_path, preprocess_dir, fold_dir, bin_minutes, target_points)
    
    end_time = datetime.now()
    print(f"完成時間: {end_time}")
    print(f"總共耗時: {end_time - start_time}")
