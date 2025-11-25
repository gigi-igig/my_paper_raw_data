import numpy as np
import pandas as pd
import os
import glob
import pickle
from tqdm import tqdm
from mangol import mandel_agol as mangol
from fold_way import bin_lightcurve
from tool import inject, pad_to_npoints, generate_signals, ha_z
from config import detrend_methods


def main(tic_ids_str, detrend_way, preprocess_root, save_root, tic_ids_signal_dict, target_points, bin_minutes = 10):
    # 每個 detrend_way 自己的資料夾
    save_dir = f"{save_root}/{detrend_way}"
    os.makedirs(f"{save_dir}/data", exist_ok=True)

    df_params_all = []
    X_all = []
    y_all = []

    np.random.seed(1)

    print(f"\n=== 開始 detrend_way: {detrend_way} ===\n")

    for tic in tqdm(tic_ids_str, desc=f"Processing TIC IDs ({detrend_way})"):

        # 每個 detrend_way 的資料來源
        input_dir = f"{preprocess_root}/{detrend_way}"

        # 找出該 TIC 的唯一檔案
        csv_files = sorted(glob.glob(f"{input_dir}/{tic}_*.csv"))

        if len(csv_files) == 0:
            print(f"⚠ 找不到檔案：{input_dir}/{tic}_*.csv")
            continue
        elif len(csv_files) > 1:
            print(f"⚠ 警告：找到多於 1 個檔案：{csv_files}，只使用第一個")
        
        # 只取第一個檔案
        csv_path = csv_files[0]

        df = pd.read_csv(csv_path)
        noi = df["FLUX"].values + 1
        t = df["TIME"].values

        df_lc_0 = pd.DataFrame({"TIME": t, "FLUX": noi})

        # 注入訊號
        sig_params = tic_ids_signal_dict[tic]
        z = ha_z(t, sig_params['t0'], sig_params['period_days'], sig_params['a_rs'], sig_params['iang'])
        y = mangol(z, u1=0.5, u2=0, p0 = sig_params['rp_rs'] )
        yk = y * noi
        df_lc_1 = pd.DataFrame({"TIME": t, "FLUX": yk})

        # 3 種摺疊偏移
        for delta_min in [-1.5, 0, 1.5]:

            pd_test = sig_params['period_days'] + delta_min / 1440
            tag = ("p_0" if delta_min == 0
                   else f"p_m{abs(delta_min)}" if delta_min < 0
                   else f"p_p{abs(delta_min)}")

            # binning
            binned_1 = pad_to_npoints(
                bin_lightcurve(df_lc_1, pd_test, interval=None, bin_minutes=bin_minutes),
                target_points=target_points
            )
            binned_0 = pad_to_npoints(
                bin_lightcurve(df_lc_0, pd_test, interval=None, bin_minutes=bin_minutes),
                target_points=target_points
            )

            # 儲存 CSV
            fname_1 = f"{tic}_inj_{tag}.csv"
            fname_0 = f"{tic}_org_{tag}.csv"

            binned_1.to_csv(f"{save_dir}/data/{fname_1}", index=False)
            binned_0.to_csv(f"{save_dir}/data/{fname_0}", index=False)

            # CNN training 資料
            X_all.append(binned_1['flux_avg'].values)
            y_all.append(1)
            X_all.append(binned_0['flux_avg'].values)
            y_all.append(0)

            # 統計資料
            df_params_all.append({
                "TIC": tic,
                "detrend_way": detrend_way,
                "src_file": os.path.basename(csv_path),
                "file": fname_1,
                "signal": 1,
                "period_days": sig_params['period_days'],
                "delta_min": delta_min,
                "bin_minutes": bin_minutes,
                "rp_rs": sig_params['rp_rs'],
                "a_rs": sig_params['a_rs'],
                "iang": sig_params['iang'],
                "t0": sig_params['t0']
            })

            df_params_all.append({
                "TIC": tic,
                "detrend_way": detrend_way,
                "src_file": os.path.basename(csv_path),
                "file": fname_0,
                "signal": 0,
                "period_days": sig_params['period_days'],
                "delta_min": delta_min,
                "bin_minutes": bin_minutes,
                "rp_rs": np.nan,
                "a_rs": np.nan,
                "iang": np.nan,
                "t0": np.nan
            })

    # 存 X, y, df_params
    X = np.array(X_all)[..., np.newaxis]
    y = np.array(y_all)

    with open(f"{save_dir}/X.pkl", "wb") as f:
        pickle.dump(X, f)
    with open(f"{save_dir}/y.pkl", "wb") as f:
        pickle.dump(y, f)

    pd.DataFrame(df_params_all).to_csv(f"{save_dir}/df_params.csv", index=False)

    print(f"\n完成 detrend_way: {detrend_way} ，資料存到：{save_dir}\n")


if __name__ == "__main__":

    summary_csv = "/data2/gigicheng/data_21/raw_data/preprocess/2000_mean_to_0/preprocess_summary_2000_mean_to_0.csv"
    df_summary = pd.read_csv(summary_csv)

    tic_ids = df_summary["TIC ID"].dropna().astype(int).unique()
    tic_ids_str = [str(t).zfill(10) for t in tic_ids]

    preprocess_root = "/data2/gigicheng/data_21/raw_data/preprocess/2000_mean_to_0"
    save_root = "/data2/gigicheng/data_21/raw_data/inject_results/2000_bin_10_Pto1_256_d1"

    os.makedirs(save_root, exist_ok=True)
    tic_ids_signal_dict = generate_signals(tic_ids_str, period_day_begin=0.4, period_day_end=1)
    for detrend_way in detrend_methods:
        main(tic_ids_str, detrend_way, preprocess_root, save_root, tic_ids_signal_dict, target_points=256, bin_minutes = 10)