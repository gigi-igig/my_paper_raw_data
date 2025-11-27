from datetime import datetime
from preprocess.group_filter import group_norm
from preprocess.main import preprocess
import os
import pandas as pd
import numpy as np

TOI_to1 = ["0198457103", "0287225295", "0154867950",
           "0224600500", "0332064670", "0219742885"]

interval_list = [4500]
detrend_methods = ["org", "cubic_sample"]
no_interval_detrend = ["org", "linear", "wavelet"]
cut_tail_len = 30 # by GPFC

def main(input_dir, group_dir, version_dir_path, preprocess_dir):
    # group
    combo_dir = os.path.join(group_dir, version_dir_path)
    for TOI_ID in TOI_to1:
        filepath = f"{input_dir}/tess2020020091053-s0021-000000{TOI_ID}-0167-s_lc.fits"
        group_norm(0, TOI_ID, filepath, group_dir, combo_dir)
    
    # detrend
    output_path = os.path.join(preprocess_dir, version_dir_path)
    preprocess_result = []
    for TOI_ID in TOI_to1:
        csv_path = os.path.join(group_dir, f"{version_dir_path}/{TOI_ID}_group_norm.csv")
        preprocess_result.append(preprocess(csv_name=TOI_ID, csv_path=csv_path, output_path=output_path))
    summary_df = pd.DataFrame(preprocess_result)
    summary_df.to_csv(f"{output_path}/preprocess_summary.csv", index=False)
    print("\n已處理完 sample_df 的所有 TIC。")
    


if __name__ == "__main__":
    version_dir_path = "exp1"
    
    input_dir = "/data2/gigicheng/TOI_org/data/"
    group_dir = "/data2/gigicheng/data_21/raw_data/TOI_goal/group_norm"
    preprocess_dir = "/data2/gigicheng/data_21/raw_data/TOI_goal/preprocess"

    start_time = datetime.now()
    print(f"開始時間: {start_time}")
    main(input_dir, group_dir, version_dir_path, preprocess_dir)
    
    end_time = datetime.now()
    print(f"完成時間: {end_time}")
    print(f"總共耗時: {end_time - start_time}")
