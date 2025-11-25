import os
import numpy as np
import pandas as pd

# 目標資料夾
root = "/data2/gigicheng/data_21/raw_data/inject_results/2000_bin_10_Pto1_256_d1/org/data"

# 取得所有 CSV
files = sorted([f for f in os.listdir(root) if f.endswith(".csv")])

# 只取前 600 個
files = files[:6000]

flux_list = []
label_list = []
param_list = []

for filename in files:
    filepath = os.path.join(root, filename)

    # 決定 label
    if "_inj" in filename:
        label = 1
    elif "_org" in filename:
        label = 0
    else:
        raise ValueError(f"無法判定 label: {filename}")

    # 讀 CSV
    df = pd.read_csv(filepath)
    if "flux_avg" not in df.columns:
        raise ValueError(f"{filename} 沒有 flux_avg 欄位！")

    flux = df["flux_avg"].values
    flux_list.append(flux)
    label_list.append(label)
    param_list.append(filename)

# 檢查所有 flux 長度一致
lengths = [len(f) for f in flux_list]
if len(set(lengths)) != 1:
    raise ValueError(f"Flux 長度不一致: {set(lengths)}")

# 轉成 DataFrame
df_flux = pd.DataFrame(flux_list)   # shape (N, time_steps)
df_label = pd.DataFrame(label_list, columns=['label'])
df_param = pd.DataFrame(param_list, columns=['param'])

# 輸出成三個 CSV
df_flux.to_csv("flux.csv", index=False)
df_label.to_csv("label.csv", index=False)
df_param.to_csv("param.csv", index=False)

print("完成！已產生三個 CSV：")
print(f"  flux.csv   → shape {df_flux.shape}")
print(f"  label.csv  → shape {df_label.shape}")
print(f"  param.csv  → shape {df_param.shape}")
