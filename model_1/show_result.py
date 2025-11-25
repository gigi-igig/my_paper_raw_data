import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"E:\my_paper\raw_data\CNN4_layer\2000_bin_10_Pto1_256_d1\training_results_epoch.csv")

# 只取 epoch 為 5 的倍數
df = df[df['epoch'] % 5 == 0]

colors = {
    'org': 'red',
    'cubic_sample': 'blue'
}

metrics = ['acc', 'val_acc']

for metric in metrics:
    plt.figure(figsize=(12,6))
    for detrend in df['detrend'].unique():
        df_d = df[df['detrend'] == detrend]
        
        grouped = df_d.groupby(['epoch', 'fold'])[metric].mean().reset_index()
        pivot = grouped.pivot(index='epoch', columns='fold', values=metric)
        
        epoch_vals = pivot.index
        
        for i, epoch in enumerate(epoch_vals):
            data = pivot.loc[epoch].values
            median = np.median(data)
            q1 = np.min(data)
            q3 = np.max(data)
            
            # 畫箱型線
            plt.vlines(epoch, q1, q3, color=colors[detrend], linewidth=2)
            # 畫中位數線
            plt.hlines(median, epoch-0.4, epoch+0.4, color=colors[detrend], linewidth=3)
            # 標記最大值、最小值、中位數
            plt.scatter(epoch, q1, color=colors[detrend], marker='v', s=30, zorder=5, label=f"{detrend} min" if i==0 else "")
            plt.scatter(epoch, q3, color=colors[detrend], marker='^', s=30, zorder=5, label=f"{detrend} max" if i==0 else "")
            plt.scatter(epoch, median, color=colors[detrend], marker='o', s=50, zorder=5, label=f"{detrend} median" if i==0 else "")
    
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()
