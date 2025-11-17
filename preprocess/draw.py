#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib
matplotlib.use("Agg")  # 非互動式後端
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from config import colors

# -------------------------------
# 畫圖函式改寫
# -------------------------------
def plot_preprocess_by_interval(time, flux_dict, plot_dir, target_id, interval):
    """
    interval: int，例如 300 或 750 或 None
    flux_dict 內包含所有方法 (含 org, linear 等)
    """
    plt.figure(figsize=(12, 6))

    # 畫出對應 interval 或 interval=None 的方法
    for key, flux in flux_dict.items():
        method = key.split("_int")[0]
        key_interval = key.split("_int")[-1]

        if key_interval == str(interval) or key_interval == "None":
            label = method if key_interval == "None" else f"{method} (int={interval})"
            plt.scatter(time, flux, s=0.1, color=colors.get(method, "black"), label=label, alpha=0.7)

    plt.xlabel("Time [days]")
    plt.ylabel("Flux (normalized)")
    plt.title(f"{target_id} - Detrend Comparison (interval={interval})")
    plt.legend(markerscale=5)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{target_id}_interval{interval}.png"))
    plt.close()

def plot_preprocess_by_interval_html(time, flux_dict, html_dir, target_id, interval):
    """
    將 flux_dict 裡不同方法的光曲線繪製成互動 HTML (Plotly) - 散點圖版本
    
    Parameters
    ----------
    time : 1D array
        時間序列
    flux_dict : dict
        各種去趨勢方法的 flux，例如：
        {
            'cubic_sample_int300': flux_array,
            'cubic_sample_BIC_int300': flux_array,
            'only_BIC': flux_array,
            'linear_intNone': flux_array,
            'org_intNone': flux_array
        }
    html_dir : str
        HTML 存放資料夾
    target_id : str
        目標 TIC 或 ID
    interval : int
        例如 300, 750 分鐘
    """
    import plotly.graph_objects as go
    import os

    os.makedirs(html_dir, exist_ok=True)
    
    fig = go.Figure()
    
    # 畫出對應 interval 或 interval=None 的方法
    for key, flux in flux_dict.items():
        method = key.split("_int")[0]
        key_interval = key.split("_int")[-1]

        if key_interval == str(interval) or key_interval == "None":
            label = method if key_interval == "None" else f"{method} (int={interval})"
            fig.add_trace(go.Scatter(
                x=time,
                y=flux,
                mode="markers",  # 改成 scatter
                name=label,
                marker=dict(color=colors.get(method, "black"), size=3),
                opacity=0.7
            ))
    
    fig.update_layout(
        title=dict(text=f"{target_id} - Detrend Comparison (interval={interval})", x=0.5),
        xaxis_title="Time [days]",
        yaxis_title="Flux (normalized)",
        template="plotly_white",
        legend=dict(itemsizing="constant")
    )
    
    html_file = os.path.join(html_dir, f"{target_id}_interval{interval}_scatter.html")
    fig.write_html(html_file)

def plot_preprocess_all_intervals(time, flux_dict, plot_dir, target_id):
    """
    將不同 interval 的光曲線畫在同一張圖上比較
    flux_dict: 包含所有 detrend 方法與 interval，例如 "cubic_sample_int750"
    """
    plt.figure(figsize=(14, 7))

    # 依照 detrend 方法和 interval 畫圖
    for key, flux in flux_dict.items():
        if "_int" in key:
            method, key_interval = key.split("_int")
        else:
            method, key_interval = key, "None"

        label = f"{method} (int={key_interval})" if key_interval != "None" else f"{method}"
        plt.scatter(time, flux, s=0.1, color=colors.get(method, "black"), label=label, alpha=0.6)

    plt.xlabel("Time [days]")
    plt.ylabel("Flux (normalized)")
    plt.title(f"{target_id} - Detrend Comparison (all intervals)")
    plt.legend(markerscale=5, fontsize=8, ncol=2)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"{target_id}_all_intervals.png")
    plt.savefig(save_path)
    plt.close()

def plot_preprocess_all_intervals_html(time, flux_dict, html_dir, target_id):
    """
    將 flux_dict 裡不同 interval 與方法的光曲線繪製成互動 HTML (Plotly) - 散點圖版本
    Parameters
    ----------
    time : 1D array
        時間序列
    flux_dict : dict
        各種去趨勢方法與 interval，例如：
        {
            'cubic_sample_int300': flux_array,
            'cubic_sample_BIC_int750': flux_array,
            'only_BIC_intNone': flux_array,
            'linear_intNone': flux_array
        }
    html_dir : str
        HTML 存放資料夾
    target_id : str
        目標 TIC 或 ID
    """
    import plotly.graph_objects as go
    import os

    os.makedirs(html_dir, exist_ok=True)
    
    fig = go.Figure()
    
    # 畫所有方法與 interval
    for key, flux in flux_dict.items():
        if "_int" in key:
            method, key_interval = key.split("_int")
        else:
            method, key_interval = key, "None"

        label = f"{method} (int={key_interval})" if key_interval != "None" else f"{method}"
        fig.add_trace(go.Scatter(
            x=time,
            y=flux,
            mode="markers",
            name=label,
            marker=dict(color=colors.get(method, "black"), size=3),
            opacity=0.6
        ))
    
    fig.update_layout(
        title=dict(text=f"{target_id} - Detrend Comparison (all intervals)", x=0.5),
        xaxis_title="Time [days]",
        yaxis_title="Flux (normalized)",
        template="plotly_white",
        legend=dict(itemsizing="constant", font=dict(size=10), tracegroupgap=10)
    )
    
    html_file = os.path.join(html_dir, f"{target_id}_all_intervals_scatter.html")
    fig.write_html(html_file)
