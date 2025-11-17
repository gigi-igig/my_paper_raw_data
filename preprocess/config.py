# ---------------------------
# 參數設定
# ---------------------------
interval_list = [100, 350, 750, 1500, 4500]
detrend_methods = ["org", "cubic_sample", "linear", "cubic_sample_BIC", "linear_BIC", "only_BIC"]#, "wavelet"]
colors = {
    "cubic_sample": "#1f77b4",
    "cubic_sample_BIC": "#ff7f0e",
    "org": "#0f160f",
    "only_BIC": "#9467bd",
    "linear": "#d62728",
    "linear_BIC": "#11f460",
    "wavelet":"#667b0a",
}
no_interval_detrend = ["org", "linear", "wavelet"]
SHOW_LINES = None  # None 表示全部顯示
cut_tail_len = 30 # by GPFC