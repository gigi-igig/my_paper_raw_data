#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非互動式後端

def extract_target_id(fits_name: str) -> str:
    parts = fits_name.split("-")
    if len(parts) >= 3:
        middle = parts[2]
        digits = ''.join(ch for ch in middle if ch.isdigit())
        if len(digits) >= 10:
            return digits[-10:]
        return digits
    return "UNKNOWN_ID"

def dropnan_sort(time, flux):
    valid = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[valid], flux[valid]
    sort_idx = np.argsort(time)
    return time[sort_idx], flux[sort_idx]


def normalize_by_group(time, flux, dt_threshold=400/1440, flux_gap_sigma=None):
    """
    根據時間間隔 (dt_threshold) 與 flux_gap_sigma 分段，
    並在每段內進行平均正規化 (flux_norm = flux / group_mean)。

    回傳:
        time : np.ndarray
        flux_norm : np.ndarray
    """
    # === 分段依據 ===
    diff_time = np.diff(time, prepend=time[0])
    if flux_gap_sigma is not None:
        flux_std = np.nanstd(flux)
        diff_flux = np.diff(flux, prepend=flux[0])
        new_seg = (diff_time > dt_threshold) | (np.abs(diff_flux) > flux_gap_sigma * flux_std)
    else:
        new_seg = (diff_time > dt_threshold)

    seg_idx = np.flatnonzero(new_seg)
    seg_idx = np.r_[0, seg_idx, len(time)]

    # === 各段平均正規化 ===
    flux_norm = np.empty_like(flux, dtype=float)
    for i in range(len(seg_idx) - 1):
        start, end = seg_idx[i], seg_idx[i + 1]
        group_flux = flux[start:end]
        group_mean = np.nanmean(group_flux)
        flux_norm[start:end] = group_flux / group_mean if group_mean != 0 else group_flux

    return time, flux_norm
