#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import pywt
matplotlib.use("Agg")  # 非互動式後端
from scipy.interpolate import UnivariateSpline, CubicSpline


def choose_s_by_bic(time_seg, flux_seg, mask=None, s_candidates=None, max_iter=3, use_cache=True, _s_bic_cache=None):
    """根據 BIC 準則自動選擇 UnivariateSpline 的平滑參數 s。"""
    if _s_bic_cache is None:
        _s_bic_cache = {}

    if mask is None:
        mask = np.zeros_like(flux_seg, dtype=bool)

    valid = np.isfinite(time_seg) & np.isfinite(flux_seg) & (~mask)
    t, f = time_seg[valid], flux_seg[valid]
    n = len(t)
    if n < 10:
        return None, _s_bic_cache

    order = np.argsort(t)
    t, f = t[order], f[order]

    seg_key = (int(np.round(n, -1)), round(float(np.nanstd(f)), 4))
    if use_cache and seg_key in _s_bic_cache:
        return _s_bic_cache[seg_key], _s_bic_cache

    w = 1.0 / (np.abs(np.gradient(f)) + 1e-3)
    w /= np.nanmax(w)

    if s_candidates is None:
        s_candidates = np.logspace(-2, 2, 8)

    best_s, best_bic = None, np.inf
    bic_list = []

    # 初步搜尋
    for s in s_candidates:
        try:
            spline = UnivariateSpline(t, f, s=s, w=w)
            resid = f - spline(t)
            rss = np.sum(resid**2)
            k = len(spline.get_knots())
            bic = n * np.log(rss / n + 1e-12) + k * np.log(n)
            bic_list.append((s, bic))
        except Exception:
            continue

    if not bic_list:
        return None, _s_bic_cache

    s_vals, bic_vals = zip(*bic_list)
    best_idx = np.argmin(bic_vals)
    best_s, best_bic = s_vals[best_idx], bic_vals[best_idx]

    # 細化搜尋
    s_min = max(best_s / 5, min(s_candidates))
    s_max = min(best_s * 5, max(s_candidates))
    fine_s_candidates = np.logspace(np.log10(s_min), np.log10(s_max), 8)

    for s in fine_s_candidates:
        try:
            spline = UnivariateSpline(t, f, s=s, w=w)
            resid = f - spline(t)
            rss = np.sum(resid**2)
            k = len(spline.get_knots())
            bic = n * np.log(rss / n + 1e-12) + k * np.log(n)
            if bic < best_bic:
                best_bic, best_s = bic, s
        except Exception:
            continue

    # 迭代更新權重（robust fit）
    for _ in range(max_iter):
        spline = UnivariateSpline(t, f, s=best_s, w=w)
        resid = f - spline(t)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
        w_new = 1.0 / (1.0 + (resid / (6 * mad))**2)
        if np.allclose(w, w_new, atol=1e-2):
            break
        w = w_new

    if use_cache:
        _s_bic_cache[seg_key] = best_s

    return best_s, _s_bic_cache


def detrend_flux(method, t_seg, f_seg, interval=360, _s_bic_cache=None):
    """
    對單段光曲線執行去趨勢（linear, cubic_sample, *_BIC）。
    回傳 (f_detrended, _s_bic_cache)
    """
    if _s_bic_cache is None:
        _s_bic_cache = {}

    # --- 基本去趨勢 ---
    if method in ["linear", "linear_BIC"]:
        A = np.vstack([t_seg, np.ones_like(t_seg)]).T
        a, b = np.linalg.lstsq(A, f_seg, rcond=None)[0]
        baseline = a * t_seg + b
        f_seg = f_seg / baseline

    elif method in ["cubic_sample", "cubic_sample_BIC"]:
        t_min, t_max = t_seg[0], t_seg[-1]
        interval_days = interval / 1440  # 分鐘轉成天
        sample_times = np.linspace(t_min, t_max, max(int((t_max - t_min) / interval_days), 2))
        sample_flux = np.interp(sample_times, t_seg, f_seg)
        cs = CubicSpline(sample_times, sample_flux)
        f_seg = f_seg / cs(t_seg)

    elif method in ["wavelet"]:
        wavelet = "db2"
        level = 1  # 一階分解（approx + first detail），預設最多分解 4 層

        coeffs = pywt.wavedec(f_seg, wavelet=wavelet, level=level, mode="periodization")

        # coeffs = [A1, D1]
        A1, D1 = coeffs
        # 若要模擬他們「A1 和 D1 合併後送入模型」
        features = np.concatenate([A1, D1])

        # 若你只想取 baseline（低頻部分）
        baseline = pywt.waverec([A1, None], wavelet=wavelet, mode="periodization")
        baseline = baseline[:len(f_seg)]
        f_seg = f_seg / baseline


    # --- BIC 選擇最佳平滑參數 ---
    if method in ["linear_BIC", "cubic_sample_BIC", "only_BIC"]:
        interval_days = interval / 1440
        t_min, t_max = t_seg[0], t_seg[-1]
        n_samples = max(int((t_max - t_min) / interval_days), 3)
        sample_times = np.linspace(t_min, t_max, n_samples)
        sample_flux = np.interp(sample_times, t_seg, f_seg)

        s_opt, _s_bic_cache = choose_s_by_bic(sample_times, sample_flux, _s_bic_cache=_s_bic_cache)
        if s_opt is not None:
            spline = UnivariateSpline(sample_times, sample_flux, s=s_opt)
            baseline_bic = spline(t_seg)
            if len(baseline_bic) == len(f_seg) and np.all(np.isfinite(baseline_bic)):
                f_seg = f_seg / baseline_bic
            else:
                print("[警告] BIC baseline 含 nan 或長度不符，跳過此段。")
        # --- Wavelet 去趨勢 ---


    return f_seg, _s_bic_cache
