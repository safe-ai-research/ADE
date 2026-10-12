# Feature calculation for complete recordings and individual EEG windows

import numpy as np
import pandas as pd

from scipy.signal import find_peaks, welch

try:
    import pywt
    use_wavelet_available = True
except ImportError:
    pywt = None
    use_wavelet_available = False


# Bandpower
def bandpower(f, Pxx, fmin, fmax):
    idx = (f >= fmin) & (f <= fmax)
    return np.trapezoid(Pxx[idx], f[idx])


def get_wavelet_level(signal_len, wavelet_name, requested_level=None):
    if not use_wavelet_available:
        return 0

    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(signal_len, wavelet.dec_len)

    if max_level < 1:
        return 0

    if requested_level is None:
        return min(5, max_level)

    return max(1, min(int(requested_level), max_level))


# Wavelet Features
def compute_wavelet_features(
    signal,
    wavelet_name="db4",
    level=5,
    mode="symmetric"
):
    if not use_wavelet_available:
        return {
            "detail_rel": 0.0,
            "hf_rel": 0.0,
            "entropy": 0.0,
            "maxabs_norm": 0.0,
            "kurtosis": 0.0,
        }

    signal = np.asarray(signal)

    level_use = get_wavelet_level(
        len(signal),
        wavelet_name,
        requested_level=level
    )

    if level_use < 1:
        return {
            "detail_rel": 0.0,
            "hf_rel": 0.0,
            "entropy": 0.0,
            "maxabs_norm": 0.0,
            "kurtosis": 0.0,
        }

    coeffs = pywt.wavedec(
        signal,
        wavelet=wavelet_name,
        mode=mode,
        level=level_use
    )

    # coeffs = [cA_n, cD_n, cD_n-1, ..., cD1]
    approx = coeffs[0]
    details = coeffs[1:]

    approx_energy = np.sum(approx ** 2)
    detail_energies = np.array([np.sum(d ** 2) for d in details])

    total_energy = approx_energy + np.sum(detail_energies) + 1e-10

    detail_rel = np.sum(detail_energies) / total_energy

    # Highest Frequency Details:
    # cD1 is details[-1], cD2 is details[-2]
    if len(detail_energies) >= 2:
        hf_energy = detail_energies[-1] + detail_energies[-2]
    else:
        hf_energy = detail_energies[-1]

    hf_rel = hf_energy / total_energy

    # Energy-Entropy by Detail Level
    p = detail_energies / (np.sum(detail_energies) + 1e-10)
    entropy = -np.sum(p * np.log2(p + 1e-10))

    all_details = np.concatenate(details)

    maxabs_norm = np.max(np.abs(all_details)) / (np.std(signal) + 1e-10)

    z = (all_details - np.mean(all_details)) / (np.std(all_details) + 1e-10)
    kurtosis = np.mean(z ** 4)

    return {
        "detail_rel": float(detail_rel),
        "hf_rel": float(hf_rel),
        "entropy": float(entropy),
        "maxabs_norm": float(maxabs_norm),
        "kurtosis": float(kurtosis),
    }


# Robust Time-Domain Features
def compute_robust_time_features(signal, eps=1e-10):
    signal = np.asarray(signal)

    if len(signal) == 0:
        return {
            "std": 0.0,
            "mad": 0.0,
            "ptp": 0.0,
            "maxabs": 0.0,
            "max_jump_norm": 0.0,
            "line_length_norm": 0.0,
            "trend_slope_norm": 0.0,
            "clipping_ratio": 0.0,
        }

    std = float(np.std(signal))
    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    ptp = float(np.ptp(signal))
    maxabs = float(np.max(np.abs(signal)))

    diff = np.diff(signal)

    if len(diff) > 0:
        max_jump = float(np.max(np.abs(diff)))
        line_length = float(np.sum(np.abs(diff)))
    else:
        max_jump = 0.0
        line_length = 0.0

    max_jump_norm = max_jump / (std + eps)
    line_length_norm = line_length / (len(signal) * (std + eps))

    # Trend Strength relative to Window Standard deviation
    if len(signal) > 1 and std > eps:
        x = np.arange(len(signal), dtype=float)
        slope = float(np.polyfit(x, signal, 1)[0])
        trend_slope_norm = abs(slope) * len(signal) / (std + eps)
    else:
        trend_slope_norm = 0.0

    # Clipping-/Saturation approximation: Amount of values very close to local minimum or maximum
    if ptp > eps:
        sig_min = float(np.min(signal))
        sig_max = float(np.max(signal))
        tol = 0.005 * ptp

        clipping_ratio = float(
            np.mean(
                (np.abs(signal - sig_min) <= tol) |
                (np.abs(signal - sig_max) <= tol)
            )
        )
    else:
        clipping_ratio = 1.0

    return {
        "std": std,
        "mad": mad,
        "ptp": ptp,
        "maxabs": maxabs,
        "max_jump_norm": float(max_jump_norm),
        "line_length_norm": float(line_length_norm),
        "trend_slope_norm": float(trend_slope_norm),
        "clipping_ratio": float(clipping_ratio),
    }


# -------------------------
# Definition: ECG Detection
# -------------------------
def detect_ecg(
    signal,
    fs,
    hr_ref=None,                # Reference-Heartrate (e.g. from ECG_HR / PLETH_HR)
    use_autocorr=True,          # Use Auto Correlation (Default: True, if too many false-negatives → False)

    # Peak Detection Parameter
    peak_height_factor=1.0,     # multiplicated with std(signal) (Default: 1.0, Stricter 1.5, Softer 0.8)
    min_hr=40,                  # Heartrate-Minimum (Default: 40, Stricter 50, Softer 30)
    max_hr=120,                 # Heartrate-Maximum (Default: 120, Stricter 100, Softer 150)

    # Auto Correlation Parameter
    autocorr_threshold=0.3,

    # Tolerance to Reference (in bpm)
    hr_tolerance=15             # Heartrate Tolerance (Default: 15, Stricter 10, Softer 20)
):
    signal = signal - np.mean(signal)
    std = np.std(signal)

    # --- Peak Detection ---
    min_distance = fs * (60 / max_hr)   # minimal distance

    peaks, props = find_peaks(
        signal,
        height=peak_height_factor * std,
        distance=min_distance
    )

    if len(peaks) < 3:
        return False

    intervals = np.diff(peaks) / fs
    mean_interval = np.mean(intervals)

    if mean_interval == 0:
        return False

    hr_est = 60 / mean_interval

    # Plausibility Check
    if not (min_hr <= hr_est <= max_hr):
        return False

    # --- Auto Correlation ---
    if use_autocorr:
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr)//2:]

        min_lag = int(fs * (60 / max_hr))
        max_lag = int(fs * (60 / min_hr))

        if max_lag <= min_lag or max_lag >= len(corr):
            return False

        segment = corr[min_lag:max_lag]

        if len(segment) == 0:
            return False

        peak_corr = np.max(segment)

        if peak_corr < autocorr_threshold * corr[0]:
            return False

    # --- Compare with Reference-HR ---
    if hr_ref is not None:
        if abs(hr_est - hr_ref) > hr_tolerance:
            return False

    return True


def calculate_global_features(eeg1, eeg2, fs):
    # =========================
    # PSD (Welch)
    # =========================
    # Calculate the unchanged global PSD and bandpower values

    nperseg = min(1024, len(eeg1)//2)   # Custom: Minimum of segment size (Default: 1024 or len(eeg1//2))

    # GLOBAL PSD
    f_psd_global_1, Pxx_global_1 = welch(eeg1, fs, nperseg=nperseg) # EEG1
    f_psd_global_2, Pxx_global_2 = welch(eeg2, fs, nperseg=nperseg) # EEG2

    # =========================
    # BANDPOWER
    # =========================

    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta":  (13, 30),
    }

    print("\nRelative Bandpower EEG1:")
    total1 = bandpower(f_psd_global_1, Pxx_global_1, 0.5, 30)
    rel_bandpower_eeg1 = {}

    if total1 == 0:
        total1 = 1e-10

    for name, (low, high) in bands.items():
        bp = bandpower(f_psd_global_1, Pxx_global_1, low, high)
        rel_bandpower_eeg1[name] = bp / total1
        print(f"{name}: {bp / total1:.3f}")

    print("\nRelative Bandpower EEG2:")
    total2 = bandpower(f_psd_global_2, Pxx_global_2, 0.5, 30)
    rel_bandpower_eeg2 = {}

    if total2 == 0:
        total2 = 1e-10

    for name, (low, high) in bands.items():
        bp = bandpower(f_psd_global_2, Pxx_global_2, low, high)
        rel_bandpower_eeg2[name] = bp / total2
        print(f"{name}: {bp / total2:.3f}")

    # PSD for difference
    diff = eeg1 - eeg2
    f_diff, Pxx_diff = welch(diff, fs, nperseg=nperseg)

    return {
        "nperseg": nperseg,
        "f_psd_global_1": f_psd_global_1,
        "Pxx_global_1": Pxx_global_1,
        "f_psd_global_2": f_psd_global_2,
        "Pxx_global_2": Pxx_global_2,
        "rel_bandpower_eeg1": rel_bandpower_eeg1,
        "rel_bandpower_eeg2": rel_bandpower_eeg2,
        "f_diff": f_diff,
        "Pxx_diff": Pxx_diff,
    }


def calculate_method_specific_global_features(
    signal1_for_features,
    signal2_for_features,
    eeg1,
    eeg2,
    fs,
):
    # =========================
    # METHOD SPECIFIC GLOBAL FEATURES
    # =========================

    feature_corr = np.corrcoef(signal1_for_features, signal2_for_features)[0, 1]

    f_feat_1, Pxx_feat_1 = welch(
        signal1_for_features,
        fs,
        nperseg=min(1024, len(signal1_for_features) // 2)
    )

    f_feat_2, Pxx_feat_2 = welch(
        signal2_for_features,
        fs,
        nperseg=min(1024, len(signal2_for_features) // 2)
    )

    total_feat_1 = bandpower(f_feat_1, Pxx_feat_1, 0.5, 30)
    total_feat_2 = bandpower(f_feat_2, Pxx_feat_2, 0.5, 30)

    if total_feat_1 == 0:
        total_feat_1 = 1e-10

    if total_feat_2 == 0:
        total_feat_2 = 1e-10

    rel_bandpower_feat_eeg1 = {
        "Delta": bandpower(f_feat_1, Pxx_feat_1, 0.5, 4) / total_feat_1,
        "Theta": bandpower(f_feat_1, Pxx_feat_1, 4, 8) / total_feat_1,
        "Alpha": bandpower(f_feat_1, Pxx_feat_1, 8, 13) / total_feat_1,
        "Beta":  bandpower(f_feat_1, Pxx_feat_1, 13, 30) / total_feat_1,
    }

    rel_bandpower_feat_eeg2 = {
        "Delta": bandpower(f_feat_2, Pxx_feat_2, 0.5, 4) / total_feat_2,
        "Theta": bandpower(f_feat_2, Pxx_feat_2, 4, 8) / total_feat_2,
        "Alpha": bandpower(f_feat_2, Pxx_feat_2, 8, 13) / total_feat_2,
        "Beta":  bandpower(f_feat_2, Pxx_feat_2, 13, 30) / total_feat_2,
    }

    # =========================
    # RECONSTRUCTION COMPARISON TO ORIGINAL
    # =========================

    recon_corr_eeg1 = np.corrcoef(eeg1, signal1_for_features)[0, 1]
    recon_corr_eeg2 = np.corrcoef(eeg2, signal2_for_features)[0, 1]

    recon_var_ratio_eeg1 = np.var(signal1_for_features) / (np.var(eeg1) + 1e-10)
    recon_var_ratio_eeg2 = np.var(signal2_for_features) / (np.var(eeg2) + 1e-10)

    recon_rmsdiff_eeg1 = np.sqrt(np.mean((eeg1 - signal1_for_features) ** 2))
    recon_rmsdiff_eeg2 = np.sqrt(np.mean((eeg2 - signal2_for_features) ** 2))

    return {
        "feature_corr": feature_corr,
        "rel_bandpower_feat_eeg1": rel_bandpower_feat_eeg1,
        "rel_bandpower_feat_eeg2": rel_bandpower_feat_eeg2,
        "recon_corr_eeg1": recon_corr_eeg1,
        "recon_corr_eeg2": recon_corr_eeg2,
        "recon_var_ratio_eeg1": recon_var_ratio_eeg1,
        "recon_var_ratio_eeg2": recon_var_ratio_eeg2,
        "recon_rmsdiff_eeg1": recon_rmsdiff_eeg1,
        "recon_rmsdiff_eeg2": recon_rmsdiff_eeg2,
    }


def calculate_window_features(
    windows_eeg1,
    windows_eeg2,
    windows_ecg,
    autoreject_bad_windows,
    autoreject_bad_eeg1,
    autoreject_bad_eeg2,
    fs,
    hr_ref,
    use_wavelet_marker_features,
    wavelet_name,
    wavelet_level,
    wavelet_mode,
):
    # =========================
    # WINDOW FEATURES
    # =========================

    window_features = []

    if windows_ecg is not None:
        iterator = zip(
            windows_eeg1,
            windows_eeg2,
            windows_ecg,
            autoreject_bad_windows,
            autoreject_bad_eeg1,
            autoreject_bad_eeg2
        )

    else:
        iterator = [
            (w1, w2, None, ar_bad, ar_bad1, ar_bad2)
            for w1, w2, ar_bad, ar_bad1, ar_bad2 in zip(
                windows_eeg1,
                windows_eeg2,
                autoreject_bad_windows,
                autoreject_bad_eeg1,
                autoreject_bad_eeg2
            )
        ]

    for w1, w2, wecg, ar_bad, ar_bad1, ar_bad2 in iterator:

        # WINDOW PSD
        f1_win, Pxx1_win = welch(w1, fs, nperseg=min(256, len(w1)//2))
        f2_win, Pxx2_win = welch(w2, fs, nperseg=min(256, len(w2)//2))

        # Bandpower
        delta1 = bandpower(f1_win, Pxx1_win, 0.5, 4)
        theta1 = bandpower(f1_win, Pxx1_win, 4, 8)
        alpha1 = bandpower(f1_win, Pxx1_win, 8, 13)
        beta1 = bandpower(f1_win, Pxx1_win, 13, 30)

        delta2 = bandpower(f2_win, Pxx2_win, 0.5, 4)
        theta2 = bandpower(f2_win, Pxx2_win, 4, 8)
        alpha2 = bandpower(f2_win, Pxx2_win, 8, 13)
        beta2 = bandpower(f2_win, Pxx2_win, 13, 30)

        # Bandpower relative per window
        total1_win = bandpower(f1_win, Pxx1_win, 0.5, 30)
        total2_win = bandpower(f2_win, Pxx2_win, 0.5, 30)

        if total1_win == 0:
            total1_win = 1e-10

        if total2_win == 0:
            total2_win = 1e-10

        delta1_rel = delta1 / total1_win
        theta1_rel = theta1 / total1_win
        alpha1_rel = alpha1 / total1_win
        beta1_rel  = beta1  / total1_win

        delta2_rel = delta2 / total2_win
        theta2_rel = theta2 / total2_win
        alpha2_rel = alpha2 / total2_win
        beta2_rel  = beta2  / total2_win

        # Correlation
        corr_w = np.corrcoef(w1, w2)[0,1]

        # Variance
        var1 = np.var(w1)
        var2 = np.var(w2)

        # Robust Time dimension features
        robust1 = compute_robust_time_features(w1)
        robust2 = compute_robust_time_features(w2)

        std_avg = (robust1["std"] + robust2["std"]) / 2
        mad_avg = (robust1["mad"] + robust2["mad"]) / 2
        ptp_avg = (robust1["ptp"] + robust2["ptp"]) / 2
        maxabs_avg = (robust1["maxabs"] + robust2["maxabs"]) / 2

        max_jump_norm_avg = (
            robust1["max_jump_norm"] + robust2["max_jump_norm"]
        ) / 2

        line_length_norm_avg = (
            robust1["line_length_norm"] + robust2["line_length_norm"]
        ) / 2

        trend_slope_norm_avg = (
            robust1["trend_slope_norm"] + robust2["trend_slope_norm"]
        ) / 2

        clipping_ratio_avg = (
            robust1["clipping_ratio"] + robust2["clipping_ratio"]
        ) / 2

        # Highfrequency amount for HF_EMG, deliberately above Beta
        hf1 = bandpower(f1_win, Pxx1_win, 30, 45)
        hf2 = bandpower(f2_win, Pxx2_win, 30, 45)

        total1_hf = bandpower(f1_win, Pxx1_win, 0.5, 45)
        total2_hf = bandpower(f2_win, Pxx2_win, 0.5, 45)

        if total1_hf == 0:
            total1_hf = 1e-10

        if total2_hf == 0:
            total2_hf = 1e-10

        hf1_rel = hf1 / total1_hf
        hf2_rel = hf2 / total2_hf
        hf_rel_avg = (hf1_rel + hf2_rel) / 2

        # ECG Artifact detection
        if wecg is not None:
            ecg_ref_flag = detect_ecg(
                wecg,
                fs,
                hr_ref=hr_ref
            )

            ecg_eeg1_candidate = detect_ecg(
                w1,
                fs,
                hr_ref=hr_ref
            )

            ecg_eeg2_candidate = detect_ecg(
                w2,
                fs,
                hr_ref=hr_ref
            )

            # Require ECG reference confirmation
            ecg_eeg1_flag = (
                ecg_ref_flag and ecg_eeg1_candidate
            )

            ecg_eeg2_flag = (
                ecg_ref_flag and ecg_eeg2_candidate
            )

        else:
            ecg_eeg1_flag = detect_ecg(
                w1,
                fs,
                hr_ref=hr_ref
            )

            ecg_eeg2_flag = detect_ecg(
                w2,
                fs,
                hr_ref=hr_ref
            )

        ecg_flag = (
            ecg_eeg1_flag |
            ecg_eeg2_flag
        )

        # Wavelet Features
        if use_wavelet_marker_features and use_wavelet_available:
            wavelet_feat1 = compute_wavelet_features(
                w1,
                wavelet_name=wavelet_name,
                level=wavelet_level,
                mode=wavelet_mode
            )

            wavelet_feat2 = compute_wavelet_features(
                w2,
                wavelet_name=wavelet_name,
                level=wavelet_level,
                mode=wavelet_mode
            )
        else:
            wavelet_feat1 = {
                "detail_rel": 0.0,
                "hf_rel": 0.0,
                "entropy": 0.0,
                "maxabs_norm": 0.0,
                "kurtosis": 0.0,
            }

            wavelet_feat2 = {
                "detail_rel": 0.0,
                "hf_rel": 0.0,
                "entropy": 0.0,
                "maxabs_norm": 0.0,
                "kurtosis": 0.0,
            }

        window_features.append({
            "delta1": delta1,
            "theta1": theta1,
            "alpha1": alpha1,
            "beta1": beta1,

            "delta2": delta2,
            "theta2": theta2,
            "alpha2": alpha2,
            "beta2": beta2,

            "delta1_rel": delta1_rel,
            "theta1_rel": theta1_rel,
            "alpha1_rel": alpha1_rel,
            "beta1_rel": beta1_rel,

            "delta2_rel": delta2_rel,
            "theta2_rel": theta2_rel,
            "alpha2_rel": alpha2_rel,
            "beta2_rel": beta2_rel,

            "corr": corr_w,
            "var1": var1,
            "var2": var2,

            "ecg_eeg1": bool(ecg_eeg1_flag),
            "ecg_eeg2": bool(ecg_eeg2_flag),
            "ecg": ecg_flag,

            "autoreject_bad_eeg1": bool(ar_bad1),
            "autoreject_bad_eeg2": bool(ar_bad2),
            "autoreject_bad": bool(ar_bad),

            # Robust time features EEG1
            "std1": robust1["std"],
            "mad1": robust1["mad"],
            "ptp1": robust1["ptp"],
            "maxabs1": robust1["maxabs"],
            "max_jump_norm1": robust1["max_jump_norm"],
            "line_length_norm1": robust1["line_length_norm"],
            "trend_slope_norm1": robust1["trend_slope_norm"],
            "clipping_ratio1": robust1["clipping_ratio"],

            # Robust time features EEG2
            "std2": robust2["std"],
            "mad2": robust2["mad"],
            "ptp2": robust2["ptp"],
            "maxabs2": robust2["maxabs"],
            "max_jump_norm2": robust2["max_jump_norm"],
            "line_length_norm2": robust2["line_length_norm"],
            "trend_slope_norm2": robust2["trend_slope_norm"],
            "clipping_ratio2": robust2["clipping_ratio"],

            # High-frequency features
            "hf_rel1": hf1_rel,
            "hf_rel2": hf2_rel,

            # Existing combined features
            "std_avg": std_avg,
            "mad_avg": mad_avg,
            "ptp_avg": ptp_avg,
            "maxabs_avg": maxabs_avg,
            "max_jump_norm_avg": max_jump_norm_avg,
            "line_length_norm_avg": line_length_norm_avg,
            "trend_slope_norm_avg": trend_slope_norm_avg,
            "clipping_ratio_avg": clipping_ratio_avg,
            "hf_rel_avg": hf_rel_avg,

            "wavelet_detail1": wavelet_feat1["detail_rel"],
            "wavelet_hf1": wavelet_feat1["hf_rel"],
            "wavelet_entropy1": wavelet_feat1["entropy"],
            "wavelet_maxabs1": wavelet_feat1["maxabs_norm"],
            "wavelet_kurtosis1": wavelet_feat1["kurtosis"],

            "wavelet_detail2": wavelet_feat2["detail_rel"],
            "wavelet_hf2": wavelet_feat2["hf_rel"],
            "wavelet_entropy2": wavelet_feat2["entropy"],
            "wavelet_maxabs2": wavelet_feat2["maxabs_norm"],
            "wavelet_kurtosis2": wavelet_feat2["kurtosis"],
        })

    # =========================
    # FEATURE DATAFRAME
    # =========================

    df_features = pd.DataFrame(window_features)

    print("\nFeature DataFrame:")
    print(df_features.head())

    return df_features, window_features


def add_engineered_features(df_features):
    """Add the existing combined and channel-asymmetry feature columns."""
    # Feature Engineering: absolute Bandpower
    df_features["beta_avg"]  = (df_features["beta1"] + df_features["beta2"]) / 2
    df_features["delta_avg"] = (df_features["delta1"] + df_features["delta2"]) / 2

    # Feature Engineering: relative Bandpower
    df_features["beta_rel_avg"]  = (df_features["beta1_rel"] + df_features["beta2_rel"]) / 2
    df_features["delta_rel_avg"] = (df_features["delta1_rel"] + df_features["delta2_rel"]) / 2

    # Feature Engineerung: Channel asymmetry
    df_features["var_diff"] = (
        df_features["var1"] - df_features["var2"]
    ).abs()

    df_features["std_diff"] = (
        df_features["std1"] - df_features["std2"]
    ).abs()

    df_features["ptp_diff"] = (
        df_features["ptp1"] - df_features["ptp2"]
    ).abs()

    # Feature Engineering: Wavelet
    df_features["wavelet_hf_avg"] = (
        df_features["wavelet_hf1"] + df_features["wavelet_hf2"]
    ) / 2

    df_features["wavelet_detail_avg"] = (
        df_features["wavelet_detail1"] + df_features["wavelet_detail2"]
    ) / 2

    df_features["wavelet_entropy_avg"] = (
        df_features["wavelet_entropy1"] + df_features["wavelet_entropy2"]
    ) / 2

    df_features["wavelet_maxabs_avg"] = (
        df_features["wavelet_maxabs1"] + df_features["wavelet_maxabs2"]
    ) / 2

    df_features["wavelet_kurtosis_avg"] = (
        df_features["wavelet_kurtosis1"] + df_features["wavelet_kurtosis2"]
    ) / 2

    return df_features


def calculate_segment_features(eeg1, fs):
    # =========================
    # SEGMENT ANALYSIS (Beginning vs Ending)
    # =========================

    segment_window_sec = 30 # Analysis-window in seconds (Beginning and Ending)
    samples = int(segment_window_sec * fs)

    eeg1_start = eeg1[:samples]
    eeg1_end = eeg1[-samples:]

    f_start, Pxx_start = welch(eeg1_start, fs, nperseg=min(1024, len(eeg1_start)//2))
    f_end, Pxx_end = welch(eeg1_end, fs, nperseg=min(1024, len(eeg1_end)//2))

    return {
        "segment_window_sec": segment_window_sec,
        "samples": samples,
        "f_start": f_start,
        "Pxx_start": Pxx_start,
        "f_end": f_end,
        "Pxx_end": Pxx_end,
    }
