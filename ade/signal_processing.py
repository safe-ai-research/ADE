# Resampling, preprocessing, and selectable signal-processing methods

import traceback

import mne
import numpy as np

from scipy.interpolate import interp1d
from scipy.signal import butter, detrend, filtfilt, iirnotch, welch

from .features import bandpower, get_wavelet_level, use_wavelet_available

mne.set_log_level("WARNING") # MNE-Return more calm

try:
    import asrpy
    use_asr_available = True
except ImportError:
    asrpy = None
    use_asr_available = False

try:
    from pyriemann.spatialfilters import AJDC
    use_ajdc_available = True
except ImportError:
    AJDC = None
    use_ajdc_available = False

try:
    import pywt
except ImportError:
    pywt = None


ALLOWED_SIGNAL_METHODS = [
    "original",
    "ica_ecg",
    "asr",
    "ajdc",
    "wavelet",
    "adaptive_ecg",
]


def validate_signal_method(signal_method):
    if signal_method not in ALLOWED_SIGNAL_METHODS:
        raise ValueError(
            f"Unbekanntes signal_method: {signal_method}. "
            f"Erlaubt: {ALLOWED_SIGNAL_METHODS}"
        )


def preprocess(
    signal,
    fs,
    *,
    use_detrend,
    use_dc,
    use_highpass,
    use_lowpass,
    use_notch,
    use_normalization,
):
    # Apply the unchanged classic preprocessing sequence
    # Remove Trend
    if use_detrend:
        signal = detrend(signal)

    # Remove DC
    if use_dc:
        signal = signal - np.mean(signal)

    # Highpass (0.5 Hz)
    def highpass(signal, fs, cutoff=0.5):
        b, a = butter(4, cutoff/(fs/2), btype='high')
        return filtfilt(b, a, signal)

    if use_highpass:
        signal = highpass(signal, fs, 0.5)  # Custom: Highpass Frequency (Default: 0.5)

    # Lowpass (30 Hz)
    def lowpass(signal, fs, cutoff=30):
        b, a = butter(4, cutoff/(fs/2), btype='low')
        return filtfilt(b, a, signal)

    if use_lowpass:
        signal = lowpass(signal, fs, 30)    # Custom: Lowpass Frequency (Default: 30)

    # Notch Mains Voltage (50 Hz)
    def notch_filter(signal, fs, freq=50, Q=30):
        b, a = iirnotch(freq/(fs/2), Q)
        return filtfilt(b, a, signal)

    if use_notch:
        signal = notch_filter(signal, fs, 50)   # Custom: Mains Voltage Frequency (Default: EU 50 / NA 60)

    # Normalization (z-score)
    def zscore(signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

    if use_normalization:
        signal = zscore(signal)

    return signal


def resample_and_preprocess(loaded, params):
    # Create the uniform time axis and preprocess both EEG channels
    df_eeg = loaded["df_eeg"]
    df_ecg = loaded["df_ecg"]
    time_col = loaded["time_col"]
    eeg1_col = loaded["eeg1_col"]
    eeg2_col = loaded["eeg2_col"]
    ecg_ii_col = loaded["ecg_ii_col"]
    fs_target = loaded["fs_target"]

    # --- Classic Filter ---
    use_dc = params.get("use_dc", True)                         # DC (mean)
    use_detrend = params.get("use_detrend", True)               # Trend
    use_notch = params.get("use_notch", False)                  # Notch-Filter (50 Hz, EU-Mains Voltage)
    use_highpass = params.get("use_highpass", True)             # High Pass (Default: 0.5 Hz)
    use_lowpass = params.get("use_lowpass", False)              # Low Pass (Default: 30 Hz)
    use_normalization = params.get("use_normalization", False)  # Normalization (z-score Standardization)

    # =========================
    # TIME → seconds
    # =========================

    t0 = df_eeg[time_col].iloc[0]

    t = (df_eeg[time_col] - t0).dt.total_seconds().values

    eeg1 = df_eeg[eeg1_col].values
    eeg2 = df_eeg[eeg2_col].values

    print("Samples EEG1:", len(eeg1))
    print("Samples EEG2:", len(eeg2))
    print("Same length:", len(eeg1) == len(eeg2))

    # =========================
    # RESAMPLING (fix fs)
    # =========================

    t_uniform = np.arange(t[0], t[-1], 1 / fs_target)

    interp1_f = interp1d(
        t, eeg1,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    interp2_f = interp1d(
        t, eeg2,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    eeg1 = interp1_f(t_uniform)
    eeg2 = interp2_f(t_uniform)

    # ECG_II bring to same time axis
    if df_ecg is not None and len(df_ecg) > 2:
        t_ecg = (df_ecg[time_col] - t0).dt.total_seconds().values
        ecg_ii = df_ecg[ecg_ii_col].values

        interp_ecg_f = interp1d(
            t_ecg,
            ecg_ii,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )

        ecg_ref = interp_ecg_f(t_uniform)
    else:
        ecg_ref = None

    t = t_uniform
    fs = fs_target

    print("\nResampled EEG1:", len(eeg1))
    print("Resampled EEG2:", len(eeg2))
    print("Same length:", len(eeg1) == len(eeg2))

    # =========================
    # BASIC PREPROCESSING
    # =========================

    eeg1 = preprocess(
        eeg1,
        fs,
        use_detrend=use_detrend,
        use_dc=use_dc,
        use_highpass=use_highpass,
        use_lowpass=use_lowpass,
        use_notch=use_notch,
        use_normalization=use_normalization,
    )
    eeg2 = preprocess(
        eeg2,
        fs,
        use_detrend=use_detrend,
        use_dc=use_dc,
        use_highpass=use_highpass,
        use_lowpass=use_lowpass,
        use_notch=use_notch,
        use_normalization=use_normalization,
    )

    # create MNE Raw
    if ecg_ref is not None:
        data_mne = np.vstack([eeg1, eeg2, ecg_ref])
        ch_names = ["EEG1", "EEG2", "ECG_II"]
        ch_types = ["eeg", "eeg", "ecg"]
    else:
        data_mne = np.vstack([eeg1, eeg2])
        ch_names = ["EEG1", "EEG2"]
        ch_types = ["eeg", "eeg"]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types=ch_types
    )

    raw_mne = mne.io.RawArray(data_mne, info)

    # Difference Signal (shows differences)
    diff = eeg1 - eeg2

    corr = np.corrcoef(eeg1, eeg2)[0,1]
    print(f"\nKorrelation EEG1 vs EEG2: {corr:.3f}")    # ~1.0 very close, ~0.5 partially similar, ~0 independent, neg opposite

    # Difference Signal by inversion of EEG2
    corr_original = np.corrcoef(eeg1, eeg2)[0,1]
    corr_flipped  = np.corrcoef(eeg1, -eeg2)[0,1]

    print(f"\nKorrelation EEG1 vs neg EEG2: {corr_flipped:.3f}")

    return {
        "t0": t0,
        "t": t,
        "fs": fs,
        "eeg1": eeg1,
        "eeg2": eeg2,
        "ecg_ref": ecg_ref,
        "raw_mne": raw_mne,
        "diff": diff,
        "corr": corr,
        "corr_original": corr_original,
        "corr_flipped": corr_flipped,
    }


# Adaptive ECG Filtering (NLMS)
def adaptive_ecg_filter_nlms(
    eeg,
    ecg_ref,
    filter_len=32,
    mu=0.05,
    eps=1e-8
):
    eeg = np.asarray(eeg, dtype=float)
    ecg = np.asarray(ecg_ref, dtype=float)

    n = min(len(eeg), len(ecg))
    eeg = eeg[:n]
    ecg = ecg[:n]

    # Normalizing ECG-Reference, so mu appears more stable
    ecg = ecg - np.mean(ecg)
    ecg = ecg / (np.std(ecg) + eps)

    # center EEG, but no z-score-standardization
    eeg_mean = np.mean(eeg)
    eeg_centered = eeg - eeg_mean

    w = np.zeros(filter_len)
    ecg_est = np.zeros(n)
    cleaned = np.zeros(n)

    for i in range(filter_len, n):
        x = ecg[i-filter_len:i][::-1]

        # estimated ECG-amount in EEG
        y = np.dot(w, x)

        # Error = EEG without estimated ECG-amount
        e = eeg_centered[i] - y

        # NLMS-Update
        norm_x = np.dot(x, x) + eps
        w = w + (mu / norm_x) * e * x

        ecg_est[i] = y
        cleaned[i] = e

    # Keep beginning without filter history close to original
    cleaned[:filter_len] = eeg_centered[:filter_len]

    return cleaned + eeg_mean, ecg_est


# Wavelet
def wavelet_denoise_signal(
    signal,
    wavelet_name="db4",
    level=5,
    mode="symmetric",
    threshold_mode="soft",
    threshold_scale=0.5
):
    if not use_wavelet_available:
        return signal.copy()

    signal = np.asarray(signal)

    level_use = get_wavelet_level(
        len(signal),
        wavelet_name,
        requested_level=level
    )

    if level_use < 1:
        return signal.copy()

    coeffs = pywt.wavedec(
        signal,
        wavelet=wavelet_name,
        mode=mode,
        level=level_use
    )

    # Noise Estimate via highest detail level cD1
    detail_high = coeffs[-1]
    sigma = np.median(np.abs(detail_high)) / 0.6745

    threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))

    coeffs_denoised = [coeffs[0]]

    for detail in coeffs[1:]:
        coeffs_denoised.append(
            pywt.threshold(
                detail,
                value=threshold,
                mode=threshold_mode
            )
        )

    reconstructed = pywt.waverec(
        coeffs_denoised,
        wavelet=wavelet_name,
        mode=mode
    )

    return reconstructed[:len(signal)]


def apply_signal_method(
    *,
    signal_method,
    eeg1,
    eeg2,
    ecg_ref,
    raw_mne,
    fs,
    params,
):
    # Run the selected signal method and return all existing status values
    validate_signal_method(signal_method)

    run_ica = signal_method == "ica_ecg"
    run_asr = signal_method == "asr" # ERROR!!!: internal TypeError in asrpy.fit_eeg_distribution
    run_ajdc = signal_method == "ajdc"
    run_wavelet = signal_method == "wavelet"
    run_adaptive_ecg = signal_method == "adaptive_ecg"

    # ICA
    use_ica_cleaned_signal = True if signal_method == "ica_ecg" else False  # use ICA Signal

    # ASR
    use_asr_cleaned_signal = True if signal_method == "asr" else False # use ASR Signal
    asr_cutoff = params.get("asr_cutoff", 20)                 # ASR-value (Default: 20)

    # AJDC
    use_adjc_cleaned_signal = True if signal_method == "ajdc" else False      # use AJDC Signal
    ajdc_fmin = params.get("ajdc_fmin", 1.0)                     # AJDC: avoid 0 Hz, since BSS expects zero-mean
    ajdc_fmax = params.get("ajdc_fmax", 30.0)
    ajdc_window_sec = params.get("ajdc_window_sec", 1.0)
    ajdc_overlap = params.get("ajdc_overlap", 0.5)
    ajdc_dim_red = params.get("ajdc_dim_red", {"n_components": 2})  # Dimension-Suppression (None, {"n_components": 2})

    # AJDC-Source Suppression
    ajdc_auto_suppress = params.get("ajdc_auto_suppress", False)    # Remove Auto Sources (False, if to inspect first)
    ajdc_manual_supp = params.get("ajdc_manual_supp", [])           # [] none, or [0] or [1], if a source is to be removed manually

    # AJDC-Heuristics, if ajdc_auto_suppress=True
    ajdc_remove_ecg = params.get("ajdc_remove_ecg", True)
    ajdc_remove_emg = params.get("ajdc_remove_emg", True)
    ajdc_ecg_corr_threshold = params.get("ajdc_ecg_corr_threshold", 0.20)
    ajdc_beta_rel_threshold = params.get("ajdc_beta_rel_threshold", 0.35)

    # Wavelet
    # Wavelet-Denoising
    wavelet_name = params.get("wavelet_name", "db4")    # "db4", "sym4", "coif3" (Default: db4) ; Al-Qazzaz et al.: "db7", "sym9"
    wavelet_threshold_scale = float(params.get("wavelet_threshold_scale", 0.1))    # 0.1 mild, 0.25 stronger, 0.5 aggressive, 1.0 highly aggressive (Default: 0.1)
    wavelet_level = params.get("wavelet_level", 5)
    wavelet_mode = params.get("wavelet_mode", "symmetric")
    wavelet_threshold_mode = params.get("wavelet_threshold_mode", "soft")

    # Adaptive ECG Filtering
    adaptive_ecg_filter_len = int(params.get("adaptive_ecg_filter_len", 32))    # filter_len: Length of adaptive FIR-Filter
    adaptive_ecg_mu = float(params.get("adaptive_ecg_mu", 0.05))                # mu: step width; smaller = careful, bigger = aggressive

    # =========================
    # SIGNAL SEPARATION - Defaults
    # =========================

    # --- ICA ---
    ica = None
    ecg_indices = []
    ecg_scores = []

    eeg1_ica_clean = eeg1.copy()
    eeg2_ica_clean = eeg2.copy()

    # --- ASR ---
    asr_success = False
    asr_error = None

    eeg1_asr_clean = eeg1.copy()
    eeg2_asr_clean = eeg2.copy()

    # --- AJDC ---
    ajdc = None
    ajdc_sources = None
    ajdc_source_metrics = []
    ajdc_supp = []
    ajdc_success = False
    ajdc_error = None
    ajdc_n_sources = 0

    eeg1_ajdc_clean = eeg1.copy()
    eeg2_ajdc_clean = eeg2.copy()

    # --- Adaptive ECG ---
    adaptive_ecg_success = False
    adaptive_ecg_error = None

    eeg1_adaptive_ecg_clean = eeg1.copy()
    eeg2_adaptive_ecg_clean = eeg2.copy()

    adaptive_ecg_corr_before_eeg1 = np.nan
    adaptive_ecg_corr_before_eeg2 = np.nan
    adaptive_ecg_corr_after_eeg1 = np.nan
    adaptive_ecg_corr_after_eeg2 = np.nan

    # =========================
    # SIGNAL SEPARATION - MNE ICA
    # =========================

    if run_ica:
        raw_for_ica = raw_mne.copy()

        # For ICA-Fit often useful: somewhat stronger Highpass than in the Analysis Pipeline
        raw_for_ica.filter(l_freq=1.0, h_freq=30.0) # Bandpass (Default: l_freq=1.0, h_freq=30.0)

        ica = mne.preprocessing.ICA(
            n_components=2,
            method="fastica",
            random_state=42,
            max_iter="auto"
        )

        # Important: only fit EEG-Channels, do not fit ECG_II also as source
        ica.fit(raw_for_ica, picks="eeg")

        # get components as Time Series
        ica_sources = ica.get_sources(raw_for_ica).get_data()

        ica_comp1 = ica_sources[0]
        ica_comp2 = ica_sources[1]

        print("ICA sources:", ica_sources.shape)

        # --- Component Detection: ECG ---
        if "ECG_II" in raw_for_ica.ch_names:
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw_for_ica,
                ch_name="ECG_II",
                method="correlation"
            )
        else:
            ecg_indices = []
            ecg_scores = []

        print("ECG-related ICA components:", ecg_indices)
        print("ECG scores:", ecg_scores)

        # # Optional: ICLabel
        # if use_icalabel_available:
        #     ic_labels = label_components(raw_for_ica, ica, method="iclabel")
        #     print(ic_labels["labels"])
        #     print(ic_labels["y_pred_proba"])

        # Remove ECG-component
        if len(ecg_indices) > 0:
            ica.exclude = ecg_indices

            raw_clean = raw_mne.copy()
            ica.apply(raw_clean)

            clean_data = raw_clean.get_data()

            eeg1_ica_clean = clean_data[0]
            eeg2_ica_clean = clean_data[1]

            print(f"ICA removed ECG components: {ecg_indices}")

        else:
            print("ICA: Keine ECG-Komponente gefunden, Signal bleibt unverändert.")

    else:
        print("ICA übersprungen.")

    # =========================
    # SIGNAL SEPARATION – ASR
    # =========================

    if run_asr and use_asr_available:
        try:
            # Only use EEG-Channels for ASR
            raw_asr = raw_mne.copy().pick_types(eeg=True, ecg=False)
            raw_asr.load_data()

            sfreq_asr = float(raw_asr.info["sfreq"])

            print("ASR input shape:", raw_asr.get_data().shape)
            print("ASR sfreq:", sfreq_asr)

            asr = asrpy.ASR(
                sfreq=sfreq_asr,
                cutoff=float(asr_cutoff),
                max_bad_chans=0.5,   # important for only 2 EEG-channels
                win_len=0.5,
                win_overlap=0.66
            )

            # optionally with returns of calibration data for debug
            clean_calib, sample_mask = asr.fit(
                raw_asr,
                picks="eeg",
                return_clean_window=True
            )

            print("ASR calibration data shape:", clean_calib.shape)
            print("ASR retained samples:", sample_mask.sum(), "/", sample_mask.size)

            raw_asr_clean = asr.transform(
                raw_asr,
                picks="eeg",
                maxdims=1,                       # bei 2 Kanälen max. 1 Dimension entfernen
                stepsize=max(1, int(sfreq_asr // 3))
            )

            asr_clean_data = raw_asr_clean.get_data()

            eeg1_asr_clean = asr_clean_data[0]
            eeg2_asr_clean = asr_clean_data[1]

            asr_success = True

            print(f"ASR: Reinigung durchgeführt. cutoff={asr_cutoff}")

        except Exception as e:
            asr_error = str(e)
            print(f"ASR: Fehler/übersprungen: {e}")
            traceback.print_exc()

            eeg1_asr_clean = eeg1.copy()
            eeg2_asr_clean = eeg2.copy()

    elif run_asr and not use_asr_available:
        asr_error = "asrpy not installed"
        print("ASR übersprungen: asrpy ist nicht installiert.")

    else:
        print("ASR übersprungen.")

    # =========================
    # SIGNAL SEPARATION – AJDC
    # =========================

    if run_ajdc and use_ajdc_available:
        try:
            # Only EEG-channels for AJDC
            X_ajdc = np.vstack([eeg1, eeg2])  # shape: channels x samples

            print("AJDC input shape:", X_ajdc.shape)

            ajdc = AJDC(
                window=int(ajdc_window_sec * fs),
                overlap=ajdc_overlap,
                fmin=ajdc_fmin,
                fmax=ajdc_fmax,
                fs=fs,
                dim_red=ajdc_dim_red,
                verbose=True
            )

            # AJDC expects for fit:
            # shape = subjects x conditions x channels x samples
            X_ajdc_fit = X_ajdc[np.newaxis, np.newaxis, :, :]

            ajdc.fit(X_ajdc_fit)

            # Transform to source dimension:
            # transform expects shape = matrices x channels x samples
            ajdc_sources = ajdc.transform(X_ajdc[np.newaxis, :, :])[0]

            ajdc_n_sources = ajdc_sources.shape[0]

            print("AJDC sources shape:", ajdc_sources.shape)
            print("AJDC n_sources:", ajdc_n_sources)

            # -------------------------
            # Calculate Source metrics
            # -------------------------
            ajdc_source_metrics = []

            def safe_corr(a, b):
                if b is None:
                    return np.nan

                if len(a) != len(b):
                    return np.nan

                if np.std(a) == 0 or np.std(b) == 0:
                    return np.nan

                return float(np.corrcoef(a, b)[0, 1])

            for src_idx in range(ajdc_n_sources):
                src = ajdc_sources[src_idx]

                f_src, Pxx_src = welch(
                    src,
                    fs,
                    nperseg=min(1024, len(src) // 2)
                )

                total_src = bandpower(f_src, Pxx_src, 0.5, 30)
                if total_src == 0:
                    total_src = 1e-10

                delta_rel = bandpower(f_src, Pxx_src, 0.5, 4) / total_src
                theta_rel = bandpower(f_src, Pxx_src, 4, 8) / total_src
                alpha_rel = bandpower(f_src, Pxx_src, 8, 13) / total_src
                beta_rel  = bandpower(f_src, Pxx_src, 13, 30) / total_src

                corr_ecg = safe_corr(src, ecg_ref)

                src_metrics = {
                    "source": src_idx,
                    "delta_rel": float(delta_rel),
                    "theta_rel": float(theta_rel),
                    "alpha_rel": float(alpha_rel),
                    "beta_rel": float(beta_rel),
                    "var": float(np.var(src)),
                    "corr_ecg": corr_ecg,
                }

                ajdc_source_metrics.append(src_metrics)

                print(
                    f"AJDC source {src_idx}: "
                    f"delta={delta_rel:.3f}, "
                    f"theta={theta_rel:.3f}, "
                    f"alpha={alpha_rel:.3f}, "
                    f"beta={beta_rel:.3f}, "
                    f"var={np.var(src):.3f}, "
                    f"corr_ecg={corr_ecg:.3f}"
                )

            # -------------------------
            # determine Sources for Removal
            # -------------------------
            ajdc_supp = list(ajdc_manual_supp)

            if ajdc_auto_suppress:
                for src_metrics in ajdc_source_metrics:
                    src_idx = src_metrics["source"]

                    # ECG-similar Source
                    if (
                        ajdc_remove_ecg and
                        not np.isnan(src_metrics["corr_ecg"]) and
                        abs(src_metrics["corr_ecg"]) > ajdc_ecg_corr_threshold
                    ):
                        ajdc_supp.append(src_idx)

                    # EMG-similar Source
                    if (
                        ajdc_remove_emg and
                        src_metrics["beta_rel"] > ajdc_beta_rel_threshold
                    ):
                        ajdc_supp.append(src_idx)

            ajdc_supp = sorted(set(ajdc_supp))

            print("AJDC suppressed sources:", ajdc_supp)

            # -------------------------
            # Back Transformation / Reconstruction
            # -------------------------
            if len(ajdc_supp) > 0:
                ajdc_clean_data = ajdc.inverse_transform(
                    ajdc_sources[np.newaxis, :, :],
                    supp=ajdc_supp
                )[0]

                print(f"AJDC: Quellen entfernt: {ajdc_supp}")

            else:
                ajdc_clean_data = ajdc.inverse_transform(
                    ajdc_sources[np.newaxis, :, :],
                    supp=None
                )[0]

                print("AJDC: No Sources removed. Reconstructing with all Sources.")

            eeg1_ajdc_clean = ajdc_clean_data[0]
            eeg2_ajdc_clean = ajdc_clean_data[1]

            ajdc_success = True

        except Exception as e:
            ajdc_error = str(e)
            print(f"AJDC: Error/skipped: {e}")
            traceback.print_exc()

            eeg1_ajdc_clean = eeg1.copy()
            eeg2_ajdc_clean = eeg2.copy()

    elif run_ajdc and not use_ajdc_available:
        ajdc_error = "pyriemann not installed"
        print("AJDC skipped: pyriemann is not installed.")

    else:
        print("AJDC skipped.")

    # =========================
    # Reference-based – ADAPTIVE ECG FILTERING
    # =========================

    if run_adaptive_ecg:
        if ecg_ref is None:
            adaptive_ecg_error = "No ECG reference available"
            print("Adaptive ECG übersprungen: keine ECG-Referenz vorhanden.")

        else:
            try:
                # Correlation pre filtering
                if np.std(ecg_ref) > 0:
                    adaptive_ecg_corr_before_eeg1 = float(np.corrcoef(eeg1, ecg_ref)[0, 1])
                    adaptive_ecg_corr_before_eeg2 = float(np.corrcoef(eeg2, ecg_ref)[0, 1])

                eeg1_adaptive_ecg_clean, ecg_est_1 = adaptive_ecg_filter_nlms(
                    eeg1,
                    ecg_ref,
                    filter_len=adaptive_ecg_filter_len,
                    mu=adaptive_ecg_mu
                )

                eeg2_adaptive_ecg_clean, ecg_est_2 = adaptive_ecg_filter_nlms(
                    eeg2,
                    ecg_ref,
                    filter_len=adaptive_ecg_filter_len,
                    mu=adaptive_ecg_mu
                )

                # Länge absichern
                eeg1_adaptive_ecg_clean = eeg1_adaptive_ecg_clean[:len(eeg1)]
                eeg2_adaptive_ecg_clean = eeg2_adaptive_ecg_clean[:len(eeg2)]

                # Korrelation nach Filterung
                if np.std(ecg_ref) > 0:
                    adaptive_ecg_corr_after_eeg1 = float(
                        np.corrcoef(eeg1_adaptive_ecg_clean, ecg_ref)[0, 1]
                    )
                    adaptive_ecg_corr_after_eeg2 = float(
                        np.corrcoef(eeg2_adaptive_ecg_clean, ecg_ref)[0, 1]
                    )

                adaptive_ecg_success = True

                print(
                    "Adaptive ECG-Filterung durchgeführt: "
                    f"filter_len={adaptive_ecg_filter_len}, mu={adaptive_ecg_mu}"
                )

                print(
                    "ECG-Korrelation EEG1 vor/nach: "
                    f"{adaptive_ecg_corr_before_eeg1:.4f} -> {adaptive_ecg_corr_after_eeg1:.4f}"
                )

                print(
                    "ECG-Korrelation EEG2 vor/nach: "
                    f"{adaptive_ecg_corr_before_eeg2:.4f} -> {adaptive_ecg_corr_after_eeg2:.4f}"
                )

            except Exception as e:
                adaptive_ecg_error = str(e)
                print(f"Adaptive ECG: Fehler/übersprungen: {e}")
                traceback.print_exc()

                eeg1_adaptive_ecg_clean = eeg1.copy()
                eeg2_adaptive_ecg_clean = eeg2.copy()

    else:
        print("Adaptive ECG-Filterung übersprungen.")

    # =========================
    # TIME-FREQUENCY – WAVELET DENOISING
    # =========================

    eeg1_wavelet_clean = eeg1.copy()
    eeg2_wavelet_clean = eeg2.copy()

    wavelet_success = False
    wavelet_error = None

    if run_wavelet and use_wavelet_available:
        try:
            eeg1_wavelet_clean = wavelet_denoise_signal(
                eeg1,
                wavelet_name=wavelet_name,
                level=wavelet_level,
                mode=wavelet_mode,
                threshold_mode=wavelet_threshold_mode,
                threshold_scale=wavelet_threshold_scale
            )

            eeg2_wavelet_clean = wavelet_denoise_signal(
                eeg2,
                wavelet_name=wavelet_name,
                level=wavelet_level,
                mode=wavelet_mode,
                threshold_mode=wavelet_threshold_mode,
                threshold_scale=wavelet_threshold_scale
            )

            wavelet_success = True

            print(
                f"Wavelet-Denoising durchgeführt: "
                f"{wavelet_name}, level={wavelet_level}, "
                f"threshold_scale={wavelet_threshold_scale}"
            )

        except Exception as e:
            wavelet_error = str(e)
            print(f"Wavelet: Fehler/übersprungen: {e}")
            traceback.print_exc()

            eeg1_wavelet_clean = eeg1.copy()
            eeg2_wavelet_clean = eeg2.copy()

    elif run_wavelet and not use_wavelet_available:
        wavelet_error = "pywt / PyWavelets not installed"
        print("Wavelet übersprungen: PyWavelets ist nicht installiert.")

    else:
        print("Wavelet-Denoising übersprungen.")

    # =========================
    # CHOOSE SIGNAL FOR FEATURE-CALCULATION
    # =========================

    if signal_method == "original":
        signal1_for_features = eeg1
        signal2_for_features = eeg2
        print("Features werden aus Original-EEG berechnet.")

    elif signal_method == "ica_ecg":
        if run_ica:
            signal1_for_features = eeg1_ica_clean
            signal2_for_features = eeg2_ica_clean
            print("Features werden aus ICA-cleaned EEG berechnet.")
        else:
            signal1_for_features = eeg1
            signal2_for_features = eeg2
            print("Hinweis: signal_method='ica_ecg', aber run_ica=False. Nutze Originalsignal.")

    elif signal_method == "asr":
        if run_asr and use_asr_available and asr_success:
            signal1_for_features = eeg1_asr_clean
            signal2_for_features = eeg2_asr_clean
            print("Features werden aus ASR-cleaned EEG berechnet.")
        else:
            signal1_for_features = eeg1
            signal2_for_features = eeg2
            print("Hinweis: signal_method='asr', aber ASR nicht erfolgreich/verfügbar. Nutze Originalsignal.")

    elif signal_method == "ajdc":
        if run_ajdc and use_ajdc_available and ajdc_success:
            signal1_for_features = eeg1_ajdc_clean
            signal2_for_features = eeg2_ajdc_clean
            print("Features werden aus AJDC-rekonstruiertem EEG berechnet.")
        else:
            signal1_for_features = eeg1
            signal2_for_features = eeg2
            print("Hinweis: signal_method='ajdc', aber AJDC nicht erfolgreich/verfügbar. Nutze Originalsignal.")
    elif signal_method == "wavelet":
        if run_wavelet and use_wavelet_available and wavelet_success:
            signal1_for_features = eeg1_wavelet_clean
            signal2_for_features = eeg2_wavelet_clean
            print("Features werden aus Wavelet-denoised EEG berechnet.")
        else:
            signal1_for_features = eeg1
            signal2_for_features = eeg2
            print("Hinweis: signal_method='wavelet', aber Wavelet nicht erfolgreich/verfügbar. Nutze Originalsignal.")
    elif signal_method == "adaptive_ecg":
        if run_adaptive_ecg and adaptive_ecg_success:
            signal1_for_features = eeg1_adaptive_ecg_clean
            signal2_for_features = eeg2_adaptive_ecg_clean
            print("Features werden aus adaptive-ECG-cleaned EEG berechnet.")
        else:
            signal1_for_features = eeg1
            signal2_for_features = eeg2
            print("Hinweis: adaptive_ecg nicht erfolgreich/verfügbar. Nutze Originalsignal.")

    else:
        raise ValueError(f"Unbekanntes signal_method: {signal_method}")

    selected_signal_method_success = (
        signal_method == "original"
        or (signal_method == "ica_ecg" and ica is not None)
        or (signal_method == "asr" and asr_success)
        or (signal_method == "ajdc" and ajdc_success)
        or (signal_method == "wavelet" and wavelet_success)
        or (
            signal_method == "adaptive_ecg"
            and adaptive_ecg_success
        )
    )

    return {
        "signal1_for_features": signal1_for_features,
        "signal2_for_features": signal2_for_features,
        "selected_signal_method_success": selected_signal_method_success,
        "run_ica": run_ica,
        "run_asr": run_asr,
        "run_ajdc": run_ajdc,
        "run_wavelet": run_wavelet,
        "run_adaptive_ecg": run_adaptive_ecg,
        "ica": ica,
        "ecg_indices": ecg_indices,
        "ecg_scores": ecg_scores,
        "asr_cutoff": asr_cutoff,
        "asr_success": asr_success,
        "asr_error": asr_error,
        "ajdc_success": ajdc_success,
        "ajdc_error": ajdc_error,
        "ajdc_n_sources": ajdc_n_sources,
        "ajdc_supp": ajdc_supp,
        "adaptive_ecg_success": adaptive_ecg_success,
        "adaptive_ecg_error": adaptive_ecg_error,
        "adaptive_ecg_filter_len": adaptive_ecg_filter_len,
        "adaptive_ecg_mu": adaptive_ecg_mu,
        "adaptive_ecg_corr_before_eeg1": adaptive_ecg_corr_before_eeg1,
        "adaptive_ecg_corr_before_eeg2": adaptive_ecg_corr_before_eeg2,
        "adaptive_ecg_corr_after_eeg1": adaptive_ecg_corr_after_eeg1,
        "adaptive_ecg_corr_after_eeg2": adaptive_ecg_corr_after_eeg2,
        "wavelet_success": wavelet_success,
        "wavelet_error": wavelet_error,
        "wavelet_name": wavelet_name,
        "wavelet_level": wavelet_level,
        "wavelet_mode": wavelet_mode,
        "wavelet_threshold_mode": wavelet_threshold_mode,
        "wavelet_threshold_scale": wavelet_threshold_scale,
    }
