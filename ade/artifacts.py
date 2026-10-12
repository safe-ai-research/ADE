# AutoReject integration and rule-based artifact-marker assignment

import traceback

import mne
import numpy as np
import pandas as pd

from .features import use_wavelet_available

try:
    from autoreject import get_rejection_threshold
    use_autoreject_available = True
except ImportError:
    get_rejection_threshold = None
    use_autoreject_available = False


ARTIFACT_FLAG_COLUMNS = {
    "SLOW_CORR": "has_slow_corr",
    "EMG": "has_emg",
    "ECG": "has_ecg",
    "BAD_AUTO": "has_bad_auto",
    "WAVELET_HF": "has_wavelet_hf",
    "WAVELET_TRANSIENT": "has_wavelet_transient",
    "AMP_BURST": "has_amp_burst",
    "JUMP": "has_jump",
    "FLATLINE": "has_flatline",
    "HF_EMG": "has_hf_emg",
    "LOW_DRIFT": "has_low_drift",
    "CLIPPING": "has_clipping",
}


ARTIFACT_CHANNEL_COLUMNS = {
    "EMG_EEG1": "has_emg_eeg1",
    "EMG_EEG2": "has_emg_eeg2",

    "ECG_EEG1": "has_ecg_eeg1",
    "ECG_EEG2": "has_ecg_eeg2",

    "BAD_AUTO_EEG1": "has_bad_auto_eeg1",
    "BAD_AUTO_EEG2": "has_bad_auto_eeg2",

    "WAVELET_HF_EEG1": "has_wavelet_hf_eeg1",
    "WAVELET_HF_EEG2": "has_wavelet_hf_eeg2",

    "WAVELET_TRANSIENT_EEG1": "has_wavelet_transient_eeg1",
    "WAVELET_TRANSIENT_EEG2": "has_wavelet_transient_eeg2",

    "AMP_BURST_EEG1": "has_amp_burst_eeg1",
    "AMP_BURST_EEG2": "has_amp_burst_eeg2",

    "JUMP_EEG1": "has_jump_eeg1",
    "JUMP_EEG2": "has_jump_eeg2",

    "FLATLINE_EEG1": "has_flatline_eeg1",
    "FLATLINE_EEG2": "has_flatline_eeg2",

    "HF_EMG_EEG1": "has_hf_emg_eeg1",
    "HF_EMG_EEG2": "has_hf_emg_eeg2",

    "LOW_DRIFT_EEG1": "has_low_drift_eeg1",
    "LOW_DRIFT_EEG2": "has_low_drift_eeg2",

    "CLIPPING_EEG1": "has_clipping_eeg1",
    "CLIPPING_EEG2": "has_clipping_eeg2",
}


def pooled_quantile(df_features, column1, column2, quantile):
    values = pd.concat(
        [
            df_features[column1],
            df_features[column2]
        ],
        ignore_index=True
    )

    return values.quantile(quantile)


def run_autoreject_detection(windows_eeg1, windows_eeg2, fs, params):
    # --- Autoreject ---
    run_autoreject = params.get("run_autoreject", True)
    autoreject_random_state = int(
        params.get("autoreject_random_state", 42)
    )
    autoreject_threshold_eeg_override = params.get(
        "autoreject_threshold_eeg_override"
    )
    if autoreject_threshold_eeg_override in (None, ""):
        autoreject_threshold_eeg_override = None
    else:
        autoreject_threshold_eeg_override = float(
            autoreject_threshold_eeg_override
        )
        if (
            not np.isfinite(autoreject_threshold_eeg_override)
            or autoreject_threshold_eeg_override <= 0.0
        ):
            raise ValueError(
                "autoreject_threshold_eeg_override must be a finite "
                "number greater than zero."
            )
    autoreject_threshold_override_source = str(
        params.get("autoreject_threshold_override_source", "")
    )
    # autoreject_window_sec = params.get("autoreject_window_sec", 5)
    # autoreject_overlap = params.get("autoreject_overlap", 0.5)

    # =========================
    # ARTIFACT DETECTION – AUTOREJECT
    # =========================

    autoreject_success = False
    autoreject_error = None
    autoreject_threshold_eeg = None
    autoreject_threshold_mode = (
        "disabled"
        if not run_autoreject
        else (
            "fixed_override"
            if autoreject_threshold_eeg_override is not None
            else "recording_adaptive"
        )
    )
    autoreject_bad_eeg1 = np.zeros(len(windows_eeg1), dtype=bool)
    autoreject_bad_eeg2 = np.zeros(len(windows_eeg2), dtype=bool)
    autoreject_bad_windows = (autoreject_bad_eeg1 | autoreject_bad_eeg2)

    if run_autoreject and (
        use_autoreject_available
        or autoreject_threshold_eeg_override is not None
    ):
        try:
            # convert own Pipeline-windows to MNE-EpochsArray
            # shape needed: epochs x channels x samples
            epochs_data_ar = np.stack(
                [windows_eeg1, windows_eeg2],
                axis=1
            )

            info_ar = mne.create_info(
                ch_names=["EEG1", "EEG2"],
                sfreq=fs,
                ch_types=["eeg", "eeg"]
            )

            epochs_ar = mne.EpochsArray(
                epochs_data_ar,
                info_ar,
                tmin=0,
                baseline=None,
                verbose=False
            )

            print("Autoreject input epochs shape:", epochs_data_ar.shape)
            print("Autoreject epochs:", len(epochs_ar))

            if autoreject_threshold_eeg_override is None:
                reject_thresholds = get_rejection_threshold(
                    epochs_ar,
                    ch_types="eeg",
                    random_state=autoreject_random_state,
                )
            else:
                reject_thresholds = {
                    "eeg": autoreject_threshold_eeg_override
                }

            print("Autoreject thresholds:", reject_thresholds)
            print("Autoreject threshold mode:", autoreject_threshold_mode)

            autoreject_threshold_eeg = reject_thresholds.get("eeg", None)

            if autoreject_threshold_eeg is not None:
                autoreject_threshold_eeg = float(autoreject_threshold_eeg)

            # Manual Application of global Peak-to-Peak-Thresholds
            p2p = np.ptp(epochs_data_ar, axis=2)  # epochs x channels

            if autoreject_threshold_eeg is not None:
                autoreject_bad_eeg1 = (
                    p2p[:, 0] > autoreject_threshold_eeg
                )

                autoreject_bad_eeg2 = (
                    p2p[:, 1] > autoreject_threshold_eeg
                )

                autoreject_bad_windows = (
                    autoreject_bad_eeg1 |
                    autoreject_bad_eeg2
                )

            else:
                autoreject_bad_eeg1 = np.zeros(
                    len(windows_eeg1),
                    dtype=bool
                )

                autoreject_bad_eeg2 = np.zeros(
                    len(windows_eeg2),
                    dtype=bool
                )

                autoreject_bad_windows = (
                    autoreject_bad_eeg1 |
                    autoreject_bad_eeg2
                )

            autoreject_success = True

            print(
                "Autoreject bad windows:",
                int(autoreject_bad_windows.sum()),
                "/",
                len(autoreject_bad_windows)
            )

        except Exception as e:
            autoreject_error = str(e)
            print(f"Autoreject: Error/skipped: {e}")
            traceback.print_exc()

    elif run_autoreject and not use_autoreject_available:
        autoreject_error = "autoreject not installed"
        print("Autoreject skipped: Package is not installed.")

    else:
        print("Autoreject skipped.")

    return {
        "run_autoreject": run_autoreject,
        "autoreject_random_state": autoreject_random_state,
        "autoreject_threshold_eeg_override": autoreject_threshold_eeg_override,
        "autoreject_threshold_override_source": autoreject_threshold_override_source,
        "autoreject_success": autoreject_success,
        "autoreject_error": autoreject_error,
        "autoreject_threshold_eeg": autoreject_threshold_eeg,
        "autoreject_threshold_mode": autoreject_threshold_mode,
        "autoreject_bad_eeg1": autoreject_bad_eeg1,
        "autoreject_bad_eeg2": autoreject_bad_eeg2,
        "autoreject_bad_windows": autoreject_bad_windows,
    }


def apply_artifact_markers(df_features, params):
    # Apply the unchanged rule-based artifact-marker definitions
    # Artifact / Marker Thresholds
    threshold_delta_slowcorr = params.get("threshold_delta_slowcorr", 0.60)     # relative Delta-amount
    threshold_corr_slowcorr = params.get("threshold_corr_slowcorr", 0.30)       # Channel Correlation
    threshold_beta_emg_rel = params.get("threshold_beta_emg_rel", 0.25)         # relative Beta-amount
    threshold_var_emg_quantile = params.get("threshold_var_emg_quantile", 0.75) # Variance-threshold for EMG

    # Additional robust artifact marker
    use_robust_artifact_markers = params.get("use_robust_artifact_markers", True)

    threshold_amp_burst_quantile = params.get("threshold_amp_burst_quantile", 0.95) # AMP_Burst
    threshold_jump_quantile = params.get("threshold_jump_quantile", 0.95)           # Jump
    threshold_flatline_quantile = params.get("threshold_flatline_quantile", 0.01)   # Flatline

    threshold_hf_emg_rel = params.get("threshold_hf_emg_rel", 0.15)                 # HF_EMG_rel
    threshold_hf_emg_line_quantile = params.get("threshold_hf_emg_line_quantile", 0.75) # HF_EMG_line_quantile

    threshold_low_drift_delta_rel = params.get("threshold_low_drift_delta_rel", 0.80)   # low_drift_delta_rel
    threshold_low_drift_slope_quantile = params.get("threshold_low_drift_slope_quantile", 0.90) # low_drift_slope_quantile

    threshold_clipping_ratio = params.get("threshold_clipping_ratio", 0.02)         # Clipping

    # LOW_DRIFT is by default soft, because slow activity could be physiological
    low_drift_is_hard_artifact = params.get("low_drift_is_hard_artifact", False)    # set low_drift as hard artifact?

    # Wavelet-Features / Markers
    use_wavelet_marker_features = params.get("use_wavelet_marker_features", True)

    threshold_wavelet_hf_quantile = params.get("threshold_wavelet_hf_quantile", 0.90)
    threshold_wavelet_transient_quantile = params.get("threshold_wavelet_transient_quantile", 0.90)

    wavelet_markers_are_hard_artifacts = params.get("wavelet_markers_are_hard_artifacts", False)

    # Thresholds
    threshold_var_emg = pooled_quantile(
        df_features,
        "var1",
        "var2",
        threshold_var_emg_quantile
    )

    if use_wavelet_marker_features and use_wavelet_available:
        threshold_wavelet_hf = pooled_quantile(
            df_features,
            "wavelet_hf1",
            "wavelet_hf2",
            threshold_wavelet_hf_quantile
        )

        threshold_wavelet_transient = pooled_quantile(
            df_features,
            "wavelet_maxabs1",
            "wavelet_maxabs2",
            threshold_wavelet_transient_quantile
        )

    else:
        threshold_wavelet_hf = np.inf
        threshold_wavelet_transient = np.inf

    # Thresholds for robust Artifact Marker
    if use_robust_artifact_markers:
        threshold_amp_burst = pooled_quantile(
            df_features,
            "ptp1",
            "ptp2",
            threshold_amp_burst_quantile
        )

        threshold_jump = pooled_quantile(
            df_features,
            "max_jump_norm1",
            "max_jump_norm2",
            threshold_jump_quantile
        )

        threshold_flatline_std = pooled_quantile(
            df_features,
            "std1",
            "std2",
            threshold_flatline_quantile
        )

        threshold_flatline_ptp = pooled_quantile(
            df_features,
            "ptp1",
            "ptp2",
            threshold_flatline_quantile
        )

        threshold_hf_emg_line = pooled_quantile(
            df_features,
            "line_length_norm1",
            "line_length_norm2",
            threshold_hf_emg_line_quantile
        )

        threshold_low_drift_slope = pooled_quantile(
            df_features,
            "trend_slope_norm1",
            "trend_slope_norm2",
            threshold_low_drift_slope_quantile
        )
    else:
        threshold_amp_burst = np.inf
        threshold_jump = np.inf
        threshold_flatline_std = -np.inf
        threshold_flatline_ptp = -np.inf
        threshold_hf_emg_line = np.inf
        threshold_low_drift_slope = np.inf

    df_features["has_slow_corr"] = (
        (df_features["delta_rel_avg"] > threshold_delta_slowcorr) &
        (df_features["corr"].abs() > threshold_corr_slowcorr)
    )
    df_features["has_emg_eeg1"] = (
        (df_features["beta1_rel"] > threshold_beta_emg_rel) &
        (df_features["var1"] > threshold_var_emg)
    )
    df_features["has_emg_eeg2"] = (
        (df_features["beta2_rel"] > threshold_beta_emg_rel) &
        (df_features["var2"] > threshold_var_emg)
    )
    df_features["has_emg"] = (
        df_features["has_emg_eeg1"] | df_features["has_emg_eeg2"]
    )
    df_features["has_ecg_eeg1"] = df_features["ecg_eeg1"]
    df_features["has_ecg_eeg2"] = df_features["ecg_eeg2"]
    df_features["has_ecg"] = (
        df_features["has_ecg_eeg1"] | df_features["has_ecg_eeg2"]
    )
    df_features["has_bad_auto_eeg1"] = (
        df_features["autoreject_bad_eeg1"]
    )
    df_features["has_bad_auto_eeg2"] = (
        df_features["autoreject_bad_eeg2"]
    )
    df_features["has_bad_auto"] = (
        df_features["has_bad_auto_eeg1"] | df_features["has_bad_auto_eeg2"]
    )
    df_features["has_wavelet_hf_eeg1"] = (
        use_wavelet_marker_features &
        (df_features["wavelet_hf1"] > threshold_wavelet_hf)
    )
    df_features["has_wavelet_hf_eeg2"] = (
        use_wavelet_marker_features &
        (df_features["wavelet_hf2"] > threshold_wavelet_hf)
    )
    df_features["has_wavelet_hf"] = (
        df_features["has_wavelet_hf_eeg1"] | df_features["has_wavelet_hf_eeg2"]
    )
    df_features["has_wavelet_transient_eeg1"] = (
        use_wavelet_marker_features &
        (df_features["wavelet_maxabs1"] > threshold_wavelet_transient)
    )
    df_features["has_wavelet_transient_eeg2"] = (
        use_wavelet_marker_features &
        (df_features["wavelet_maxabs2"] > threshold_wavelet_transient)
    )
    df_features["has_wavelet_transient"] = (
        df_features["has_wavelet_transient_eeg1"] | df_features["has_wavelet_transient_eeg2"]
    )
    df_features["has_amp_burst_eeg1"] = (
        df_features["ptp1"] > threshold_amp_burst
    )
    df_features["has_amp_burst_eeg2"] = (
        df_features["ptp2"] > threshold_amp_burst
    )
    df_features["has_amp_burst"] = (
        df_features["has_amp_burst_eeg1"] | df_features["has_amp_burst_eeg2"]
    )
    df_features["has_jump_eeg1"] = (
        df_features["max_jump_norm1"] > threshold_jump
    )
    df_features["has_jump_eeg2"] = (
        df_features["max_jump_norm2"] > threshold_jump
    )
    df_features["has_jump"] = (
        df_features["has_jump_eeg1"] | df_features["has_jump_eeg2"]
    )
    df_features["has_flatline_eeg1"] = (
        (df_features["std1"] <= threshold_flatline_std) | (df_features["ptp1"] <= threshold_flatline_ptp)
    )
    df_features["has_flatline_eeg2"] = (
        (df_features["std2"] <= threshold_flatline_std) | (df_features["ptp2"] <= threshold_flatline_ptp)
    )
    df_features["has_flatline"] = (
        df_features["has_flatline_eeg1"] | df_features["has_flatline_eeg2"]
    )
    df_features["has_hf_emg_eeg1"] = (
        (df_features["hf_rel1"] > threshold_hf_emg_rel) &
        (df_features["line_length_norm1"] > threshold_hf_emg_line)
    )
    df_features["has_hf_emg_eeg2"] = (
        (df_features["hf_rel2"] > threshold_hf_emg_rel) &
        (df_features["line_length_norm2"] > threshold_hf_emg_line)
    )
    df_features["has_hf_emg"] = (
        df_features["has_hf_emg_eeg1"] | df_features["has_hf_emg_eeg2"]
    )
    df_features["has_low_drift_eeg1"] = (
        (df_features["delta1_rel"] > threshold_low_drift_delta_rel) &
        (df_features["trend_slope_norm1"] > threshold_low_drift_slope)
    )
    df_features["has_low_drift_eeg2"] = (
        (df_features["delta2_rel"] > threshold_low_drift_delta_rel) &
        (df_features["trend_slope_norm2"] > threshold_low_drift_slope)
    )
    df_features["has_low_drift"] = (
        df_features["has_low_drift_eeg1"] | df_features["has_low_drift_eeg2"]
    )
    df_features["has_clipping_eeg1"] = (
        use_robust_artifact_markers &
        (df_features["clipping_ratio1"] > threshold_clipping_ratio)
    )
    df_features["has_clipping_eeg2"] = (
        use_robust_artifact_markers &
        (df_features["clipping_ratio2"] > threshold_clipping_ratio)
    )
    df_features["has_clipping"] = (
        df_features["has_clipping_eeg1"] | df_features["has_clipping_eeg2"]
    )

    artifact_flag_columns = ARTIFACT_FLAG_COLUMNS

    def collect_artifact_types(row):
        labels = [
            marker
            for marker, column in artifact_flag_columns.items()
            if bool(row[column])
        ]

        if not labels:
            return ["NO_MARKER"]

        return labels

    df_features["artifact_type"] = df_features.apply(
        collect_artifact_types,
        axis=1
    )
    df_features["is_strict_clean"] = (
        df_features["artifact_type"].apply(lambda x: x == ["NO_MARKER"])
    )
    df_features["is_mixed"] = (
        df_features["artifact_type"].apply(lambda x: len(x) > 1)
    )

    # Hard Artifacts:
    # SLOW_CORR deliberately doesn't count as hard artifact,
    # because slow correlated EEG can by physiological for EEG sedation
    df_features["has_hard_artifact_eeg1"] = (
        df_features["has_emg_eeg1"] |
        df_features["has_hf_emg_eeg1"] |
        df_features["has_ecg_eeg1"] |
        df_features["has_bad_auto_eeg1"] |
        df_features["has_amp_burst_eeg1"] |
        df_features["has_jump_eeg1"] |
        df_features["has_flatline_eeg1"] |
        df_features["has_clipping_eeg1"]
    )

    df_features["has_hard_artifact_eeg2"] = (
        df_features["has_emg_eeg2"] |
        df_features["has_hf_emg_eeg2"] |
        df_features["has_ecg_eeg2"] |
        df_features["has_bad_auto_eeg2"] |
        df_features["has_amp_burst_eeg2"] |
        df_features["has_jump_eeg2"] |
        df_features["has_flatline_eeg2"] |
        df_features["has_clipping_eeg2"]
    )

    if low_drift_is_hard_artifact:
        df_features["has_hard_artifact_eeg1"] |= (
            df_features["has_low_drift_eeg1"]
        )

        df_features["has_hard_artifact_eeg2"] |= (
            df_features["has_low_drift_eeg2"]
        )

    if wavelet_markers_are_hard_artifacts:
        df_features["has_hard_artifact_eeg1"] |= (
            df_features["has_wavelet_hf_eeg1"] | df_features["has_wavelet_transient_eeg1"]
        )

        df_features["has_hard_artifact_eeg2"] |= (
            df_features["has_wavelet_hf_eeg2"] | df_features["has_wavelet_transient_eeg2"]
        )

    # Convenient Clean:
    # clean = no hard artifact
    # SLOW_CORR may be included
    df_features["has_hard_artifact"] = (
        df_features["has_hard_artifact_eeg1"] |
        df_features["has_hard_artifact_eeg2"]
    )

    df_features["has_no_hard_artifact_eeg1"] = (
        ~df_features["has_hard_artifact_eeg1"]
    )

    df_features["has_no_hard_artifact_eeg2"] = (
        ~df_features["has_hard_artifact_eeg2"]
    )

    df_features["has_no_hard_artifact"] = (
        ~df_features["has_hard_artifact"]
    )

    thresholds = {
        "threshold_delta_slowcorr": threshold_delta_slowcorr,
        "threshold_corr_slowcorr": threshold_corr_slowcorr,
        "threshold_beta_emg_rel": threshold_beta_emg_rel,
        "threshold_var_emg_quantile": threshold_var_emg_quantile,
        "threshold_var_emg": threshold_var_emg,
        "threshold_wavelet_hf_quantile": threshold_wavelet_hf_quantile,
        "threshold_wavelet_hf": threshold_wavelet_hf,
        "threshold_wavelet_transient_quantile": threshold_wavelet_transient_quantile,
        "threshold_wavelet_transient": threshold_wavelet_transient,
        "threshold_amp_burst_quantile": threshold_amp_burst_quantile,
        "threshold_amp_burst": threshold_amp_burst,
        "threshold_jump_quantile": threshold_jump_quantile,
        "threshold_jump": threshold_jump,
        "threshold_flatline_quantile": threshold_flatline_quantile,
        "threshold_flatline_std": threshold_flatline_std,
        "threshold_flatline_ptp": threshold_flatline_ptp,
        "threshold_hf_emg_rel": threshold_hf_emg_rel,
        "threshold_hf_emg_line_quantile": threshold_hf_emg_line_quantile,
        "threshold_hf_emg_line": threshold_hf_emg_line,
        "threshold_low_drift_delta_rel": threshold_low_drift_delta_rel,
        "threshold_low_drift_slope_quantile": threshold_low_drift_slope_quantile,
        "threshold_low_drift_slope": threshold_low_drift_slope,
        "threshold_clipping_ratio": threshold_clipping_ratio,
        "use_robust_artifact_markers": use_robust_artifact_markers,
        "use_wavelet_marker_features": use_wavelet_marker_features,
        "wavelet_markers_are_hard_artifacts": wavelet_markers_are_hard_artifacts,
        "low_drift_is_hard_artifact": low_drift_is_hard_artifact,
    }

    return df_features, thresholds
