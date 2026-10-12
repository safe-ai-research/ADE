# Rule-based quality labels

from .artifacts import pooled_quantile


def apply_quality_labels(df_features):
    """Create QUALITY, USABLE and STRICT_CLEAN without changing definitions."""
    # =========================
    # WINDOW CLASSIFICATION
    # =========================

    threshold_corr = 0.3 # (Default: 0.3 slighty good windows, 0.5 good windows)
    threshold_var_quality = pooled_quantile( # (Default: 0.75 slightly strict, 0.85 softer)
        df_features,
        "var1",
        "var2",
        0.75
    )
    # -------------------------
    # QUALITY LABEL
    # -------------------------
    # Technically useful window via signal quality criteria
    # There can be still heuristically deteced artifact type despite quality_label=1

    # --- Quality without Autoreject as basis for comparison ---
    df_features["quality_base_label_eeg1"] = (
        (df_features["corr"].abs() > threshold_corr) &      # Channel similarity sufficient
        (df_features["var1"] < threshold_var_quality)       # Variance not too high
    ).astype(int) # 1 = teechnically viable, 0 = technically noticable/worse

    df_features["quality_base_label_eeg2"] = (
        (df_features["corr"].abs() > threshold_corr) &      # Channel similarity sufficient
        (df_features["var2"] < threshold_var_quality)       # Variance not too high
    ).astype(int) # 1 = teechnically viable, 0 = technically noticable/worse

    df_features["quality_base_label"] = (
        (df_features["quality_base_label_eeg1"] == 1) &
        (df_features["quality_base_label_eeg2"] == 1)
    ).astype(int)

    # --- Quality with Autoreject-Veto ---
    df_features["quality_label_eeg1"] = (
        (df_features["quality_base_label_eeg1"] == 1) &
        (~df_features["autoreject_bad_eeg1"])               # not rejected by Autoreject
    ).astype(int)

    df_features["quality_label_eeg2"] = (
        (df_features["quality_base_label_eeg2"] == 1) &
        (~df_features["autoreject_bad_eeg2"])               # not rejected by Autoreject
    ).astype(int)

    df_features["quality_label"] = (
        (df_features["quality_label_eeg1"] == 1) &
        (df_features["quality_label_eeg2"] == 1)
    ).astype(int)

    # -------------------------
    # USABLE LABEL
    # -------------------------
    # Technically viable and no hard artifact marker
    # SLOW_CORR may be included
    df_features["usable_label_eeg1"] = (
        (df_features["quality_label_eeg1"] == 1) &
        (df_features["has_no_hard_artifact_eeg1"])
    ).astype(int)

    df_features["usable_label_eeg2"] = (
        (df_features["quality_label_eeg2"] == 1) &
        (df_features["has_no_hard_artifact_eeg2"])
    ).astype(int)

    df_features["usable_label"] = (
        (df_features["usable_label_eeg1"] == 1) &
        (df_features["usable_label_eeg2"] == 1)
    ).astype(int)

    # -------------------------
    # STRICT CLEAN LABEL
    # -------------------------

    df_features["has_any_marker_eeg1"] = (
        df_features["has_slow_corr"] |
        df_features["has_emg_eeg1"] |
        df_features["has_ecg_eeg1"] |
        df_features["has_bad_auto_eeg1"] |
        df_features["has_wavelet_hf_eeg1"] |
        df_features["has_wavelet_transient_eeg1"] |
        df_features["has_amp_burst_eeg1"] |
        df_features["has_jump_eeg1"] |
        df_features["has_flatline_eeg1"] |
        df_features["has_hf_emg_eeg1"] |
        df_features["has_low_drift_eeg1"] |
        df_features["has_clipping_eeg1"]
    )

    df_features["has_any_marker_eeg2"] = (
        df_features["has_slow_corr"] |
        df_features["has_emg_eeg2"] |
        df_features["has_ecg_eeg2"] |
        df_features["has_bad_auto_eeg2"] |
        df_features["has_wavelet_hf_eeg2"] |
        df_features["has_wavelet_transient_eeg2"] |
        df_features["has_amp_burst_eeg2"] |
        df_features["has_jump_eeg2"] |
        df_features["has_flatline_eeg2"] |
        df_features["has_hf_emg_eeg2"] |
        df_features["has_low_drift_eeg2"] |
        df_features["has_clipping_eeg2"]
    )

    # Technically viable and absolutely no marker
    df_features["strict_clean_label_eeg1"] = (
        (df_features["quality_label_eeg1"] == 1) &
        (~df_features["has_any_marker_eeg1"])
    ).astype(int)

    df_features["strict_clean_label_eeg2"] = (
        (df_features["quality_label_eeg2"] == 1) &
        (~df_features["has_any_marker_eeg2"])
    ).astype(int)

    df_features["strict_clean_label"] = (
        (df_features["strict_clean_label_eeg1"] == 1) &
        (df_features["strict_clean_label_eeg2"] == 1)
    ).astype(int)

    return df_features, {
        "threshold_corr": threshold_corr,
        "threshold_var_quality": threshold_var_quality,
    }


def form_quality_groups(df_features):
    # Create the existing compatibility groups and print their summaries
    # Keeping old label-column as alias, so older code-/plot-amounts do not break
    # label = quality_label
    df_features["label"] = df_features["quality_label"]

    df_exploded = df_features.explode("artifact_type")

    # -------------------------
    # Form groups
    # -------------------------
    df_quality_good = df_features[df_features["quality_label"] == 1]
    df_quality_bad = df_features[df_features["quality_label"] == 0]

    # Compatibility-alias for previous export-code
    df_good = df_quality_good

    df_good_usable= df_quality_good[df_quality_good["has_no_hard_artifact"]]
    df_good_strict_clean = df_quality_good[df_quality_good["is_strict_clean"]]

    df_good_slowcorr = df_quality_good[df_quality_good["has_slow_corr"]]
    df_good_emg = df_quality_good[df_quality_good["has_emg"]]
    df_good_ecg = df_quality_good[df_quality_good["has_ecg"]]
    df_good_bad_auto = df_quality_good[df_quality_good["has_bad_auto"]]
    df_good_mixed = df_quality_good[df_quality_good["is_mixed"]]

    df_good_wavelet_hf = df_quality_good[df_quality_good["has_wavelet_hf"]]
    df_good_wavelet_transient = df_quality_good[df_quality_good["has_wavelet_transient"]]

    df_good_amp_burst = df_quality_good[df_quality_good["has_amp_burst"]]
    df_good_jump = df_quality_good[df_quality_good["has_jump"]]
    df_good_flatline = df_quality_good[df_quality_good["has_flatline"]]
    df_good_hf_emg = df_quality_good[df_quality_good["has_hf_emg"]]
    df_good_low_drift = df_quality_good[df_quality_good["has_low_drift"]]
    df_good_clipping = df_quality_good[df_quality_good["has_clipping"]]

    # -------------------------
    # Output
    # -------------------------
    print("\nQuality-good windows:", int(df_features["quality_label"].sum()))
    print("Quality-bad windows:", int(len(df_features) - df_features["quality_label"].sum()))

    print("\nUsable windows:", int(df_features["usable_label"].sum()))
    print("Not usable windows:", int(len(df_features) - df_features["usable_label"].sum()))

    print("\nStrict-clean windows:", int(df_features["strict_clean_label"].sum()))
    print("Not strict-clean windows:", int(len(df_features) - df_features["strict_clean_label"].sum()))

    print("\nArtifact Summary:")
    print(df_exploded["artifact_type"].value_counts())

    print("\nMixed windows:", int(df_features["is_mixed"].sum()))

    print("\nQuality-good windows by artifact type:")
    print(df_exploded[df_exploded["quality_label"] == 1]["artifact_type"].value_counts())

    print("\nQuality-bad windows by artifact type:")
    print(df_exploded[df_exploded["quality_label"] == 0]["artifact_type"].value_counts())

    print("\nUsable / no-hard-artifact windows by marker:")
    print(df_exploded[df_exploded["usable_label"] == 1]["artifact_type"].value_counts())

    print("\nStrict-clean windows by marker:")
    print(df_exploded[df_exploded["strict_clean_label"] == 1]["artifact_type"].value_counts())

    return {
        "df_exploded": df_exploded,
        "df_quality_good": df_quality_good,
        "df_quality_bad": df_quality_bad,
        "df_good": df_good,
        "df_good_usable": df_good_usable,
        "df_good_strict_clean": df_good_strict_clean,
        "df_good_slowcorr": df_good_slowcorr,
        "df_good_emg": df_good_emg,
        "df_good_ecg": df_good_ecg,
        "df_good_bad_auto": df_good_bad_auto,
        "df_good_mixed": df_good_mixed,
        "df_good_wavelet_hf": df_good_wavelet_hf,
        "df_good_wavelet_transient": df_good_wavelet_transient,
        "df_good_amp_burst": df_good_amp_burst,
        "df_good_jump": df_good_jump,
        "df_good_flatline": df_good_flatline,
        "df_good_hf_emg": df_good_hf_emg,
        "df_good_low_drift": df_good_low_drift,
        "df_good_clipping": df_good_clipping,
    }
