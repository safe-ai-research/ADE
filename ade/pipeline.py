# eeg base preprocessing pipeline
# Ordered processing pipeline for one recording

import os

import numpy as np

from .artifacts import apply_artifact_markers, run_autoreject_detection
from .data_io import (
    build_filename_col,
    build_metrics,
    export_artifact_annotations,
    load_recording,
    plot_results,
    write_metrics_block,
)
from .features import (
    add_engineered_features,
    bandpower,
    calculate_global_features,
    calculate_method_specific_global_features,
    calculate_segment_features,
    calculate_window_features,
)
from .quality import apply_quality_labels, form_quality_groups
from .signal_processing import (
    apply_signal_method,
    resample_and_preprocess,
    validate_signal_method,
)
from .windowing import create_windows


def run_pipeline(
    dataset,
    filename,
    file_ending,
    params,
    base_path,
    file_path_sheet,
):
    # Run the complete ADE processing sequence for one recording
    file = filename + file_ending
    file_path = os.path.join(base_path, dataset, file)

    # --- Signal Seperation ---
    signal_method = params.get("signal_method", "original")      # "original", "ica_ecg", "asr", "ajdc", "wavelet", "adaptive_ecg"
    validate_signal_method(signal_method)

    # Windowing
    window_sec = params.get("window_sec", 5)
    window_overlap = params.get("window_overlap", 0.5)

    # Plots
    make_plots = params.get("make_plots", False)

    # --- Export detected artifact windows ---
    export_artifact_windows = params.get(
        "export_artifact_windows",
        False
    )

    artifact_export_dir = params.get(
        "artifact_export_dir",
        os.path.join(
            base_path,
            dataset,
            "artifact_annotations_pipeline"
        )
    )

    # =========================
    # LOAD
    # See: ade/data_io.py -> load_recording()
    # =========================
    loaded = load_recording(file_path)
    hr_ref = loaded["hr_ref"]

    # =========================
    # RESAMPLING + BASIC PREPROCESSING
    # See: ade/signal_processing.py -> resample_and_preprocess()
    # =========================
    processed = resample_and_preprocess(loaded, params)
    t0 = processed["t0"]
    t = processed["t"]
    fs = processed["fs"]
    eeg1 = processed["eeg1"]
    eeg2 = processed["eeg2"]
    ecg_ref = processed["ecg_ref"]
    raw_mne = processed["raw_mne"]
    diff = processed["diff"]
    corr = processed["corr"]

    # =========================
    # PSD + GLOBAL BANDPOWER
    # See: ade/features.py -> calculate_global_features()
    # =========================
    global_features = calculate_global_features(eeg1, eeg2, fs)

    # =========================
    # WINDOWING
    # See: ade/windowing.py -> create_windows()
    # =========================
    windows_eeg1 = create_windows(eeg1, fs, window_sec=window_sec, overlap=window_overlap)
    windows_eeg2 = create_windows(eeg2, fs, window_sec=window_sec, overlap=window_overlap)

    if ecg_ref is not None:
        windows_ecg = create_windows(ecg_ref, fs, window_sec=window_sec, overlap=window_overlap)
    else:
        windows_ecg = None

    print(f"\nAnzahl Fenster: {len(windows_eeg1)}")

    if windows_ecg is not None:
        print(f"Anzahl ECG-Fenster: {len(windows_ecg)}")

    # =========================
    # SIGNAL METHOD
    # See: ade/signal_processing.py -> apply_signal_method()
    # =========================
    signal_result = apply_signal_method(
        signal_method=signal_method,
        eeg1=eeg1,
        eeg2=eeg2,
        ecg_ref=ecg_ref,
        raw_mne=raw_mne,
        fs=fs,
        params=params,
    )
    signal1_for_features = signal_result["signal1_for_features"]
    signal2_for_features = signal_result["signal2_for_features"]
    selected_signal_method_success = signal_result[
        "selected_signal_method_success"
    ]

    windows_eeg1 = create_windows(signal1_for_features, fs, window_sec=window_sec, overlap=window_overlap)
    windows_eeg2 = create_windows(signal2_for_features, fs, window_sec=window_sec, overlap=window_overlap)

    # =========================
    # METHOD SPECIFIC GLOBAL FEATURES
    # See: ade/features.py -> calculate_method_specific_global_features()
    # =========================
    method_features = calculate_method_specific_global_features(
        signal1_for_features=signal1_for_features,
        signal2_for_features=signal2_for_features,
        eeg1=eeg1,
        eeg2=eeg2,
        fs=fs,
    )

    # =========================
    # ARTIFACT DETECTION – AUTOREJECT
    # See: ade/artifacts.py -> run_autoreject_detection()
    # =========================
    autoreject_result = run_autoreject_detection(
        windows_eeg1,
        windows_eeg2,
        fs,
        params,
    )

    # =========================
    # WINDOW FEATURES
    # See: ade/features.py -> calculate_window_features()
    # =========================
    use_wavelet_marker_features = params.get("use_wavelet_marker_features", True)
    df_features, window_features = calculate_window_features(
        windows_eeg1=windows_eeg1,
        windows_eeg2=windows_eeg2,
        windows_ecg=windows_ecg,
        autoreject_bad_windows=autoreject_result["autoreject_bad_windows"],
        autoreject_bad_eeg1=autoreject_result["autoreject_bad_eeg1"],
        autoreject_bad_eeg2=autoreject_result["autoreject_bad_eeg2"],
        fs=fs,
        hr_ref=hr_ref,
        use_wavelet_marker_features=use_wavelet_marker_features,
        wavelet_name=signal_result["wavelet_name"],
        wavelet_level=signal_result["wavelet_level"],
        wavelet_mode=signal_result["wavelet_mode"],
    )
    df_features = add_engineered_features(df_features)

    # =========================
    # ARTIFACT / MARKER DETECTION
    # See: ade/artifacts.py -> apply_artifact_markers()
    # =========================
    df_features, artifact_thresholds = apply_artifact_markers(
        df_features,
        params,
    )

    # Return of Features
    print(f"Feature-Fenster: {len(window_features)}")

    # =========================
    # WINDOW CLASSIFICATION
    # See: ade/quality.py -> apply_quality_labels()
    # =========================
    df_features, quality_thresholds = apply_quality_labels(df_features)

    quality_groups = form_quality_groups(df_features)

    # =========================
    # EXPORT: ARTIFACT WINDOW PER FILE
    # See: ade/data_io.py -> export_artifact_annotations()
    # =========================
    artifact_export_success = selected_signal_method_success
    export_artifact_annotations(
        export_artifact_windows=export_artifact_windows,
        artifact_export_success=artifact_export_success,
        signal_method=signal_method,
        artifact_export_dir=artifact_export_dir,
        filename=filename,
        df_features=df_features,
        fs=fs,
        window_sec=window_sec,
        window_overlap=window_overlap,
        t0=t0,
        artifact_thresholds=artifact_thresholds,
        quality_thresholds=quality_thresholds,
        autoreject_result=autoreject_result,
    )

    # =========================
    # SEGMENT ANALYSIS (Beginning vs Ending)
    # See: ade/features.py -> calculate_segment_features()
    # =========================
    segment_features = calculate_segment_features(eeg1, fs)

    # =========================
    # FEATURES (Overview)
    # =========================
    f_psd_global_1 = global_features["f_psd_global_1"]
    Pxx_global_1 = global_features["Pxx_global_1"]
    f_psd_global_2 = global_features["f_psd_global_2"]
    Pxx_global_2 = global_features["Pxx_global_2"]

    features = {
        "corr": corr,
        "delta_eeg1": bandpower(f_psd_global_1, Pxx_global_1, 0.5, 4),
        "theta_eeg1": bandpower(f_psd_global_1, Pxx_global_1, 4, 8),
        "alpha_eeg1": bandpower(f_psd_global_1, Pxx_global_1, 8, 13),
        "beta_eeg1": bandpower(f_psd_global_1, Pxx_global_1, 13, 30),

        "delta_eeg2": bandpower(f_psd_global_2, Pxx_global_2, 0.5, 4),
        "theta_eeg2": bandpower(f_psd_global_2, Pxx_global_2, 4, 8),
        "alpha_eeg2": bandpower(f_psd_global_2, Pxx_global_2, 8, 13),
        "beta_eeg2": bandpower(f_psd_global_2, Pxx_global_2, 13, 30),
    }

    print("\nFeature Summary:")
    for k, v in features.items():
        print(f"{k}: {v:.4f}")

    # =========================
    # VALIDATIONS
    # =========================

    # ECG
    if ecg_ref is not None:
        # ECG Ref
        print("ECG ref min:", np.min(ecg_ref))
        print("ECG ref max:", np.max(ecg_ref))
        print("ECG ref std:", np.std(ecg_ref))

        # Correlation EEG vs ECG
        print("corr EEG1 vs ECG_II:", np.corrcoef(eeg1, ecg_ref)[0, 1])
        print("corr EEG2 vs ECG_II:", np.corrcoef(eeg2, ecg_ref)[0, 1])
    else:
        print("Keine ECG_II-Referenz vorhanden.")

    # =========================
    # PLOTS
    # See: ade/data_io.py -> plot_results()
    # =========================
    plot_results(
        make_plots=make_plots,
        t=t,
        eeg1=eeg1,
        eeg2=eeg2,
        diff=diff,
        global_features=global_features,
        segment_features=segment_features,
        window_features=window_features,
        df_features=df_features,
        ecg_ref=ecg_ref,
    )

    # =========================
    # EXPORT TO SHEET
    # See: ade/data_io.py -> build_metrics() + write_metrics_block()
    # =========================
    filename_col = build_filename_col(
        filename,
        signal_method,
        signal_result,
    )

    metrics = build_metrics(
        loaded=loaded,
        processed=processed,
        global_features=global_features,
        method_features=method_features,
        signal_result=signal_result,
        autoreject_result=autoreject_result,
        artifact_thresholds=artifact_thresholds,
        df_features=df_features,
        quality_groups=quality_groups,
        signal_method=signal_method,
        filename_col=filename_col,
    )

    write_metrics_block(file_path_sheet, filename_col, metrics)
