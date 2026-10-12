# Raw-data loading and result exports for the ADE pipeline

import gzip
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import use_autoreject_available
from .features import bandpower, use_wavelet_available
from .signal_processing import use_ajdc_available, use_asr_available
from .windowing import create_window_indices


def get_data_file_parts(file_path):
    """
    Returns filename and file_ending according to run_pipeline.
    Examples:
    10-123.csv      -> ("10-123", ".csv")
    19-456.csv.gz   -> ("19-456", ".csv.gz")
    """
    if file_path.name.endswith(".csv.gz"):
        filename = file_path.name[:-len(".csv.gz")]
        file_ending = ".csv.gz"
    elif file_path.name.endswith(".csv"):
        filename = file_path.name[:-len(".csv")]
        file_ending = ".csv"
    else:
        return None

    return filename, file_ending


def discover_datasets_to_run(base_path, dataset_name, exclude_files):
    # Discover all supported input files for one dataset folder
    data_dir = base_path / dataset_name

    data_files = sorted(
        file_path
        for file_path in data_dir.iterdir()
        if file_path.is_file()
        and (file_path.name.endswith(".csv") or file_path.name.endswith(".csv.gz"))
    )

    datasets_to_run = []

    for file_path in data_files:
        parts = get_data_file_parts(file_path)

        if parts is None:
            continue

        filename, file_ending = parts

        if filename in exclude_files:
            continue

        datasets_to_run.append((dataset_name, filename, file_ending))

    print(f"Found files for {dataset_name}: {len(datasets_to_run)}")

    print("First files:")
    for entry in datasets_to_run[:5]:
        print(entry)

    print("Last files:")
    for entry in datasets_to_run[-5:]:
        print(entry)

    if not datasets_to_run:
        raise FileNotFoundError(f"No CSV- or CSV.GZ-files found in: {data_dir}")

    return datasets_to_run


def load_recording(file_path):
    # Load the selected EEG/ECG columns without changing signal processing
    fs_target = 125  # Target Sample Rate

    time_col     = "Time"
    eeg1_col     = "Intellivue/EEG_1"
    eeg2_col     = "Intellivue/EEG_2"
    ecg_ii_col   = "Intellivue/ECG_II"  # for DOSE I: ECG_II, DOSE-II: ECG_II_WAV
    ecg_hr_col   = "Intellivue/ECG_HR"
    pleth_hr_col = "Intellivue/PLETH_HR"

    # ECG
    ecg_ii_candidates = [
        "Intellivue/ECG_II",
        "Intellivue/ECG_II_WAV"
    ]

    # =========================
    # LOAD
    # =========================
    if str(file_path).endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            df = pd.read_csv(f, low_memory=False)
    else:
        df = pd.read_csv(file_path, low_memory=False)

    # =========================
    # SELECT + CLEANy
    # =========================

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # --- EEG-Data ---
    df_eeg = df[[time_col, eeg1_col, eeg2_col]].copy()

    df_eeg[eeg1_col] = pd.to_numeric(df_eeg[eeg1_col], errors="coerce")
    df_eeg[eeg2_col] = pd.to_numeric(df_eeg[eeg2_col], errors="coerce")

    df_eeg = df_eeg.dropna(subset=[time_col, eeg1_col, eeg2_col])

    # --- ECG_II Reference Signal ---
    ecg_ii_col = next((col for col in ecg_ii_candidates if col in df.columns), None)

    if ecg_ii_col is not None:
        print("ECG-Waveform-Column:", ecg_ii_col)
    else:
        print("No ECG-Waveform-Column found.")

    if ecg_ii_col is not None:
        df_ecg = df[[time_col, ecg_ii_col]].copy()
        df_ecg[ecg_ii_col] = pd.to_numeric(df_ecg[ecg_ii_col], errors="coerce")
        df_ecg = df_ecg.dropna(subset=[time_col, ecg_ii_col])
    else:
        df_ecg = None

    # --- Heartrate from ECG_HR and PLETH_HR ---
    hr_values = []

    if ecg_hr_col in df.columns:
        hr_values.extend(pd.to_numeric(df[ecg_hr_col], errors="coerce").dropna())

    if pleth_hr_col in df.columns:
        hr_values.extend(pd.to_numeric(df[pleth_hr_col], errors="coerce").dropna())

    hr_ref = np.mean(hr_values) if len(hr_values) > 0 else None

    print("HR ref:", hr_ref)

    return {
        "fs_target": fs_target,
        "time_col": time_col,
        "eeg1_col": eeg1_col,
        "eeg2_col": eeg2_col,
        "ecg_ii_col": ecg_ii_col,
        "ecg_hr_col": ecg_hr_col,
        "pleth_hr_col": pleth_hr_col,
        "df": df,
        "df_eeg": df_eeg,
        "df_ecg": df_ecg,
        "hr_ref": hr_ref,
    }


def write_metrics_block(
    file_path_sheet, filename_col, metrics,
    overwrite=True,         # Custom: Overwrite existing values (Default: True)
    create_backup=False,    # Custom: Create backup (Default: False)
    sort_columns=True,      # Custom: Sort columns (Default: True)
    sort_index=True,        # Custom: sort Index by Metrics-List (Default: True)
    verbose=False,          # Custom: Debug (Default: False)
):
    # --- Load ---
    if os.path.exists(file_path_sheet):
        df = pd.read_excel(file_path_sheet, engine="openpyxl", index_col=0)

        # force Index & Column names as strings (prevents Type conflicts with .loc)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        if create_backup:
            file_path_sheet = Path(file_path_sheet)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            backup_path = file_path_sheet.with_name(
                f"{file_path_sheet.stem}_backup_{timestamp}"
                f"{file_path_sheet.suffix}"
            )

            df.to_excel(backup_path)

            if verbose:
                print(f"[Created Backup] {backup_path}")

    else:
        df = pd.DataFrame()

    filename_col = str(filename_col)

    # --- Fix column ---
    if filename_col in df.columns:
        if verbose:
            print(f"[Overwrite Spalte] {filename_col}")
        df = df.drop(columns=[filename_col])
    else:
        if verbose:
            print(f"[Neue Spalte] {filename_col}")

    # --- Make new column as Series ---
    new_col = pd.Series(
        {str(row_name): value for row_name, value in metrics.items()},
        name=filename_col
    )

    # Merge Index, so new rows are kept
    df = df.reindex(df.index.union(new_col.index))

    # Add new column
    df = pd.concat([df, new_col], axis=1)

    # Defrag
    df = df.copy()

    # --- Sort ---
    if sort_columns:
        df = df.reindex(sorted(df.columns), axis=1)

    if sort_index:
        df = df.reindex(metrics.keys())

    if not sort_index:
        df = df.sort_index()

    # --- Save ---
    with pd.ExcelWriter(file_path_sheet, engine="openpyxl") as writer:
        df.to_excel(writer)

    if verbose:
        print("[Saved]")


def export_artifact_annotations(
    *,
    export_artifact_windows,
    artifact_export_success,
    signal_method,
    artifact_export_dir,
    filename,
    df_features,
    fs,
    window_sec,
    window_overlap,
    t0,
    artifact_thresholds,
    quality_thresholds,
    autoreject_result,
):
    # Export artifact-only, all-window, and threshold sheets
    # =========================
    # EXPORT: ARTIFACT WINDOW PER FILE
    # =========================

    if export_artifact_windows and not artifact_export_success:
        print(
            "Artifact export skipped: "
            f"signal_method='{signal_method}' was not successful. "
            "The original fallback signal will not be exported "
            "under the method name."
        )

    artifact_output_path = ""

    if not (export_artifact_windows and artifact_export_success):
        return artifact_output_path

    # Name of return column:
    # Export name -> existing df_features-column
    artifact_marker_columns = {
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

    artifact_channel_columns = {
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

    quality_label_columns = {
        # EEG1
        "QUALITY_BASE_LABEL_EEG1": "quality_base_label_eeg1",
        "QUALITY_LABEL_EEG1": "quality_label_eeg1",
        "USABLE_LABEL_EEG1": "usable_label_eeg1",
        "STRICT_CLEAN_LABEL_EEG1": "strict_clean_label_eeg1",

        # EEG2
        "QUALITY_BASE_LABEL_EEG2": "quality_base_label_eeg2",
        "QUALITY_LABEL_EEG2": "quality_label_eeg2",
        "USABLE_LABEL_EEG2": "usable_label_eeg2",
        "STRICT_CLEAN_LABEL_EEG2": "strict_clean_label_eeg2",

        # Combined
        "QUALITY_BASE_LABEL": "quality_base_label",
        "QUALITY_LABEL": "quality_label",
        "USABLE_LABEL": "usable_label",
        "STRICT_CLEAN_LABEL": "strict_clean_label",
        "IS_MIXED": "is_mixed",
    }

    window_indices = create_window_indices(
        t0=t0,
        number_of_windows=len(df_features),
        fs=fs,
        window_sec=window_sec,
        window_overlap=window_overlap,
    )
    begin_idx = window_indices["begin_idx"]
    end_idx = window_indices["end_idx"]

    # Make base annotation table for ALL windows
    df_annotation_export = pd.DataFrame({
        "begin_idx": begin_idx,
        "end_idx": end_idx,
    })

    # Add artifact markers as 1/0
    for export_column, source_column in artifact_marker_columns.items():
        df_annotation_export[export_column] = (
            df_features[source_column]
            .astype(int)
            .to_numpy()
        )

    # Add channel-specific artifact markers as 1/0
    for export_column, source_column in artifact_channel_columns.items():
        df_annotation_export[export_column] = (
            df_features[source_column]
            .astype(int)
            .to_numpy()
        )

    # Add quality labels as 1/0
    for export_column, source_column in quality_label_columns.items():
        df_annotation_export[export_column] = (
            df_features[source_column]
            .astype(int)
            .to_numpy()
        )

    marker_names = list(artifact_marker_columns.keys())

    # =========================
    # ARTIFACT-ONLY VIEW
    # =========================

    # Only return windows in which at least one marker has been detected
    artifact_window_mask = (
        df_annotation_export[marker_names]
        .any(axis=1)
    )

    df_artifact_export = (
        df_annotation_export
        .loc[artifact_window_mask]
        .copy()
    )

    # Normal numbering 1, 2, 3, ...
    # The headline of the index column stays empty,
    # similar to the format of the expert annotations
    df_artifact_export.index = np.arange(
        1,
        len(df_artifact_export) + 1
    )
    df_artifact_export.index.name = ""

    # =========================
    # ALL-WINDOW VIEW
    # =========================

    df_window_export = df_annotation_export.copy()

    feature_export_columns = [
        # Absolute bandpower
        "delta1",
        "theta1",
        "alpha1",
        "beta1",
        "delta2",
        "theta2",
        "alpha2",
        "beta2",

        # Relative bandpower
        "delta1_rel",
        "theta1_rel",
        "alpha1_rel",
        "beta1_rel",
        "delta2_rel",
        "theta2_rel",
        "alpha2_rel",
        "beta2_rel",

        # Channel / time-domain features
        "corr",

        "var1",
        "var2",
        "var_diff",

        "std1",
        "std2",
        "std_diff",

        "mad1",
        "mad2",

        "ptp1",
        "ptp2",
        "ptp_diff",

        "maxabs1",
        "maxabs2",

        "max_jump_norm1",
        "max_jump_norm2",

        "line_length_norm1",
        "line_length_norm2",

        "trend_slope_norm1",
        "trend_slope_norm2",

        "clipping_ratio1",
        "clipping_ratio2",

        "hf_rel1",
        "hf_rel2",

        # Combined / compatibility features
        "std_avg",
        "mad_avg",
        "ptp_avg",
        "maxabs_avg",
        "max_jump_norm_avg",
        "line_length_norm_avg",
        "trend_slope_norm_avg",
        "clipping_ratio_avg",
        "hf_rel_avg",

        # Wavelet EEG1
        "wavelet_detail1",
        "wavelet_hf1",
        "wavelet_entropy1",
        "wavelet_maxabs1",
        "wavelet_kurtosis1",

        # Wavelet EEG2
        "wavelet_detail2",
        "wavelet_hf2",
        "wavelet_entropy2",
        "wavelet_maxabs2",
        "wavelet_kurtosis2",

        # Engineered combined features
        "beta_avg",
        "delta_avg",
        "beta_rel_avg",
        "delta_rel_avg",
        "wavelet_hf_avg",
        "wavelet_detail_avg",
        "wavelet_entropy_avg",
        "wavelet_maxabs_avg",
        "wavelet_kurtosis_avg",
    ]

    for feature_column in feature_export_columns:
        df_window_export[feature_column] = (
            df_features[feature_column]
            .to_numpy()
        )

    # Normal numbering 1, 2, 3, ...
    df_window_export.index = np.arange(
        1,
        len(df_window_export) + 1
    )
    df_window_export.index.name = ""

    # Create output folder, if it does not yet exist
    os.makedirs(
        artifact_export_dir,
        exist_ok=True
    )

    artifact_output_path = os.path.join(
        artifact_export_dir,
        f"{filename}_artifacts.xlsx"
    )

    """
    Preserve the effective per-recording thresholds. 
    Several ADE markers are based on pooled recording quantiles, 
    so before/after marker counts must be interpreted together with these values
    """
    threshold_delta_slowcorr = artifact_thresholds["threshold_delta_slowcorr"]
    threshold_corr_slowcorr = artifact_thresholds["threshold_corr_slowcorr"]
    threshold_beta_emg_rel = artifact_thresholds["threshold_beta_emg_rel"]
    threshold_var_emg_quantile = artifact_thresholds["threshold_var_emg_quantile"]
    threshold_var_emg = artifact_thresholds["threshold_var_emg"]
    threshold_wavelet_hf_quantile = artifact_thresholds["threshold_wavelet_hf_quantile"]
    threshold_wavelet_hf = artifact_thresholds["threshold_wavelet_hf"]
    threshold_wavelet_transient_quantile = artifact_thresholds["threshold_wavelet_transient_quantile"]
    threshold_wavelet_transient = artifact_thresholds["threshold_wavelet_transient"]
    threshold_amp_burst_quantile = artifact_thresholds["threshold_amp_burst_quantile"]
    threshold_amp_burst = artifact_thresholds["threshold_amp_burst"]
    threshold_jump_quantile = artifact_thresholds["threshold_jump_quantile"]
    threshold_jump = artifact_thresholds["threshold_jump"]
    threshold_flatline_quantile = artifact_thresholds["threshold_flatline_quantile"]
    threshold_flatline_std = artifact_thresholds["threshold_flatline_std"]
    threshold_flatline_ptp = artifact_thresholds["threshold_flatline_ptp"]
    threshold_hf_emg_rel = artifact_thresholds["threshold_hf_emg_rel"]
    threshold_hf_emg_line_quantile = artifact_thresholds["threshold_hf_emg_line_quantile"]
    threshold_hf_emg_line = artifact_thresholds["threshold_hf_emg_line"]
    threshold_low_drift_delta_rel = artifact_thresholds["threshold_low_drift_delta_rel"]
    threshold_low_drift_slope_quantile = artifact_thresholds["threshold_low_drift_slope_quantile"]
    threshold_low_drift_slope = artifact_thresholds["threshold_low_drift_slope"]
    threshold_clipping_ratio = artifact_thresholds["threshold_clipping_ratio"]
    threshold_corr = quality_thresholds["threshold_corr"]
    threshold_var_quality = quality_thresholds["threshold_var_quality"]

    autoreject_threshold_eeg_override = autoreject_result["autoreject_threshold_eeg_override"]
    autoreject_threshold_eeg = autoreject_result["autoreject_threshold_eeg"]
    autoreject_threshold_override_source = autoreject_result["autoreject_threshold_override_source"]
    autoreject_random_state = autoreject_result["autoreject_random_state"]

    df_threshold_export = pd.DataFrame(
        [
            {
                "marker": "SLOW_CORR",
                "basis": "fixed",
                "threshold_1_name": "delta_rel_avg_gt",
                "threshold_1_value": threshold_delta_slowcorr,
                "threshold_2_name": "abs_corr_gt",
                "threshold_2_value": threshold_corr_slowcorr,
            },
            {
                "marker": "EMG",
                "basis": "fixed_and_recording_quantile",
                "threshold_1_name": "beta_rel_gt",
                "threshold_1_value": threshold_beta_emg_rel,
                "threshold_2_name": f"variance_q{threshold_var_emg_quantile:g}_gt",
                "threshold_2_value": threshold_var_emg,
            },
            {
                "marker": "ECG",
                "basis": "fixed_detection_logic",
                "threshold_1_name": "autocorr_gt",
                "threshold_1_value": 0.3,
                "threshold_2_name": "hr_tolerance_bpm",
                "threshold_2_value": 15.0,
                "notes": (
                    "peak_height_factor=1; min_hr=40; max_hr=120; "
                    "ECG-reference confirmation when available"
                ),
            },
            {
                "marker": "BAD_AUTO",
                "basis": (
                    "autoreject_fixed_override"
                    if autoreject_threshold_eeg_override is not None
                    else "autoreject_recording_adaptive"
                ),
                "threshold_1_name": "autoreject_eeg",
                "threshold_1_value": autoreject_threshold_eeg,
                "threshold_2_name": "",
                "threshold_2_value": np.nan,
                "notes": (
                    "source=" + autoreject_threshold_override_source
                    if autoreject_threshold_eeg_override is not None
                    and autoreject_threshold_override_source
                    else f"random_state={autoreject_random_state}"
                ),
            },
            {
                "marker": "WAVELET_HF",
                "basis": "recording_quantile",
                "threshold_1_name": f"wavelet_hf_q{threshold_wavelet_hf_quantile:g}_gt",
                "threshold_1_value": threshold_wavelet_hf,
                "threshold_2_name": "",
                "threshold_2_value": np.nan,
            },
            {
                "marker": "WAVELET_TRANSIENT",
                "basis": "recording_quantile",
                "threshold_1_name": f"wavelet_maxabs_q{threshold_wavelet_transient_quantile:g}_gt",
                "threshold_1_value": threshold_wavelet_transient,
                "threshold_2_name": "",
                "threshold_2_value": np.nan,
            },
            {
                "marker": "AMP_BURST",
                "basis": "recording_quantile",
                "threshold_1_name": f"ptp_q{threshold_amp_burst_quantile:g}_gt",
                "threshold_1_value": threshold_amp_burst,
                "threshold_2_name": "",
                "threshold_2_value": np.nan,
            },
            {
                "marker": "JUMP",
                "basis": "recording_quantile",
                "threshold_1_name": f"max_jump_norm_q{threshold_jump_quantile:g}_gt",
                "threshold_1_value": threshold_jump,
                "threshold_2_name": "",
                "threshold_2_value": np.nan,
            },
            {
                "marker": "FLATLINE",
                "basis": "recording_quantile",
                "threshold_1_name": f"std_q{threshold_flatline_quantile:g}_le",
                "threshold_1_value": threshold_flatline_std,
                "threshold_2_name": f"ptp_q{threshold_flatline_quantile:g}_le",
                "threshold_2_value": threshold_flatline_ptp,
            },
            {
                "marker": "HF_EMG",
                "basis": "fixed_and_recording_quantile",
                "threshold_1_name": "hf_rel_gt",
                "threshold_1_value": threshold_hf_emg_rel,
                "threshold_2_name": f"line_length_q{threshold_hf_emg_line_quantile:g}_gt",
                "threshold_2_value": threshold_hf_emg_line,
            },
            {
                "marker": "LOW_DRIFT",
                "basis": "fixed_and_recording_quantile",
                "threshold_1_name": "delta_rel_gt",
                "threshold_1_value": threshold_low_drift_delta_rel,
                "threshold_2_name": f"trend_slope_q{threshold_low_drift_slope_quantile:g}_gt",
                "threshold_2_value": threshold_low_drift_slope,
            },
            {
                "marker": "CLIPPING",
                "basis": "fixed",
                "threshold_1_name": "clipping_ratio_gt",
                "threshold_1_value": threshold_clipping_ratio,
                "threshold_2_name": "",
                "threshold_2_value": np.nan,
            },
            {
                "marker": "QUALITY_BASE",
                "basis": "fixed_and_recording_quantile",
                "threshold_1_name": "abs_corr_gt",
                "threshold_1_value": threshold_corr,
                "threshold_2_name": "variance_q0.75_lt",
                "threshold_2_value": threshold_var_quality,
            },
        ]
    )

    # write Excel-file
    with pd.ExcelWriter(
        artifact_output_path,
        engine="openpyxl",
        datetime_format="yyyy-mm-dd hh:mm:ss.000"
    ) as writer:

        # Keep artifact sheet first for compatibility
        df_artifact_export.to_excel(
            writer,
            sheet_name="artifacts",
            index=True,
            index_label=""
        )

        # All windows including features and quality labels
        df_window_export.to_excel(
            writer,
            sheet_name="windows",
            index=True,
            index_label=""
        )

        df_threshold_export.to_excel(
            writer,
            sheet_name="thresholds",
            index=False,
        )

        for sheet_name, dataframe in [
            ("artifacts", df_artifact_export),
            ("windows", df_window_export),
        ]:

            worksheet = writer.sheets[sheet_name]

            # Date format with milliseconds
            for cell in worksheet["B"][1:]:
                cell.number_format = "yyyy-mm-dd hh:mm:ss.000"

            for cell in worksheet["C"][1:]:
                cell.number_format = "yyyy-mm-dd hh:mm:ss.000"

            # Freeze header
            worksheet.freeze_panes = "A2"

            # Autofilter
            worksheet.auto_filter.ref = worksheet.dimensions

            # Column widths
            worksheet.column_dimensions["A"].width = 8
            worksheet.column_dimensions["B"].width = 24
            worksheet.column_dimensions["C"].width = 24

            # Remaining columns
            for column_index in range(
                4,
                len(dataframe.columns) + 2
            ):
                header_cell = worksheet.cell(
                    row=1,
                    column=column_index
                )

                column_letter = header_cell.column_letter
                header_text = str(header_cell.value)

                worksheet.column_dimensions[column_letter].width = max(
                    12,
                    min(24, len(header_text) + 2)
                )

    print(
        "\nAll window annotations exported:",
        len(df_window_export)
    )

    print(
        "Artifact windows exported:",
        len(df_artifact_export)
    )

    print(
        "Annotation files:",
        artifact_output_path
    )

    return artifact_output_path


# --- add Suffix ---
# AJDC
def build_filename_col(filename, signal_method, signal_result):
    wavelet_threshold_scale = signal_result["wavelet_threshold_scale"]
    wavelet_name = signal_result["wavelet_name"]
    adaptive_ecg_mu = signal_result["adaptive_ecg_mu"]
    adaptive_ecg_filter_len = signal_result["adaptive_ecg_filter_len"]
    ajdc_success = signal_result["ajdc_success"]
    ajdc_supp = signal_result["ajdc_supp"]

    if signal_method == "wavelet":
        threshold_suffix = f"{float(wavelet_threshold_scale):g}".replace(".", "p")
        return f"{filename}_wavelet_{wavelet_name}_s{threshold_suffix}"

    if signal_method == "adaptive_ecg":
        mu_suffix = f"{float(adaptive_ecg_mu):g}".replace(".", "p")
        return f"{filename}_adaptive_ecg_l{adaptive_ecg_filter_len}_mu{mu_suffix}"

    if signal_method != "ajdc":
        return f"{filename}_{signal_method}"

    # AJDC has been picked, but did not run successfully
    if not ajdc_success:
        return f"{filename}_ajdc_failed"

    # If ajdc_supp is for some reason None
    if ajdc_supp is None:
        return f"{filename}_ajdc_all"

    # No Sources Removed = Reconstruction with all Sources
    if len(ajdc_supp) == 0:
        return f"{filename}_ajdc_all"

    # One or more sources have been removed
    removed_sources = "-".join(map(str, ajdc_supp))
    return f"{filename}_ajdc_rm{removed_sources}"


def build_metrics(
    *,
    loaded,
    processed,
    global_features,
    method_features,
    signal_result,
    autoreject_result,
    artifact_thresholds,
    df_features,
    quality_groups,
    signal_method,
    filename_col,
):
    # Build the unchanged recording-level metrics dictionary
    eeg1 = processed["eeg1"]
    eeg2 = processed["eeg2"]
    ecg_ref = processed["ecg_ref"]
    corr = processed["corr"]
    hr_ref = loaded["hr_ref"]

    f_psd_global_1 = global_features["f_psd_global_1"]
    Pxx_global_1 = global_features["Pxx_global_1"]
    f_psd_global_2 = global_features["f_psd_global_2"]
    Pxx_global_2 = global_features["Pxx_global_2"]
    rel_bandpower_eeg1 = global_features["rel_bandpower_eeg1"]
    rel_bandpower_eeg2 = global_features["rel_bandpower_eeg2"]

    feature_corr = method_features["feature_corr"]
    rel_bandpower_feat_eeg1 = method_features["rel_bandpower_feat_eeg1"]
    rel_bandpower_feat_eeg2 = method_features["rel_bandpower_feat_eeg2"]
    recon_corr_eeg1 = method_features["recon_corr_eeg1"]
    recon_corr_eeg2 = method_features["recon_corr_eeg2"]
    recon_var_ratio_eeg1 = method_features["recon_var_ratio_eeg1"]
    recon_var_ratio_eeg2 = method_features["recon_var_ratio_eeg2"]
    recon_rmsdiff_eeg1 = method_features["recon_rmsdiff_eeg1"]
    recon_rmsdiff_eeg2 = method_features["recon_rmsdiff_eeg2"]

    df_good_usable = quality_groups["df_good_usable"]
    df_good_strict_clean = quality_groups["df_good_strict_clean"]
    df_good_slowcorr = quality_groups["df_good_slowcorr"]
    df_good_emg = quality_groups["df_good_emg"]
    df_good_ecg = quality_groups["df_good_ecg"]
    df_good_bad_auto = quality_groups["df_good_bad_auto"]
    df_good_mixed = quality_groups["df_good_mixed"]
    df_good_wavelet_hf = quality_groups["df_good_wavelet_hf"]
    df_good_wavelet_transient = quality_groups["df_good_wavelet_transient"]
    df_good_amp_burst = quality_groups["df_good_amp_burst"]
    df_good_jump = quality_groups["df_good_jump"]
    df_good_flatline = quality_groups["df_good_flatline"]
    df_good_hf_emg = quality_groups["df_good_hf_emg"]
    df_good_low_drift = quality_groups["df_good_low_drift"]
    df_good_clipping = quality_groups["df_good_clipping"]

    run_autoreject = autoreject_result["run_autoreject"]
    autoreject_success = autoreject_result["autoreject_success"]
    autoreject_threshold_eeg = autoreject_result["autoreject_threshold_eeg"]
    autoreject_threshold_mode = autoreject_result["autoreject_threshold_mode"]
    autoreject_random_state = autoreject_result["autoreject_random_state"]
    autoreject_threshold_eeg_override = autoreject_result["autoreject_threshold_eeg_override"]
    autoreject_threshold_override_source = autoreject_result["autoreject_threshold_override_source"]
    autoreject_bad_windows = autoreject_result["autoreject_bad_windows"]
    autoreject_error = autoreject_result["autoreject_error"]

    run_ica = signal_result["run_ica"]
    run_asr = signal_result["run_asr"]
    run_ajdc = signal_result["run_ajdc"]
    run_wavelet = signal_result["run_wavelet"]
    run_adaptive_ecg = signal_result["run_adaptive_ecg"]
    ecg_indices = signal_result["ecg_indices"]
    asr_cutoff = signal_result["asr_cutoff"]
    asr_success = signal_result["asr_success"]
    asr_error = signal_result["asr_error"]
    ajdc_success = signal_result["ajdc_success"]
    ajdc_error = signal_result["ajdc_error"]
    ajdc_n_sources = signal_result["ajdc_n_sources"]
    ajdc_supp = signal_result["ajdc_supp"]
    adaptive_ecg_success = signal_result["adaptive_ecg_success"]
    adaptive_ecg_error = signal_result["adaptive_ecg_error"]
    adaptive_ecg_filter_len = signal_result["adaptive_ecg_filter_len"]
    adaptive_ecg_mu = signal_result["adaptive_ecg_mu"]
    adaptive_ecg_corr_before_eeg1 = signal_result["adaptive_ecg_corr_before_eeg1"]
    adaptive_ecg_corr_before_eeg2 = signal_result["adaptive_ecg_corr_before_eeg2"]
    adaptive_ecg_corr_after_eeg1 = signal_result["adaptive_ecg_corr_after_eeg1"]
    adaptive_ecg_corr_after_eeg2 = signal_result["adaptive_ecg_corr_after_eeg2"]
    wavelet_success = signal_result["wavelet_success"]
    wavelet_error = signal_result["wavelet_error"]
    wavelet_name = signal_result["wavelet_name"]
    wavelet_level = signal_result["wavelet_level"]
    wavelet_threshold_mode = signal_result["wavelet_threshold_mode"]
    wavelet_threshold_scale = signal_result["wavelet_threshold_scale"]

    threshold_amp_burst = artifact_thresholds["threshold_amp_burst"]
    threshold_jump = artifact_thresholds["threshold_jump"]
    threshold_flatline_std = artifact_thresholds["threshold_flatline_std"]
    threshold_flatline_ptp = artifact_thresholds["threshold_flatline_ptp"]
    threshold_hf_emg_rel = artifact_thresholds["threshold_hf_emg_rel"]
    threshold_hf_emg_line = artifact_thresholds["threshold_hf_emg_line"]
    threshold_low_drift_delta_rel = artifact_thresholds["threshold_low_drift_delta_rel"]
    threshold_low_drift_slope = artifact_thresholds["threshold_low_drift_slope"]
    threshold_clipping_ratio = artifact_thresholds["threshold_clipping_ratio"]
    low_drift_is_hard_artifact = artifact_thresholds["low_drift_is_hard_artifact"]
    use_robust_artifact_markers = artifact_thresholds["use_robust_artifact_markers"]
    wavelet_markers_are_hard_artifacts = artifact_thresholds["wavelet_markers_are_hard_artifacts"]

    metrics = {
        "EEG1_Samples": len(eeg1),
        "EEG2_Samples": len(eeg2),
        "Same Length": len(eeg1) == len(eeg2),
        "corr": corr,

        "EEG1_Delta": rel_bandpower_eeg1["Delta"],
        "EEG1_Theta": rel_bandpower_eeg1["Theta"],
        "EEG1_Alpha": rel_bandpower_eeg1["Alpha"],
        "EEG1_Beta": rel_bandpower_eeg1["Beta"],
        "EEG2_Delta": rel_bandpower_eeg2["Delta"],
        "EEG2_Theta": rel_bandpower_eeg2["Theta"],
        "EEG2_Alpha": rel_bandpower_eeg2["Alpha"],
        "EEG2_Beta": rel_bandpower_eeg2["Beta"],

        "EEG1_Windows_n": len(df_features),

        # Method specific global features
        "Feature_corr": feature_corr,

        "Feature_EEG1_Delta": rel_bandpower_feat_eeg1["Delta"],
        "Feature_EEG1_Theta": rel_bandpower_feat_eeg1["Theta"],
        "Feature_EEG1_Alpha": rel_bandpower_feat_eeg1["Alpha"],
        "Feature_EEG1_Beta": rel_bandpower_feat_eeg1["Beta"],

        "Feature_EEG2_Delta": rel_bandpower_feat_eeg2["Delta"],
        "Feature_EEG2_Theta": rel_bandpower_feat_eeg2["Theta"],
        "Feature_EEG2_Alpha": rel_bandpower_feat_eeg2["Alpha"],
        "Feature_EEG2_Beta": rel_bandpower_feat_eeg2["Beta"],

        # Reconstruction comparison to Original
        "Recon_corr_EEG1": float(recon_corr_eeg1),
        "Recon_corr_EEG2": float(recon_corr_eeg2),
        "Recon_var_ratio_EEG1": recon_var_ratio_eeg1,
        "Recon_var_ratio_EEG2": recon_var_ratio_eeg2,
        "Recon_rmsdiff_EEG1": recon_rmsdiff_eeg1,
        "Recon_rmsdiff_EEG2": recon_rmsdiff_eeg2,

        # Label: Quality
        "Windows_quality_base_good": int(df_features["quality_base_label"].sum()),
        "Windows_quality_base_bad": int(len(df_features) - df_features["quality_base_label"].sum()),
        "Autoreject_removed_quality_windows": int(
            ((df_features["quality_base_label"] == 1) & (df_features["autoreject_bad"])).sum()
        ),

        "Windows_quality_good": int(df_features["quality_label"].sum()),
        "Windows_quality_bad": int(len(df_features) - df_features["quality_label"].sum()),

        # Label: Usable
        "Windows_usable": int(df_features["usable_label"].sum()),
        "Windows_not_usable": int(len(df_features) - df_features["usable_label"].sum()),

        # Label: Strict Clean
        "Windows_strict_clean": int(df_features["strict_clean_label"].sum()),
        "Windows_not_strict_clean": int(len(df_features) - df_features["strict_clean_label"].sum()),

        # Quality-good via Marker / Artifact Type
        "Quality Good & No Hard Artifact": len(df_good_usable),
        "Quality Good & Strict Clean": len(df_good_strict_clean),
        "Quality Good & SLOW_CORR": len(df_good_slowcorr),
        "Quality Good & ECG": len(df_good_ecg),
        "Quality Good & EMG": len(df_good_emg),
        "Quality Good & BAD_AUTO": len(df_good_bad_auto),
        "Quality Good & Mixed": len(df_good_mixed),

        # Total marker
        "All SLOW_CORR": int(df_features["has_slow_corr"].sum()),
        "All Hard Artifact": int(df_features["has_hard_artifact"].sum()),

        # Autoreject
        "Autoreject_run": run_autoreject,
        "Autoreject_available": use_autoreject_available,
        "Autoreject_success": autoreject_success,
        "Autoreject_threshold_eeg": autoreject_threshold_eeg if autoreject_threshold_eeg is not None else "",
        "Autoreject_threshold_mode": autoreject_threshold_mode,
        "Autoreject_random_state": autoreject_random_state,
        "Autoreject_threshold_override_requested": (
            autoreject_threshold_eeg_override
            if autoreject_threshold_eeg_override is not None
            else ""
        ),
        "Autoreject_threshold_override_source": (
            autoreject_threshold_override_source
        ),
        "Autoreject_bad_windows": int(autoreject_bad_windows.sum()),
        "Autoreject_error": autoreject_error if autoreject_error is not None else "",

        # Signal Separation
        "Signal_method": signal_method,

        # ICA
        "ICA_run": run_ica,
        "ICA_ECG_components_n": len(ecg_indices),

        # ASR
        "ASR_run": run_asr,
        "ASR_available": use_asr_available,
        "ASR_used_for_features": signal_method == "asr" and asr_success,
        "ASR_cutoff": asr_cutoff,
        "ASR_success": asr_success,
        "ASR_error": asr_error if asr_error is not None else "",

        # AJDC
        "AJDC_run": run_ajdc,
        "AJDC_available": use_ajdc_available,
        "AJDC_success": ajdc_success,
        "AJDC_used_for_features": signal_method == "ajdc" and ajdc_success,
        "AJDC_n_sources": ajdc_n_sources,
        "AJDC_suppressed_sources": ",".join(map(str, ajdc_supp)),
        "AJDC_variant": filename_col,
        "AJDC_error": ajdc_error if ajdc_error is not None else "",

        # Adaptive ECG
        "Adaptive_ECG_run": run_adaptive_ecg,
        "Adaptive_ECG_success": adaptive_ecg_success,
        "Adaptive_ECG_used_for_features": signal_method == "adaptive_ecg" and adaptive_ecg_success,
        "Adaptive_ECG_filter_len": adaptive_ecg_filter_len,
        "Adaptive_ECG_mu": adaptive_ecg_mu,
        "Adaptive_ECG_corr_before_EEG1": adaptive_ecg_corr_before_eeg1,
        "Adaptive_ECG_corr_before_EEG2": adaptive_ecg_corr_before_eeg2,
        "Adaptive_ECG_corr_after_EEG1": adaptive_ecg_corr_after_eeg1,
        "Adaptive_ECG_corr_after_EEG2": adaptive_ecg_corr_after_eeg2,
        "Adaptive_ECG_corr_delta_EEG1": (
            adaptive_ecg_corr_after_eeg1 - adaptive_ecg_corr_before_eeg1
            if not np.isnan(adaptive_ecg_corr_before_eeg1) and not np.isnan(adaptive_ecg_corr_after_eeg1)
            else np.nan
        ),
        "Adaptive_ECG_corr_delta_EEG2": (
            adaptive_ecg_corr_after_eeg2 - adaptive_ecg_corr_before_eeg2
            if not np.isnan(adaptive_ecg_corr_before_eeg2) and not np.isnan(adaptive_ecg_corr_after_eeg2)
            else np.nan
        ),
        "Adaptive_ECG_error": adaptive_ecg_error if adaptive_ecg_error is not None else "",

        # Wavelet
        "Wavelet_run": run_wavelet,
        "Wavelet_available": use_wavelet_available,
        "Wavelet_success": wavelet_success,
        "Wavelet_used_for_features": signal_method == "wavelet" and wavelet_success,
        "Wavelet_name": wavelet_name,
        "Wavelet_level": wavelet_level,
        "Wavelet_threshold_mode": wavelet_threshold_mode,
        "Wavelet_threshold_scale": float(wavelet_threshold_scale),
        "Wavelet_markers_are_hard_artifacts": wavelet_markers_are_hard_artifacts,
        "Wavelet_error": wavelet_error if wavelet_error is not None else "",

        "All WAVELET_HF": int(df_features["has_wavelet_hf"].sum()),
        "All WAVELET_TRANSIENT": int(df_features["has_wavelet_transient"].sum()),
        "Quality Good & WAVELET_HF": len(df_good_wavelet_hf),
        "Quality Good & WAVELET_TRANSIENT": len(df_good_wavelet_transient),

        # Robust Artifact marker
        "Robust_markers_enabled": use_robust_artifact_markers,

        "All AMP_BURST": int(df_features["has_amp_burst"].sum()),
        "All JUMP": int(df_features["has_jump"].sum()),
        "All FLATLINE": int(df_features["has_flatline"].sum()),
        "All HF_EMG": int(df_features["has_hf_emg"].sum()),
        "All LOW_DRIFT": int(df_features["has_low_drift"].sum()),
        "All CLIPPING": int(df_features["has_clipping"].sum()),

        "Quality Good & AMP_BURST": int(len(df_good_amp_burst)),
        "Quality Good & JUMP": int(len(df_good_jump)),
        "Quality Good & FLATLINE": int(len(df_good_flatline)),
        "Quality Good & HF_EMG": int(len(df_good_hf_emg)),
        "Quality Good & LOW_DRIFT": int(len(df_good_low_drift)),
        "Quality Good & CLIPPING": int(len(df_good_clipping)),

        "Threshold_AMP_BURST_ptp": float(threshold_amp_burst),
        "Threshold_JUMP_norm": float(threshold_jump),
        "Threshold_FLATLINE_std": float(threshold_flatline_std),
        "Threshold_FLATLINE_ptp": float(threshold_flatline_ptp),
        "Threshold_HF_EMG_rel": float(threshold_hf_emg_rel),
        "Threshold_HF_EMG_line": float(threshold_hf_emg_line),
        "Threshold_LOW_DRIFT_delta_rel": float(threshold_low_drift_delta_rel),
        "Threshold_LOW_DRIFT_slope": float(threshold_low_drift_slope),
        "Threshold_CLIPPING_ratio": float(threshold_clipping_ratio),
        "LOW_DRIFT_is_hard_artifact": low_drift_is_hard_artifact,

        "ECG_ref_available": ecg_ref is not None,
        # "ECG_col": ecg_ii_col if ecg_ii_col is not None else "None",  # Name: ECG-Spalte in Data
        "HR_ref": hr_ref,
    }

    return metrics


def plot_results(
    *,
    make_plots,
    t,
    eeg1,
    eeg2,
    diff,
    global_features,
    segment_features,
    window_features,
    df_features,
    ecg_ref,
):
    # Create the existing optional diagnostic plots
    if not make_plots:
        return

    import matplotlib.pyplot as plt

    f_psd_global_1 = global_features["f_psd_global_1"]
    Pxx_global_1 = global_features["Pxx_global_1"]
    f_psd_global_2 = global_features["f_psd_global_2"]
    Pxx_global_2 = global_features["Pxx_global_2"]
    f_diff = global_features["f_diff"]
    Pxx_diff = global_features["Pxx_diff"]
    f_start = segment_features["f_start"]
    Pxx_start = segment_features["Pxx_start"]
    f_end = segment_features["f_end"]
    Pxx_end = segment_features["Pxx_end"]

    # --- Time ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2,1,1)
    plt.plot(t, eeg1, label="EEG_1")
    plt.plot(t, eeg2, label="EEG_2", alpha=0.7)
    plt.axhline(0, linestyle="--")
    # plt.xlim(0, 10)
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("EEG Time dimension (both channels)")

    plt.subplot(2,1,2)
    plt.plot(t, diff, color="purple")
    plt.axhline(0, linestyle="--")
    # plt.xlim(0, 10)
    plt.ylabel("Amplitude")
    plt.title("Difference (EEG1 - EEG2)")

    plt.tight_layout()

    # --- PSD ---
    plt.figure(figsize=(10, 8))

    plt.subplot(2,1,1)
    plt.semilogy(f_psd_global_1, Pxx_global_1, label="EEG_1")
    plt.semilogy(f_psd_global_2, Pxx_global_2, label="EEG_2", alpha=0.7)
    plt.semilogy(f_diff, Pxx_diff, label="Diff", linestyle="--")
    plt.legend()
    plt.xlim(0,50)
    plt.title("PSD Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    plt.subplot(2,1,2)
    plt.semilogy(f_start, Pxx_start, label="First 30s")
    plt.semilogy(f_end, Pxx_end, label="Last 30s", alpha=0.7)
    plt.xlim(0,50)
    plt.legend()
    plt.title("PSD: First and last 30 seconds(EEG1)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    plt.tight_layout()

    # --- Window ---
    corrs = [wf["corr"] for wf in window_features]

    plt.figure()
    plt.plot(corrs)
    plt.title("Correlation over time (window)")
    plt.xlabel("Window")
    plt.ylabel("Correlation")

    # --- Feature Visualization ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Correlation
    corr_plot = df_features["corr"].values
    var_plot = df_features["var1"].values
    labels_plot = df_features["label"].values

    axs[0].plot(corr_plot, label="Correlation")
    axs[0].set_ylim(-1, 1)
    axs[0].set_title("Correlation")
    axs[0].set_ylabel("Corr")

    # color Good/Bad
    for i in range(len(labels_plot)):
        if labels_plot[i] == 1:
            axs[0].axvspan(i, i+1, color="green", alpha=0.1)
        else:
            axs[0].axvspan(i, i+1, color="red", alpha=0.05)

    axs[0].legend()

    # Variance
    axs[1].plot(var_plot, label="Varianz EEG1", color="orange")
    axs[1].set_title("Variance EEG1")
    axs[1].set_ylabel("Var")
    axs[1].set_xlabel("Window")

    for i in range(len(labels_plot)):
        if labels_plot[i] == 1:
            axs[1].axvspan(i, i+1, color="green", alpha=0.1)
        else:
            axs[1].axvspan(i, i+1, color="red", alpha=0.05)

    axs[1].legend()

    plt.tight_layout()

    # # Label Visualization
    # plt.figure(figsize=(10,5))
    # plt.plot(df_features["corr"], label="Correlation")
    # plt.plot(df_features["label"], label="Good(1)/Bad(0)", alpha=0.7)
    # plt.legend()
    # plt.title("Fensterbewertung")
    # plt.xlabel("Fenster")

    # --- ECG ---
    # ECG_II
    if ecg_ref is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(t[:5000], ecg_ref[:5000])
        plt.title("ECG_II Reference Signal – first section")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    plt.show()
