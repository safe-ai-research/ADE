# Paths and user-adjustable batch parameters for the ADE pipeline

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

base_path = PROJECT_ROOT / "Data"    # Path: Data folder (>Path</Dataset/FilenameFileextension)
file_path_sheet = PROJECT_ROOT / "results" / "EEG_comparison_data.xlsx" # Path to Excel table (File)
artifact_export_root = PROJECT_ROOT / "results" / "Artifact_annotations_pipeline"


def format_parameter_tag(value):
    # Convert numeric parameters into filename-safe tags
    return f"{float(value):g}".replace(".", "p")


def build_artifact_variant(params):
    # Build a unique export-folder name for a signal configuration

    # Optional explicit override for named experiment versions
    explicit_variant = params.get("artifact_variant")
    if explicit_variant:
        return str(explicit_variant)

    signal_method = params.get("signal_method", "original")

    if signal_method == "wavelet":
        method_tag = (
            f"wavelet_{params.get('wavelet_name', 'db4')}"
            f"_l{params.get('wavelet_level', 5)}"
            f"_{params.get('wavelet_threshold_mode', 'soft')}"
            f"_s{format_parameter_tag(params.get('wavelet_threshold_scale', 0.1))}"
        )

    elif signal_method == "adaptive_ecg":
        method_tag = (
            f"adaptive_ecg"
            f"_l{params.get('adaptive_ecg_filter_len', 32)}"
            f"_mu{format_parameter_tag(params.get('adaptive_ecg_mu', 0.05))}"
        )

    elif signal_method == "asr":
        method_tag = (
            f"asr"
            f"_c{format_parameter_tag(params.get('asr_cutoff', 20))}"
        )

    elif signal_method == "ajdc":
        if params.get("ajdc_auto_suppress", False):
            method_tag = "ajdc_auto"
        else:
            suppressed_sources = params.get("ajdc_manual_supp", []) or []

            if suppressed_sources:
                removed_tag = "-".join(
                    str(source)
                    for source in suppressed_sources
                )
                method_tag = f"ajdc_rm{removed_tag}"
            else:
                method_tag = "ajdc_all"

    else:
        # original, ica_ecg, etc.
        method_tag = signal_method

    window_tag = (
        f"w{format_parameter_tag(params.get('window_sec', 5))}"
        f"_o{format_parameter_tag(params.get('window_overlap', 0.5))}"
    )

    return f"{method_tag}_{window_tag}"


# Parameters
parameter_sets = [
    {"signal_method": "original"},
    # # --- adaptive ecg filter ---
    # {"signal_method": "adaptive_ecg", "adaptive_ecg_filter_len": 16, "adaptive_ecg_mu": 0.01},
    # {"signal_method": "adaptive_ecg", "adaptive_ecg_filter_len": 32, "adaptive_ecg_mu": 0.05},
    # {"signal_method": "adaptive_ecg", "adaptive_ecg_filter_len": 32, "adaptive_ecg_mu": 0.10},
    # # --- wavelet ---
    # # Threshold: 0.1
    # {"signal_method": "wavelet", "wavelet_name": "db4", "wavelet_threshold_scale": 0.1},
    # {"signal_method": "wavelet", "wavelet_name": "db7", "wavelet_threshold_scale": 0.1},
    # {"signal_method": "wavelet", "wavelet_name": "sym4", "wavelet_threshold_scale": 0.1},
    # {"signal_method": "wavelet", "wavelet_name": "sym9", "wavelet_threshold_scale": 0.1},
    # {"signal_method": "wavelet", "wavelet_name": "coif3", "wavelet_threshold_scale": 0.1},
    # # Threshold: 0.25
    # {"signal_method": "wavelet", "wavelet_name": "db4", "wavelet_threshold_scale": 0.25},
    # {"signal_method": "wavelet", "wavelet_name": "db7", "wavelet_threshold_scale": 0.25},
    # {"signal_method": "wavelet", "wavelet_name": "sym4", "wavelet_threshold_scale": 0.25},
    # {"signal_method": "wavelet", "wavelet_name": "sym9", "wavelet_threshold_scale": 0.25},
    # {"signal_method": "wavelet", "wavelet_name": "coif3", "wavelet_threshold_scale": 0.25},
    # # --- AJDC ---
    # # all
    # {"signal_method": "ajdc", "ajdc_auto_suppress": False, "ajdc_manual_supp": [],
    # "ajdc_fmin": 1.0, "ajdc_fmax": 30.0, "ajdc_window_sec": 1.0, "ajdc_overlap": 0.5,
    # "ajdc_dim_red": {"n_components": 2},},
    # # rm0
    # {"signal_method": "ajdc", "ajdc_auto_suppress": False, "ajdc_manual_supp": [0],
    # "ajdc_fmin": 1.0, "ajdc_fmax": 30.0, "ajdc_window_sec": 1.0, "ajdc_overlap": 0.5,
    # "ajdc_dim_red": {"n_components": 2},},
    # # rm1
    # {"signal_method": "ajdc", "ajdc_auto_suppress": False, "ajdc_manual_supp": [1],
    # "ajdc_fmin": 1.0, "ajdc_fmax": 30.0, "ajdc_window_sec": 1.0, "ajdc_overlap": 0.5,
    # "ajdc_dim_red": {"n_components": 2},},
    # # --- ICA ECG ---
    # {"signal_method": "ica_ecg",}
]


common_params = {
    # Classic Filter
    "use_dc": True,                 # DC (Mean)
    "use_detrend": True,            # Trend
    "use_notch": False,             # Notch-Filter (50 Hz, EU Main Voltage)
    "use_highpass": True,           # High Pass (Default: 0.5 Hz)
    "use_lowpass": False,           # Low Pass (Default: 30 Hz)
    "use_normalization": False,     # Normalization (z-score)

    # Windowing
    "window_sec": 5,                # Window width in seconds (Default: 5)
    "window_overlap": 0.0,          # Window overlap (Default: 0.5)

    # Autoreject
    "run_autoreject": True,
    # "autoreject_window_sec": 5,
    # "autoreject_overlap": 0.5,

    # Marker Thresholds
    "threshold_delta_slowcorr": 0.60,
    "threshold_corr_slowcorr": 0.30,
    "threshold_beta_emg_rel": 0.25,
    "threshold_var_emg_quantile": 0.75,

    # Robust Artifact Markers
    "use_robust_artifact_markers": True,    # Set Robust Artifact Markers as Hard Artifacts?

    "threshold_amp_burst_quantile": 0.95,
    "threshold_jump_quantile": 0.95,
    "threshold_flatline_quantile": 0.01,

    "threshold_hf_emg_rel": 0.15,           # Default: 0.15; or less sensitive, like 0.20
    "threshold_hf_emg_line_quantile": 0.75, # Default: 0.75; or less sensitive, like 0.85

    "threshold_low_drift_delta_rel": 0.80,
    "threshold_low_drift_slope_quantile": 0.90,

    "threshold_clipping_ratio": 0.02,
    "low_drift_is_hard_artifact": False,

    # Adaptive ECG Filtering
    "adaptive_ecg_filter_len": 32,
    "adaptive_ecg_mu": 0.05,

    # Wavelet Marker
    "use_wavelet_marker_features": True,
    "threshold_wavelet_hf_quantile": 0.90,
    "threshold_wavelet_transient_quantile": 0.90,
    "wavelet_markers_are_hard_artifacts": False,

    # Plots
    "make_plots": False,

    # Excel-Export of detected artifact windows
    "export_artifact_windows": False,
}
