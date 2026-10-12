# ADE - An Artifact Detection Framework for 2-Channel EEG

A configurable Python pipeline for preprocessing two-channel EEG recordings, extracting signal-quality features, detecting potential artifact windows, and comparing multiple signal-processing methods.

The project is intended as a preprocessing and exploratory analysis framework for EEG data used in downstream machine-learning workflows. It does **not** provide clinical diagnoses and its automatically generated labels should not be treated as expert ground truth.

During the creation of this project, the public dataset DOSE-I was used. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18483292.svg)](https://doi.org/10.5281/zenodo.18483292)

This project is licensed under [`Apache License 2.0`](LICENSE).

## Features

The pipeline currently supports:

- CSV and compressed CSV (`.csv.gz`) input files
- automatic batch discovery of recordings
- resampling to a fixed sampling rate of 125 Hz
- configurable basic preprocessing:
  - detrending
  - DC removal
  - high-pass filtering
  - low-pass filtering
  - 50 Hz notch filtering
  - optional z-score normalization
- multiple signal-processing variants:
  - original preprocessed EEG
  - ECG-informed adaptive filtering using NLMS
  - wavelet denoising
  - ICA-based ECG component removal
  - AJDC source separation and reconstruction
  - experimental ASR processing
- window-based feature extraction, including channel-asymmetry features
- heuristic artifact-marker detection
- channel-specific and combined signal-quality labels
- Excel exports for recording-level metrics, detected artifact windows, all-window features, and effective detection thresholds
- separate annotation directories for different processing and windowing configurations
- optional diagnostic plots

The implementation is split into the `ade` package. Batch settings are defined in [`ade/config.py`](ade/config.py), while [`run_batch.py`](run_batch.py) remains the executable entry point.

## Processing Overview

For every recording, the pipeline performs the following main steps:

1. Load the CSV or compressed CSV file.
2. Parse timestamps and select the EEG channels.
3. Detect an available ECG waveform column and derive a heart-rate reference where available.
4. Resample the signals onto a uniform 125 Hz time axis using linear interpolation.
5. Apply the configured basic preprocessing.
6. Calculate baseline global features and create the initial windows.
7. Apply the selected signal-processing method and recreate the EEG windows from its output.
8. Calculate method-specific global features and reconstruction comparisons.
9. Estimate or apply an AutoReject peak-to-peak threshold, if enabled.
10. Calculate window features and engineered cross-channel features.
11. Detect potential artifact markers.
12. Assign quality, usability, and strict-clean labels per channel and for both channels together.
13. Optionally export window annotations, features, and effective thresholds.
14. Calculate first-versus-last segment comparisons, optionally display diagnostic plots, and update the recording-level comparison table.

[`ade/pipeline.py`](ade/pipeline.py) coordinates these steps through functions in the individual modules. Baseline global features describe the conventionally preprocessed EEG; method-specific global features and window labels describe the selected method's output.

## Project Structure

| Path | Responsibility |
|---|---|
| [`run_batch.py`](run_batch.py) | Select recordings, merge parameters, create export paths, and execute the batch |
| [`pipeline.py`](pipeline.py) | Compatibility import for existing callers of `run_pipeline` |
| [`ade/__init__.py`](ade/__init__.py) | Expose `run_pipeline` from the package |
| [`ade/config.py`](ade/config.py) | Project paths, common parameters, method variants, and annotation-directory names |
| [`ade/pipeline.py`](ade/pipeline.py) | Coordinate the processing sequence for one recording |
| [`ade/data_io.py`](ade/data_io.py) | Discover and load recordings, write Excel outputs, and display plots |
| [`ade/signal_processing.py`](ade/signal_processing.py) | Resample signals, apply basic preprocessing, and run the selected signal method |
| [`ade/windowing.py`](ade/windowing.py) | Create signal windows, sample positions, and absolute timestamps |
| [`ade/features.py`](ade/features.py) | Calculate global, segment, window, and engineered features |
| [`ade/artifacts.py`](ade/artifacts.py) | Apply AutoReject detection, artifact markers, and hard-artifact rules |
| [`ade/quality.py`](ade/quality.py) | Assign quality, usable, and strict-clean labels and form summary groups |
| [`requirements.txt`](requirements.txt) | Runtime dependencies |
| `Data/<dataset>/` | Local input recordings, such as `Data/DOSE-I/10-006.csv` |
| `results/EEG_comparison_data.xlsx` | Recording-level comparison table |
| `results/Artifact_annotations_pipeline/<variant>/` | Optional annotation workbooks for each processing configuration |

The `Data` and `results` paths are resolved relative to the project root in `ade/config.py`. Absolute user-specific paths are therefore not required.

> The EEG datasets are not included in this repository. Make sure that you are permitted to use and store the data before adding recordings to the project.

## Expected Input Format

Input files must use either the `.csv` or `.csv.gz` extension.

The following columns are required:

| Column | Purpose |
|---|---|
| `Time` | Timestamp of the sample |
| `Intellivue/EEG_1` | First EEG channel |
| `Intellivue/EEG_2` | Second EEG channel |

The pipeline additionally checks for these optional columns:

| Column | Purpose |
|---|---|
| `Intellivue/ECG_II` | ECG waveform used as a reference |
| `Intellivue/ECG_II_WAV` | Alternative ECG waveform column |
| `Intellivue/ECG_HR` | ECG-derived heart rate |
| `Intellivue/PLETH_HR` | Plethysmography-derived heart rate |

At least the timestamp and both EEG columns must be present.

## Requirements

The provided runtime dependency file targets **Python 3.12**. The core packages are:

```text
numpy
pandas
scipy
matplotlib
mne
openpyxl
scikit-learn
```

Additional detection and processing functionality uses:

```text
autoreject
PyWavelets
pyriemann
asrpy
```

`autoreject` provides the automatic rejection threshold. `PyWavelets` provides both wavelet marker features and wavelet denoising, `pyriemann` provides AJDC, and `asrpy` provides ASR. `scikit-learn` is required by MNE's FastICA path. ICLabel is currently only present as commented-out example code and is not an active dependency.

Bandpower calculation uses `numpy.trapezoid`, so use NumPy 2.0 or newer. Package versions are not pinned in `requirements.txt`.

### Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the provided runtime dependencies from the project directory:

```powershell
python -m pip install -r requirements.txt
```

For a smaller installation, install the core packages first and add the detection or processing packages you need:

```powershell
python -m pip install "numpy>=2.0" pandas scipy matplotlib mne openpyxl scikit-learn
python -m pip install autoreject PyWavelets
python -m pip install pyriemann asrpy
```

The default configuration enables AutoReject and wavelet marker features, so install `autoreject` and `PyWavelets` to retain that functionality.

To preserve the exact environment used for an experiment without replacing the provided dependency file:

```powershell
python -m pip freeze > requirements-lock.txt
```

## Configuration

Configuration is split between two files:

- [`ade/config.py`](ade/config.py): paths, `common_params`, `parameter_sets`, and annotation-directory naming.
- [`run_batch.py`](run_batch.py): `datasets_to_run` and, inside `main()`, the dataset folder and exclusion list used for automatic discovery.

The checked-in configuration runs `DOSE-I/10-006.csv` with the `original` method. It uses five-second windows with **no overlap**, enables AutoReject and wavelet marker features, and disables both plots and window-annotation export.

### Data Paths

The paths are defined in `ade/config.py`. Its parent directory is the `ade` package, so the project root is one level above it:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

base_path = PROJECT_ROOT / "Data"
file_path_sheet = PROJECT_ROOT / "results" / "EEG_comparison_data.xlsx"
artifact_export_root = (
    PROJECT_ROOT
    / "results"
    / "Artifact_annotations_pipeline"
)
```

### Dataset Selection

To process individual recordings, edit `datasets_to_run` in `run_batch.py`:

```python
datasets_to_run = [
    ("DOSE-I", "10-006", ".csv"),
    ("DOSE-I", "10-020", ".csv.gz"),
]
```

Each tuple contains:

```text
(dataset folder, filename without extension, file extension)
```

When `datasets_to_run` is empty, the batch runner scans the configured dataset directory for `.csv` and `.csv.gz` files. Set the folder name inside `main()`:

```python
datasets_to_run = []
dataset_name = "DOSE-I"
```

For automatic discovery, files can be excluded by their filename without the extension. Edit `exclude_files` inside `main()`:

```python
exclude_files = {
    "10-123",
}
```

### Signal-Processing Methods

The processing variants are configured in `parameter_sets` in `ade/config.py`.

Process only the original preprocessed signal:

```python
parameter_sets = [
    {"signal_method": "original"},
]
```

Run several variants:

```python
parameter_sets = [
    {"signal_method": "original"},
    {
        "signal_method": "adaptive_ecg",
        "adaptive_ecg_filter_len": 32,
        "adaptive_ecg_mu": 0.05,
    },
    {
        "signal_method": "wavelet",
        "wavelet_name": "db4",
        "wavelet_threshold_scale": 0.1,
    },
    {
        "signal_method": "ica_ecg",
    },
]
```

Supported method identifiers are:

| Identifier | Description |
|---|---|
| `original` | Uses the conventionally preprocessed EEG |
| `adaptive_ecg` | Applies an ECG-referenced normalized LMS filter |
| `wavelet` | Applies wavelet denoising |
| `ica_ecg` | Uses ICA to identify and remove ECG-related components |
| `ajdc` | Applies AJDC source separation and signal reconstruction |
| `asr` | Experimental Artifact Subspace Reconstruction path |

### Common Parameters

Parameters shared by all runs are stored in `common_params` in `ade/config.py`.

Example using the current preprocessing and windowing defaults, with window export enabled:

```python
common_params = {
    "use_dc": True,
    "use_detrend": True,
    "use_notch": False,
    "use_highpass": True,
    "use_lowpass": False,
    "use_normalization": False,

    "run_autoreject": True,
    "window_sec": 5,
    "window_overlap": 0.0,

    "make_plots": False,

    "export_artifact_windows": True,
}
```

This is an excerpt; retain the marker settings already present in `common_params`. To enable overlapping windows, set `"window_overlap": 0.5` for 50% overlap. Use a positive `window_sec` and an overlap in the range `0 <= window_overlap < 1` that gives a nonzero sample step.

Method-specific dictionaries in `parameter_sets` override matching values in `common_params` before each pipeline run. `run_batch.py` then sets `artifact_export_dir` to `artifact_export_root / build_artifact_variant(params_run)`.

### Annotation Directory Names

`build_artifact_variant()` in `ade/config.py` combines the method settings with the window duration and overlap. Decimal points are replaced with `p` for path names.

Examples:

| Configuration | Directory name |
|---|---|
| Original, 5 s, no overlap | `original_w5_o0` |
| Original, 5 s, 50% overlap | `original_w5_o0p5` |
| Wavelet `db4`, level 5, soft threshold, scale 0.1, 5 s, no overlap | `wavelet_db4_l5_soft_s0p1_w5_o0` |
| Adaptive ECG, filter length 32, step size 0.05, 5 s, no overlap | `adaptive_ecg_l32_mu0p05_w5_o0` |

For a named experiment, set `"artifact_variant": "experiment_name"` in a parameter dictionary. This replaces the generated directory name. The generated name does not encode every preprocessing or marker setting; use an explicit name or a separate output root when those settings differ between experiments.

### Detection Settings

Marker thresholds and hard-artifact switches are configured in `common_params`. Quantile thresholds pool the corresponding feature values from both EEG channels within the current recording.

- `use_robust_artifact_markers` enables the amplitude, jump, flatline, high-frequency, drift, and clipping markers.
- `use_wavelet_marker_features` enables wavelet feature calculation and the associated markers.
- `low_drift_is_hard_artifact` and `wavelet_markers_are_hard_artifacts` decide whether those markers also exclude a window from `usable_label`; both are `False` in the current configuration.
- `run_autoreject` enables peak-to-peak rejection. By default, the threshold is estimated from the current method's windows using `autoreject_random_state=42`.

To reuse a known AutoReject threshold instead of estimating it again, set `autoreject_threshold_eeg_override` to a finite positive value in the signal's amplitude units. An optional `autoreject_threshold_override_source` records its origin. The override still requires `run_autoreject=True`.

The quality-base correlation threshold (`0.3`) and variance quantile (`0.75`) are currently defined directly in `ade/quality.py`, rather than exposed through `common_params`.

## Running the Pipeline

From the project directory:

```powershell
python run_batch.py
```

In Visual Studio, `run_batch.py` can be selected as the startup file and launched with or without the debugger.

The batch runner displays:

- the discovered recording count
- the first and last discovered files
- the current run number
- the active signal-processing method
- processing summaries
- failed runs at the end of the batch

File-discovery summaries are printed when automatic discovery is used. Notification sounds are enabled when Windows' `winsound` module is available. Console-title updates still use the Windows `title` command and may print shell errors on other systems.

### Calling the Pipeline from Python

The package exposes the same `run_pipeline` entry point:

```python
from ade import run_pipeline
from ade.config import (
    artifact_export_root,
    base_path,
    build_artifact_variant,
    common_params,
    file_path_sheet,
)

params = {**common_params, "signal_method": "original"}
params["artifact_export_dir"] = (
    artifact_export_root / build_artifact_variant(params)
)
file_path_sheet.parent.mkdir(parents=True, exist_ok=True)

run_pipeline(
    dataset="DOSE-I",
    filename="10-006",
    file_ending=".csv",
    params=params,
    base_path=base_path,
    file_path_sheet=file_path_sheet,
)
```

Existing scripts can continue to use `from pipeline import run_pipeline`; the root-level `pipeline.py` forwards this import to `ade.pipeline`. The function writes its configured outputs and currently has no explicit return value.

## Extracted Features

The pipeline calculates recording-level and window-level features, including:

### Frequency-domain features

- absolute delta, theta, alpha, and beta bandpower
- relative delta, theta, alpha, and beta bandpower
- high-frequency power between 30 and 45 Hz
- Welch power spectral density estimates

### Time-domain features

- variance
- standard deviation
- median absolute deviation
- peak-to-peak amplitude
- maximum absolute amplitude
- normalized maximum sample jump
- normalized line length
- normalized trend slope
- approximate clipping ratio

### Relationship features

- correlation between the two EEG channels
- correlation between EEG and ECG where available
- reconstruction correlation and variance ratios for processed signals
- reconstruction root-mean-square differences from the preprocessed baseline
- absolute differences between the channels' variance, standard deviation, and peak-to-peak amplitude (`var_diff`, `std_diff`, `ptp_diff`)

### Wavelet features

- relative detail energy
- relative high-frequency detail energy
- detail-energy entropy
- normalized maximum detail coefficient
- detail-coefficient kurtosis

## Artifact Markers

The pipeline can assign one or more potential markers to each window:

| Marker | Interpretation |
|---|---|
| `SLOW_CORR` | Strong slow activity combined with inter-channel correlation |
| `EMG` | Increased beta activity and variance |
| `ECG` | ECG-like periodic activity detected in the EEG |
| `BAD_AUTO` | Window rejected by the global AutoReject threshold |
| `AMP_BURST` | Unusually large peak-to-peak amplitude |
| `JUMP` | Abrupt sample-to-sample change |
| `FLATLINE` | Very low variance or peak-to-peak amplitude |
| `HF_EMG` | Increased 30–45 Hz activity and line length |
| `LOW_DRIFT` | Slow dominance combined with a strong window trend |
| `CLIPPING` | Many values close to the local minimum or maximum |
| `WAVELET_HF` | High relative energy in high-frequency wavelet details |
| `WAVELET_TRANSIENT` | Unusually large wavelet detail coefficients |
| `NO_MARKER` | No configured marker was detected |

Most markers are stored separately for EEG1 and EEG2, as well as in a combined flag that is set when either channel is flagged. `SLOW_CORR` is a shared marker based on average relative delta power and inter-channel correlation. `NO_MARKER` is the internal artifact-type value for a window with no marker; it is not a separate exported marker column.

Several thresholds are quantile-based and are calculated from the windows of the current recording and selected signal method. Effective thresholds are included in the optional annotation workbook's `thresholds` sheet.

These labels are heuristic indicators. They are intended for data exploration, preprocessing, and comparison with expert annotations—not as verified clinical artifact labels.

## Window Labels

The pipeline generates binary labels for each channel (`*_eeg1`, `*_eeg2`) and for the pair. A combined label is `1` only when both channel-specific labels are `1`.

### `quality_base_label`

A window is considered technically viable when:

- the absolute channel correlation exceeds `0.3`, and
- that channel's variance remains below the 75th percentile pooled from both channels' window variances.

### `quality_label`

Extends `quality_base_label` with an AutoReject veto.

### `usable_label`

A window must:

- pass `quality_label`, and
- contain no marker currently classified as a hard artifact.

Slow correlated activity may remain usable because it can also reflect physiological slow EEG. `LOW_DRIFT` and the wavelet markers also remain non-hard by default; their hard-artifact switches can be enabled in `ade/config.py`.

### `strict_clean_label`

A window must:

- pass `quality_label`, and
- contain no detected marker at all.

The legacy `label` column is retained internally as an alias of the combined `quality_label`. `is_strict_clean` only indicates the absence of markers; `strict_clean_label` additionally requires the quality criteria to pass.

## Output Files

### Recording-Level Comparison Table

```text
results/EEG_comparison_data.xlsx
```

Each processed recording and method variant is written under a method-specific column name. The table includes:

- sample and window counts
- global EEG features
- quality and artifact counts
- method parameters
- method success states
- reconstruction metrics
- processing errors where applicable

Method-specific suffixes are added to the output column names.

Examples:

```text
10-006_original
10-006_wavelet_db4_s0p1
10-006_adaptive_ecg_l32_mu0p05
10-006_ajdc_all
10-006_ajdc_rm0
```

A subsequent run with the same column name replaces that column. Comparison-column names do not encode all preprocessing, windowing, or marker settings; use a different `file_path_sheet` for experiments that must remain separate.

### Artifact-Window Tables

```text
results/Artifact_annotations_pipeline/<variant>/<recording>_artifacts.xlsx
```

With the current window settings, enabling `export_artifact_windows` produces, for example:

```text
results/Artifact_annotations_pipeline/original_w5_o0/10-006_artifacts.xlsx
```

Each workbook contains three sheets:

| Sheet | Contents |
|---|---|
| `artifacts` | Only windows containing at least one marker, with timestamps, combined and channel-specific marker flags, and quality labels |
| `windows` | All processed windows, including marker flags, quality labels, and exported numeric features |
| `thresholds` | Effective marker and quality thresholds, with their fixed or recording-adaptive basis and AutoReject metadata |

`begin_idx` and `end_idx` contain absolute timestamps. Marker flags and labels are written as binary values. Exported quality columns use uppercase names such as `QUALITY_LABEL`, `USABLE_LABEL_EEG1`, and `STRICT_CLEAN_LABEL_EEG2`; numeric feature names such as `var1` and `var_diff` remain lowercase.

The workbook keeps `artifacts` as its first sheet for compatibility with readers that load the first sheet by default. Read the `windows` sheet explicitly to obtain all windows and their features.

Annotation export is enabled with `"export_artifact_windows": True` and is available for every successfully executed signal method. It is skipped when a method fails or is unavailable, so the original fallback signal is not exported as a successful result of that method. Existing workbooks at the same path are replaced on a subsequent export.

## Optional Plots

Set:

```python
common_params["make_plots"] = True
```

to display diagnostic figures, including:

- both EEG channels in the time domain
- the inter-channel difference signal
- global power spectral densities
- first-versus-last segment comparison
- window-wise channel correlation
- window-wise variance and quality labels
- the ECG reference signal when available

## Optional Dependencies and Fallback Behaviour

AutoReject, PyWavelets, AJDC, and ASR imports are handled defensively.

- If AutoReject is unavailable or fails, its rejection flags remain unset and the status/error is recorded. A fixed threshold override can still be applied without the `autoreject` package.
- If PyWavelets is unavailable, wavelet marker features use placeholder values and the associated markers are disabled. The wavelet-denoising method falls back to the original preprocessed EEG.
- Unavailable or failed ASR, AJDC, wavelet, or adaptive-ECG processing falls back to the original preprocessed EEG for feature calculation. Status/error fields are recorded in the comparison table, and annotation export for that failed variant is skipped.
- ICA runs through MNE and FastICA. If no ECG component is identified, the signal remains unchanged. ICA processing exceptions are handled as failed runs by the batch runner rather than by an internal fallback.

The ASR path is experimental and is marked in `ade/signal_processing.py` as affected by an internal `asrpy.fit_eeg_distribution` error in the tested environment.

## Known Limitations

- The pipeline is currently designed around exactly two EEG channels.
- Input column names are currently hard-coded for the expected Intellivue export format.
- The target sampling rate is fixed at 125 Hz.
- Several detection thresholds are derived independently for every recording, which affects direct threshold comparability between recordings.
- Thresholds are also recalculated for the selected method's output. Interpret before/after marker counts together with the exported threshold values; fewer markers alone do not establish denoising quality.
- Artifact detection is heuristic and has not been established as clinical ground truth.
- ICA and AJDC are strongly constrained by the availability of only two EEG channels.
- ASR support is experimental.
- Console-title updates use Windows-specific functionality; notification sounds are guarded by `winsound` availability.
- Output names do not encode every experiment setting, so repeated names can overwrite earlier results.
- Window creation drops incomplete trailing segments and currently also omits a final window whose end would exactly equal the signal length.
- This repository revision exports features and heuristic labels but does not include model-training or model-persistence scripts.

## Data Protection

EEG and associated physiological recordings may contain sensitive or restricted research data.

Do not commit raw recordings, derived patient-identifiable data, or protected annotations to a public repository. Consider adding the following entries to `.gitignore`:

```gitignore
Data/
results/
*.csv
*.csv.gz
*.xlsx
__pycache__/
*.py[cod]
.venv/
```

Adjust these exclusions if selected anonymized examples are intentionally included.

## Research Use

This software is intended for research, development, and educational use. It is not a medical device and must not be used as the sole basis for clinical decisions.
