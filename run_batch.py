# Executable batch entry point for the ADE pipeline

import os

try:
    import winsound
except ImportError:  # Allows non-Windows import and syntax checks.
    winsound = None

from ade.config import (
    artifact_export_root,
    base_path,
    build_artifact_variant,
    common_params,
    file_path_sheet,
    parameter_sets,
)
from ade.data_io import discover_datasets_to_run
from pipeline import run_pipeline


# Dataset, Data file, Data file ending
datasets_to_run = [
    ("DOSE-I", "10-006", ".csv"),
]


def main():
    # Resolve recordings and execute all configured parameter sets
    resolved_datasets_to_run = list(datasets_to_run)

    if not resolved_datasets_to_run:
        dataset_name = "DOSE-I" # Folder = Name of Dataset

        # Exclusion list by filename
        exclude_files = {
            # "10-123",
        }

        resolved_datasets_to_run = discover_datasets_to_run(
            base_path=base_path,
            dataset_name=dataset_name,
            exclude_files=exclude_files,
        )

    failed_runs = []                                        # Failsafe, if there are failed runs
    total_runs = len(resolved_datasets_to_run) * len(parameter_sets) # Progress Display: Total Runs
    current_run = 0                                         # Progress Display: Starting Parameters

    file_path_sheet.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for dataset, filename, file_ending in resolved_datasets_to_run:
        for method_params in parameter_sets:
            current_run += 1

            # Progress Display
            params_run = common_params.copy()
            params_run.update(method_params)

            artifact_variant = build_artifact_variant(params_run)

            params_run["artifact_export_dir"] = (
                artifact_export_root
                / artifact_variant
            )

            print("Artifact export directory:", params_run["artifact_export_dir"].resolve())

            method_name = params_run.get("signal_method", "unknown")
            if method_name == "wavelet":
                method_name += (
                    f"_{params_run.get('wavelet_name')}"
                    f"_s{params_run.get('wavelet_threshold_scale')}"
                )

            os.system(f"title EEG Batch [{current_run}/{total_runs}] {filename} {method_name}")

            print("=" * 80)
            print(f"[{current_run}/{total_runs}] Starting: {dataset} / {filename} / {method_name}")
            print("=" * 80)

            # Start Pipeline
            try:
                run_pipeline(
                    dataset=dataset,
                    filename=filename,
                    file_ending=file_ending,
                    params=params_run,
                    base_path=base_path,
                    file_path_sheet=file_path_sheet,
                )
            except Exception as e:
                failed_runs.append({
                    "dataset": dataset,
                    "filename": filename,
                    "file_ending": file_ending,
                    "params": params_run,
                    "error": str(e),
                })

                print("[ERRORS]", dataset, filename, params_run)
                print(e)

    # ---

    print("Batch completed.")
    os.system("title EEG Batch completed")

    if failed_runs:
        print("Failed Runs:")
        for run in failed_runs:
            print(run)
        if winsound is not None:
            winsound.MessageBeep(winsound.MB_ICONHAND)      # Error SFX
    else:
        print("No errors.")
        if winsound is not None:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)  # Done SFX


if __name__ == "__main__":
    main()
