# Window creation and window-index helpers

import numpy as np
import pandas as pd


# Windowing
def create_windows(signal, fs, window_sec=5, overlap=0.5):
    window_size = int(window_sec * fs)
    step = int(window_size * (1 - overlap))

    windows = []

    for start in range(0, len(signal) - window_size, step):
        end = start + window_size
        windows.append(signal[start:end])

    return windows


def create_window_indices(t0, number_of_windows, fs, window_sec, window_overlap):
    # Create sample positions and absolute timestamps using create_windows logic
    # The starting position need to use the same logic as create_windows()
    window_size_samples = int(window_sec * fs)
    window_step_samples = int(
        window_size_samples * (1 - window_overlap)
    )

    window_numbers = np.arange(
        number_of_windows,
        dtype=int,
    )

    start_samples = window_numbers * window_step_samples
    end_samples = start_samples + window_size_samples

    # create Absolute Time Stamps from time starting point of the recording
    begin_idx = (
        pd.Timestamp(t0)
        + pd.to_timedelta(start_samples / fs, unit="s")
    )

    end_idx = (
        pd.Timestamp(t0)
        + pd.to_timedelta(end_samples / fs, unit="s")
    )

    return {
        "window_size_samples": window_size_samples,
        "window_step_samples": window_step_samples,
        "window_numbers": window_numbers,
        "start_samples": start_samples,
        "end_samples": end_samples,
        "begin_idx": begin_idx,
        "end_idx": end_idx,
    }
