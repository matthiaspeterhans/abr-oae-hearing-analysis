from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import find_peaks

from .utils import ms_to_samples


def get_earndb_events_by_channel(record_path, target_channels):
    record_path = Path(record_path)
    events_by_ch = {ch: [] for ch in target_channels}
    trg_path = record_path.with_suffix(".trg")
    trig_path = record_path.with_suffix(".trig")

    if trg_path.exists():
        ann = wfdb.rdann(str(record_path), "trg")
        for sample, symbol, chan in zip(ann.sample, ann.symbol, ann.chan):
            chan = int(chan)
            if symbol == "T" and chan in events_by_ch:
                events_by_ch[chan].append(int(sample) - 1)
        return events_by_ch

    if trig_path.exists():
        with open(trig_path, "r", encoding="ascii") as file:
            for line in file:
                parts = line.split()
                if len(parts) < 5 or parts[2] != "T":
                    continue

                sample = int(parts[1])
                chan = int(parts[4])
                if chan in events_by_ch:
                    events_by_ch[chan].append(sample - 1)
        return events_by_ch

    raise FileNotFoundError(
        f"No EARNDB trigger annotation found for {record_path} (.trg or .trig)"
    )


def find_earh_trigger_peaks(trigger_signal, fs, cfg):
    x = np.asarray(trigger_signal, dtype=float)
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        raise ValueError("Trigger signal contains no finite values")

    center = float(np.median(finite))
    high, low = np.percentile(finite, [99.9, 0.1])

    if high - center >= center - low:
        y = x - center
        polarity = 1
        amp = high - center
    else:
        y = center - x
        polarity = -1
        amp = center - low

    threshold = max(cfg.earh_trigger_threshold_fraction * amp, cfg.eps)
    min_dist = ms_to_samples(cfg.earh_trigger_min_distance_ms, fs)
    peaks, _ = find_peaks(
        y,
        height=threshold,
        prominence=0.25 * threshold,
        distance=max(1, min_dist),
    )

    info = {
        "trigger_polarity": polarity,
        "trigger_threshold": float(threshold),
        "n_trigger_peaks": int(len(peaks)),
        "median_trigger_distance_ms": (
            float(np.median(np.diff(peaks)) / fs * 1000.0)
            if len(peaks) > 1
            else np.nan
        ),
    }
    return peaks.astype(int), info


def get_earh_events_by_channel(record, signal, channels, target_channels, cfg):
    trigger_ch = channels.get("trigger")
    if trigger_ch is None:
        raise ValueError("EARH record has no trigger channel")

    peaks, info = find_earh_trigger_peaks(signal[:, trigger_ch], record.fs, cfg)
    return {ch: list(peaks) for ch in target_channels}, info
