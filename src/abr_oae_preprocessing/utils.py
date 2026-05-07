import numpy as np
from scipy.signal import butter, sosfiltfilt


def normalize_unit(unit):
    return (unit or "").lower().replace("\N{MICRO SIGN}", "u").strip()


def ms_to_samples(ms, fs):
    return int(round(ms * fs / 1000.0))


def unit_to_uv_factor(unit):
    unit = normalize_unit(unit)

    if unit in ["v", "volt", "volts"]:
        return 1e6
    if unit in ["mv", "millivolt", "millivolts"]:
        return 1e3
    if unit in ["uv", "microvolt", "microvolts"]:
        return 1.0
    if unit in ["nv", "nanovolt", "nanovolts"]:
        return 1e-3

    raise ValueError(f"Unknown unit: {unit}")


def fill_nan_1d(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)

    if finite.all() or not finite.any():
        return x

    idx = np.arange(len(x))
    y = x.copy()
    y[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return y


def bandpass_1d(x, fs, low, high, order=4):
    x = fill_nan_1d(x)

    if not np.isfinite(x).all():
        return x

    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def is_valid_epoch(epoch, min_finite_fraction=0.95):
    epoch = np.asarray(epoch, dtype=float)
    return (
        epoch.ndim == 1
        and epoch.size > 0
        and np.mean(np.isfinite(epoch)) >= min_finite_fraction
    )
