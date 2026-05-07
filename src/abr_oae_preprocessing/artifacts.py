import numpy as np

from .utils import is_valid_epoch, unit_to_uv_factor


def compute_epoch_peaks_uv(signal, t0s, ch, n_epoch, unit):
    factor = unit_to_uv_factor(unit)
    valid_t0s = []
    peaks_uv = []

    for t0 in t0s:
        start = int(t0)
        stop = start + n_epoch

        if start < 0 or stop > signal.shape[0]:
            continue

        epoch = signal[start:stop, ch]
        if not is_valid_epoch(epoch):
            continue

        peak = np.nanmax(np.abs((epoch - np.nanmean(epoch)) * factor))
        if np.isfinite(peak):
            valid_t0s.append(int(t0))
            peaks_uv.append(float(peak))

    return np.asarray(valid_t0s, dtype=int), np.asarray(peaks_uv, dtype=float)


def rejection_qc(policy, peaks_uv, rejected, **extra):
    qc = {
        "policy": policy,
        "n_valid_for_rejection": int(len(peaks_uv)),
        "n_rejected": int(len(rejected)),
        "retention": float(1.0 - len(rejected) / max(1, len(peaks_uv))),
        "peak_uv_p50": float(np.percentile(peaks_uv, 50)) if len(peaks_uv) else np.nan,
        "peak_uv_p95": float(np.percentile(peaks_uv, 95)) if len(peaks_uv) else np.nan,
        "peak_uv_p99": float(np.percentile(peaks_uv, 99)) if len(peaks_uv) else np.nan,
    }
    qc.update(extra)
    return qc


def reject_threshold_uv(t0s, peaks_uv, threshold_uv):
    rejected = set(np.asarray(t0s)[peaks_uv > threshold_uv].astype(int))
    qc = rejection_qc(
        "threshold_uv",
        peaks_uv,
        rejected,
        threshold_uv=float(threshold_uv),
    )
    return rejected, qc


def reject_mad_outliers(t0s, peaks_uv, k=3.0):
    if len(peaks_uv) == 0:
        return set(), rejection_qc(
            "mad",
            peaks_uv,
            set(),
            mad_k=float(k),
            cutoff_uv=np.nan,
        )

    median = np.median(peaks_uv)
    mad = np.median(np.abs(peaks_uv - median))

    if mad == 0 or not np.isfinite(mad):
        return set(), rejection_qc(
            "mad",
            peaks_uv,
            set(),
            mad_k=float(k),
            cutoff_uv=np.nan,
        )

    cutoff = median + k * 1.4826 * mad
    rejected = set(np.asarray(t0s)[peaks_uv > cutoff].astype(int))
    qc = rejection_qc(
        "mad",
        peaks_uv,
        rejected,
        mad_k=float(k),
        cutoff_uv=float(cutoff),
    )
    return rejected, qc


def get_rejected_abr_samples(dataset, signal, events_by_ch, abr_ch, abr_unit, n_epoch, cfg):
    policy = cfg.artifact_policy_by_dataset.get(dataset, "threshold_uv")
    t0s = events_by_ch.get(abr_ch, [])
    valid_t0s, peaks_uv = compute_epoch_peaks_uv(signal, t0s, abr_ch, n_epoch, abr_unit)

    if policy == "threshold_uv":
        return reject_threshold_uv(valid_t0s, peaks_uv, cfg.artifact_threshold_uv)
    if policy == "mad":
        return reject_mad_outliers(valid_t0s, peaks_uv, k=cfg.mad_k)

    raise ValueError("artifact policy must be 'threshold_uv' or 'mad'")
