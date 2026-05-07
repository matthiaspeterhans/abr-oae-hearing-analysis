from pathlib import Path

import numpy as np

from .artifacts import get_rejected_abr_samples
from .records import detect_channels, parse_condition, read_record_with_gain_override
from .triggers import get_earh_events_by_channel, get_earndb_events_by_channel
from .utils import bandpass_1d, is_valid_epoch, ms_to_samples, normalize_unit


def maybe_filter_signal(signal, record, channels, cfg):
    signal = signal.copy()
    abr_ch = channels.get("abr")

    if abr_ch is not None and cfg.filter_abr:
        signal[:, abr_ch] = bandpass_1d(
            signal[:, abr_ch],
            fs=record.fs,
            low=cfg.abr_low_hz,
            high=cfg.abr_high_hz,
            order=cfg.filter_order,
        )

    return signal


def extract_epochs_for_record(record_path, dataset, cfg):
    record_path = Path(record_path)
    cond = parse_condition(record_path, dataset)
    record, signal, applied_gain_overrides, applied_gain_multipliers = (
        read_record_with_gain_override(record_path, cfg, dataset=dataset)
    )
    channels = detect_channels(record, dataset)
    signal = maybe_filter_signal(signal, record, channels, cfg)

    oae_ch = channels.get("oae")
    if dataset == "earndb" and cfg.correct_oae_gain and oae_ch is not None:
        oae_unit = normalize_unit(record.units[oae_ch])
        oae_gain = float(record.adc_gain[oae_ch])
        if oae_unit == "v" and ((not np.isfinite(oae_gain)) or oae_gain <= 1.0):
            raise ValueError(
                f"Uncorrected EARNDB OAE gain in {record_path}: "
                f"channel={oae_ch}, gain={oae_gain}"
            )

    target_roles = ["abr", "oae"]
    target_channels = [channels[role] for role in target_roles if channels.get(role) is not None]
    if not target_channels:
        raise ValueError(f"No ABR/OAE channels found in {record_path}")

    if dataset == "earndb":
        events_by_ch = get_earndb_events_by_channel(record_path, target_channels)
        trigger_info = {"trigger_source": "wfdb_trg_channel_specific"}
    elif dataset == "earh":
        events_by_ch, trigger_info = get_earh_events_by_channel(
            record,
            signal,
            channels,
            target_channels,
            cfg,
        )
        trigger_info["trigger_source"] = "trigger_channel_peak"
    else:
        raise ValueError("dataset must be 'earndb' or 'earh'")

    n_epoch = ms_to_samples(cfg.epoch_ms, record.fs)
    abr_ch = channels.get("abr")
    abr_reject_samples = set()
    artifact_qc = {}

    if abr_ch is not None and abr_ch in events_by_ch:
        abr_reject_samples, artifact_qc = get_rejected_abr_samples(
            dataset=dataset,
            signal=signal,
            events_by_ch=events_by_ch,
            abr_ch=abr_ch,
            abr_unit=record.units[abr_ch],
            n_epoch=n_epoch,
            cfg=cfg,
        )

    trials_by_role = {}
    raw_counts_by_role = {}
    counts_by_role = {}
    rejected_by_role = {}
    invalid_by_role = {}

    for role in target_roles:
        ch = channels.get(role)
        if ch is None or ch not in events_by_ch:
            continue

        kept = []
        n_invalid = 0
        n_rejected = 0
        raw_counts_by_role[role] = len(events_by_ch[ch])

        for t0 in events_by_ch[ch]:
            start = int(t0)
            stop = start + n_epoch

            if start < 0 or stop > signal.shape[0]:
                n_invalid += 1
                continue

            epoch = signal[start:stop, ch]
            if not is_valid_epoch(epoch):
                n_invalid += 1
                continue

            if role == "abr" and int(t0) in abr_reject_samples:
                n_rejected += 1
                continue

            kept.append(epoch)

        trials_by_role[role] = np.asarray(kept, dtype=float)
        counts_by_role[role] = len(kept)
        rejected_by_role[role] = n_rejected
        invalid_by_role[role] = n_invalid

    meta = {
        **cond,
        "record": record_path.name,
        "fs": float(record.fs),
        "epoch_ms": cfg.epoch_ms,
        "n_epoch_samples": int(n_epoch),
        "sig_name": record.sig_name,
        "units": record.units,
        "adc_gain": record.adc_gain,
        "channels": channels,
        "raw_counts": raw_counts_by_role,
        "counts": counts_by_role,
        "rejected": rejected_by_role,
        "invalid": invalid_by_role,
        "artifact_qc": artifact_qc,
        "applied_gain_overrides": applied_gain_overrides,
        "applied_gain_multipliers": applied_gain_multipliers,
        **trigger_info,
    }
    return trials_by_role, meta
