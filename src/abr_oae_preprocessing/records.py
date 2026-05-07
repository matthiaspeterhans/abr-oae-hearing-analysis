from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
import wfdb

from .utils import normalize_unit


def parse_earndb_condition(record_path):
    name = Path(record_path).name
    match = re.search(
        r"(?P<subject>N\d+)_evoked_raw_(?P<level>\d+)_F(?P<freq>\d+)_R(?P<rep>\d+)",
        name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Cannot parse EARNDB name: {name}")

    return {
        "dataset": "earndb",
        "subject": match.group("subject").upper(),
        "frequency_hz": int(match.group("freq")) * 1000,
        "level_db_pespl": int(match.group("level")),
        "rep": int(match.group("rep")),
    }


def parse_earh_condition(record_path):
    name = Path(record_path).name
    match = re.search(
        r"h(?P<subject>\d+)_(?P<ear>[lr])_(?P<freq>\d+)kHz_(?P<level>\d+)_(?P<rep>\d+)",
        name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Cannot parse EARH name: {name}")

    return {
        "dataset": "earh",
        "subject": f"H{match.group('subject')}",
        "ear": match.group("ear").lower(),
        "frequency_hz": int(match.group("freq")) * 1000,
        "level_db_pespl": int(match.group("level")),
        "rep": int(match.group("rep")),
    }


def parse_condition(record_path, dataset):
    if dataset == "earndb":
        return parse_earndb_condition(record_path)
    if dataset == "earh":
        return parse_earh_condition(record_path)
    raise ValueError("dataset must be 'earndb' or 'earh'")


def condition_key(cond):
    return (
        cond["dataset"],
        cond["subject"],
        cond["frequency_hz"],
        cond["level_db_pespl"],
    )


def collect_earndb_paths(root, subject=None, exclude_x=True):
    root = Path(root)
    subjects = [subject] if subject is not None else [f"N{i}" for i in range(1, 9)]
    paths = []

    for sub in subjects:
        subject_dir = root / sub
        if not subject_dir.exists():
            continue

        for hea in sorted(subject_dir.glob(f"{sub}_evoked_raw_*.hea")):
            if exclude_x and "_x" in hea.stem:
                continue
            paths.append(hea.with_suffix(""))

    return paths


def collect_earh_paths(root, subject=None):
    root = Path(root)
    dirs = [root / subject] if subject is not None else sorted(
        path for path in root.glob("h*") if path.is_dir()
    )
    paths = []

    for directory in dirs:
        paths.extend(hea.with_suffix("") for hea in sorted(directory.glob("*.hea")))

    return paths


def detect_channels(record, dataset):
    names = [(name or "").lower() for name in (record.sig_name or [])]
    units = [normalize_unit(unit) for unit in (record.units or [])]

    trigger_ch = None
    abr_ch = None
    oae_ch = None

    for idx, name in enumerate(names):
        if "trigger" in name:
            trigger_ch = idx
        if "abr" in name:
            abr_ch = idx
        if "oae" in name:
            oae_ch = idx

    for idx, unit in enumerate(units):
        if abr_ch is None and unit in ["nv", "mv", "uv"]:
            abr_ch = idx
        if oae_ch is None and unit == "v" and idx != trigger_ch:
            oae_ch = idx

    if dataset == "earh":
        if trigger_ch is None and record.n_sig >= 1:
            trigger_ch = 0
        if abr_ch is None and record.n_sig >= 2:
            abr_ch = 1
        if oae_ch is None and record.n_sig >= 3:
            oae_ch = 2

    return {"trigger": trigger_ch, "abr": abr_ch, "oae": oae_ch}


def build_oae_gain_overrides(record_paths, dataset, min_valid_gain=1.0, rtol=1e-6):
    """Find broken EARNDB OAE gains per condition and correct them in memory only."""
    grouped = defaultdict(list)

    for path in record_paths:
        path = Path(path)
        grouped[condition_key(parse_condition(path, dataset))].append(path)

    overrides = {}
    report = []

    for key, paths in grouped.items():
        rows = []

        for path in paths:
            header = wfdb.rdheader(str(path))
            channels = detect_channels(header, dataset)
            oae_ch = channels.get("oae")

            if oae_ch is None:
                continue

            rows.append({
                "path": path,
                "record": path.name,
                "oae_ch": oae_ch,
                "gain": float(header.adc_gain[oae_ch]),
                "unit": header.units[oae_ch],
            })

        valid_gains = [
            row["gain"]
            for row in rows
            if np.isfinite(row["gain"]) and row["gain"] > min_valid_gain
        ]
        invalid_rows = [
            row
            for row in rows
            if (not np.isfinite(row["gain"])) or row["gain"] <= min_valid_gain
        ]

        if len(valid_gains) == 0 or len(invalid_rows) == 0:
            continue

        consensus = float(np.median(valid_gains))
        if not np.allclose(valid_gains, consensus, rtol=rtol, atol=0):
            for row in invalid_rows:
                report.append({
                    "condition": key,
                    "record": row["record"],
                    "status": "not_corrected_valid_gains_disagree",
                    "old_gain": row["gain"],
                    "valid_gains": valid_gains,
                })
            continue

        for row in invalid_rows:
            overrides.setdefault(str(row["path"].resolve()), {})[int(row["oae_ch"])] = consensus
            report.append({
                "condition": key,
                "record": row["record"],
                "status": "corrected",
                "channel": int(row["oae_ch"]),
                "old_gain": float(row["gain"]),
                "new_gain": consensus,
                "valid_gains": valid_gains,
            })

    return overrides, pd.DataFrame(report)


def read_record_with_gain_override(record_path, cfg, dataset=None):
    """Read digital WFDB values and convert them to physical units with corrected gains."""
    record_path = Path(record_path)
    record = wfdb.rdrecord(str(record_path), physical=False)

    if record.d_signal is None:
        raise ValueError(f"No digital signal available: {record_path}")

    digital = np.asarray(record.d_signal, dtype=float)
    gains = np.asarray(record.adc_gain, dtype=float).copy()
    baselines = (
        np.zeros(record.n_sig, dtype=float)
        if record.baseline is None
        else np.asarray(record.baseline, dtype=float)
    )

    applied_overrides = {}
    applied_multipliers = {}

    override_key = str(record_path.resolve())
    if cfg.correct_oae_gain and override_key in cfg.gain_overrides:
        for ch, new_gain in cfg.gain_overrides[override_key].items():
            ch = int(ch)
            applied_overrides[ch] = {
                "old_gain": float(gains[ch]),
                "new_gain": float(new_gain),
            }
            gains[ch] = float(new_gain)

    role_multipliers = (
        cfg.gain_multipliers_by_dataset_role.get(dataset, {})
        if dataset is not None
        else {}
    )
    if role_multipliers:
        channels = detect_channels(record, dataset)

        for role, multiplier in role_multipliers.items():
            ch = channels.get(role)
            if ch is None:
                continue

            multiplier = float(multiplier)
            if (not np.isfinite(multiplier)) or multiplier <= 0:
                raise ValueError(f"Invalid gain multiplier for {dataset}/{role}: {multiplier}")

            old_gain = float(gains[ch])
            gains[ch] = old_gain * multiplier
            applied_multipliers[ch] = {
                "role": role,
                "multiplier": multiplier,
                "old_gain": old_gain,
                "new_gain": float(gains[ch]),
            }

    signal = np.full_like(digital, np.nan, dtype=float)
    for ch in range(record.n_sig):
        if np.isfinite(gains[ch]) and gains[ch] != 0:
            signal[:, ch] = (digital[:, ch] - baselines[ch]) / gains[ch]

    record.adc_gain = [float(gain) for gain in gains]
    return record, signal, applied_overrides, applied_multipliers
