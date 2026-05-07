from collections import defaultdict
from pathlib import Path

import numpy as np

from .epochs import extract_epochs_for_record
from .records import condition_key, parse_condition


def split_trials(trials):
    n_trials = len(trials)
    midpoint = n_trials // 2
    return {"A": trials[:midpoint], "B": trials[midpoint:], "all": trials}


def make_trial_blocks(trials, record_name, split_name, block_size_trials=None):
    trials = np.asarray(trials, dtype=float)
    n_trials = len(trials)

    if n_trials == 0:
        return []

    block_size = n_trials if block_size_trials is None else int(block_size_trials)
    if block_size <= 0:
        raise ValueError("block_size_trials must be None or a positive integer")

    blocks = []
    for block_idx, start in enumerate(range(0, n_trials, block_size)):
        stop = min(start + block_size, n_trials)
        blocks.append({
            "record": record_name,
            "block_id": f"{record_name}:{split_name}:b{block_idx:03d}",
            "trial_start": int(start),
            "trial_stop": int(stop),
            "trials": trials[start:stop],
        })
    return blocks


def block_mean_and_noise(trials):
    trials = np.asarray(trials, dtype=float)

    if trials.ndim != 2 or trials.shape[0] == 0:
        return None, np.nan

    trials = trials[np.mean(np.isfinite(trials), axis=1) >= 0.95]
    if trials.shape[0] == 0:
        return None, np.nan

    avg = np.nanmean(trials, axis=0)
    if not np.isfinite(avg).any():
        return None, np.nan

    avg = avg - np.nanmean(avg)
    if trials.shape[0] > 1:
        residual = trials - avg[None, :]
        noise = float(np.nanmean(np.nanvar(residual, axis=0, ddof=1)))
    else:
        noise = np.nan

    return avg, noise


def combine_blocks(block_items, average_mode="block_weighted", eps=1e-12):
    block_avgs = []
    block_noise = []
    block_counts = []
    block_records = []
    block_ids = []
    block_trial_ranges = []

    for item in block_items:
        avg, noise = block_mean_and_noise(item["trials"])
        if avg is None:
            continue

        block_avgs.append(avg)
        block_noise.append(noise)
        block_counts.append(len(item["trials"]))
        block_records.append(item["record"])
        block_ids.append(item.get("block_id", item["record"]))
        block_trial_ranges.append([
            int(item.get("trial_start", 0)),
            int(item.get("trial_stop", len(item["trials"]))),
        ])

    if len(block_avgs) == 0:
        return None, {
            "n_trials": 0,
            "n_blocks": 0,
            "block_records": [],
            "block_ids": [],
            "block_trial_ranges": [],
            "block_counts": [],
            "block_noise": [],
            "block_weights": [],
            "warning": "no_valid_blocks",
        }

    block_avgs = np.asarray(block_avgs, dtype=float)
    block_noise = np.asarray(block_noise, dtype=float)
    block_counts = np.asarray(block_counts, dtype=float)

    if average_mode == "mean":
        weights = block_counts
    elif average_mode == "block_weighted":
        finite_noise = np.isfinite(block_noise) & (block_noise > 0)
        if finite_noise.any():
            fallback_noise = np.nanmedian(block_noise[finite_noise])
            weights = 1.0 / (np.where(finite_noise, block_noise, fallback_noise) + eps)
        else:
            weights = block_counts
    else:
        raise ValueError("average_mode must be 'mean' or 'block_weighted'")

    if (not np.isfinite(weights).all()) or np.sum(weights) <= 0:
        weights = np.ones(len(block_avgs))

    weights = weights / np.sum(weights)
    avg = np.sum(block_avgs * weights[:, None], axis=0)
    avg = avg - np.nanmean(avg)

    info = {
        "n_trials": int(np.sum(block_counts)),
        "n_blocks": int(len(block_avgs)),
        "block_records": block_records,
        "block_ids": block_ids,
        "block_trial_ranges": block_trial_ranges,
        "block_counts": block_counts.astype(int).tolist(),
        "block_noise": block_noise.tolist(),
        "block_weights": weights.tolist(),
    }
    return avg, info


def average_condition(record_paths, dataset, cfg):
    record_paths = sorted(
        [Path(path) for path in record_paths],
        key=lambda path: parse_condition(path, dataset)["rep"],
    )
    blocks_by_role_split = defaultdict(lambda: defaultdict(list))
    metas = []

    for record_path in record_paths:
        trials_by_role, meta = extract_epochs_for_record(record_path, dataset, cfg)
        metas.append(meta)

        for role, trials in trials_by_role.items():
            if len(trials) == 0:
                continue

            split_dict = split_trials(trials) if cfg.split_within_each_record else {"all": trials}
            for split_name, split_trials_arr in split_dict.items():
                blocks_by_role_split[role][split_name].extend(
                    make_trial_blocks(
                        split_trials_arr,
                        record_name=record_path.name,
                        split_name=split_name,
                        block_size_trials=cfg.block_size_trials,
                    )
                )

    averages = defaultdict(dict)
    average_info = defaultdict(dict)

    for role, split_dict in blocks_by_role_split.items():
        for split_name, block_items in split_dict.items():
            avg, avg_info = combine_blocks(
                block_items,
                average_mode=cfg.average_mode,
                eps=cfg.eps,
            )
            if avg is not None:
                averages[split_name][role] = avg
            average_info[split_name][role] = avg_info

    first = metas[0]
    info = {
        "dataset": first["dataset"],
        "subject": first["subject"],
        "frequency_hz": first["frequency_hz"],
        "level_db_pespl": first["level_db_pespl"],
        "records": [meta["record"] for meta in metas],
        "n_records": len(metas),
        "fs": first["fs"],
        "epoch_ms": first["epoch_ms"],
        "n_epoch_samples": first["n_epoch_samples"],
        "average_mode": cfg.average_mode,
        "split_method": "within_each_record" if cfg.split_within_each_record else "all_only",
        "block_size_trials": cfg.block_size_trials,
        "record_metas": metas,
        "average_info": {split: dict(role_info) for split, role_info in average_info.items()},
    }
    return {split: dict(role_avg) for split, role_avg in averages.items()}, info


def preprocess_dataset(record_paths, dataset, cfg):
    grouped = defaultdict(list)
    for path in record_paths:
        grouped[condition_key(parse_condition(path, dataset))].append(Path(path))

    results = {}
    for key, paths in sorted(grouped.items()):
        averages, info = average_condition(paths, dataset, cfg)
        results[key] = {"averages": averages, "info": info}

    return results
