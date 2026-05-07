from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def condition_id(info):
    return (
        f"{info['dataset']}_{info['subject']}_"
        f"{int(info['frequency_hz'])}Hz_{int(info['level_db_pespl'])}dB"
    )


def get_unit_for_role(info, role):
    ch = info["record_metas"][0]["channels"].get(role)
    if ch is None:
        return ""
    return info["record_metas"][0]["units"][ch]


def save_npz_for_condition(result, out_path):
    averages = result["averages"]
    info = result["info"]
    arrays = {"time_ms": np.arange(info["n_epoch_samples"]) / info["fs"] * 1000.0}

    for split in ["A", "B", "all"]:
        for role in ["abr", "oae"]:
            if role in averages.get(split, {}):
                arrays[f"{role}_{split}"] = averages[split][role]

    np.savez_compressed(out_path, **arrays)


def metadata_row_from_result(result, npz_path):
    info = result["info"]
    row = {
        "dataset": info["dataset"],
        "subject": info["subject"],
        "frequency_hz": info["frequency_hz"],
        "level_db_pespl": info["level_db_pespl"],
        "fs": info["fs"],
        "epoch_ms": info["epoch_ms"],
        "n_epoch_samples": info["n_epoch_samples"],
        "n_records": info["n_records"],
        "records": json.dumps(info["records"]),
        "average_mode": info["average_mode"],
        "split_method": info["split_method"],
        "block_size_trials": info.get("block_size_trials"),
        "npz_path": str(npz_path),
    }

    for role in ["abr", "oae"]:
        for split in ["A", "B", "all"]:
            avg_info = info["average_info"].get(split, {}).get(role)
            row[f"{role}_{split}_n_trials"] = 0 if avg_info is None else avg_info.get("n_trials", 0)
            row[f"{role}_{split}_n_blocks"] = 0 if avg_info is None else avg_info.get("n_blocks", 0)

    row["record_metas_json"] = json.dumps(info["record_metas"], default=str)
    return row


def plot_condition_average(result, out_path=None, show=False):
    averages = result["averages"]
    info = result["info"]
    roles = [role for role in ["abr", "oae"] if role in averages.get("all", {})]

    if not roles:
        return None

    t_ms = np.arange(info["n_epoch_samples"]) / info["fs"] * 1000.0
    fig, axes = plt.subplots(
        len(roles),
        1,
        figsize=(12, 3.4 * len(roles)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, role in zip(axes, roles):
        unit = get_unit_for_role(info, role)
        ax.plot(t_ms, averages["all"][role], linewidth=1.6)
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.set_ylabel(f"{role.upper()} [{unit}]")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [ms]")
    fig.suptitle(
        f"{info['dataset']} {info['subject']} | {int(info['frequency_hz'])} Hz | "
        f"{int(info['level_db_pespl'])} dB",
        y=1.01,
    )
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def save_preprocessed_results(results, out_root, dataset_name):
    out_root = Path(out_root)
    avg_dir = out_root / dataset_name / "averages"
    plot_dir = out_root / dataset_name / "average_plots"
    avg_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, result in sorted(results.items()):
        info = result["info"]
        cid = safe_name(condition_id(info))
        npz_path = avg_dir / f"{cid}.npz"
        plot_path = plot_dir / f"{cid}.png"

        save_npz_for_condition(result, npz_path)
        fig = plot_condition_average(result, plot_path, show=False)
        if fig is not None:
            plt.close(fig)
        rows.append(metadata_row_from_result(result, npz_path))

    metadata = pd.DataFrame(rows)
    metadata_path = out_root / dataset_name / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print(f"Saved {len(rows)} conditions to {out_root / dataset_name}")
    print(f"Metadata: {metadata_path}")
    return metadata
