from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb


def parse_earndb_average_name(path):
    match = re.match(
        r"(?P<subject>N\d+)_evoked_ave(?P<level>\d+)_F(?P<freq>\d+)_R(?P<rep>\d+)$",
        Path(path).stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Cannot parse EARNDB average name: {Path(path).name}")

    return {
        "subject": match.group("subject").upper(),
        "level_db_pespl": int(match.group("level")),
        "frequency_hz": int(match.group("freq")) * 1000,
        "rep": int(match.group("rep")),
    }


def normalized_trace(y):
    y = np.asarray(y, dtype=float)
    y = y - np.nanmean(y)
    norm = float(np.nanmax(np.abs(y)))
    return (y / norm, norm) if np.isfinite(norm) and norm > 0 else (y, norm)


def collect_earndb_average_paths(root, subject, levels=None, frequencies_hz=None):
    root = Path(root)
    levels = None if levels is None else {int(level) for level in levels}
    frequencies_hz = None if frequencies_hz is None else {int(freq) for freq in frequencies_hz}
    paths = []

    for hea in sorted(root.glob(f"{subject}_evoked_ave*_F*_R*.hea")):
        meta = parse_earndb_average_name(hea)
        if levels is not None and meta["level_db_pespl"] not in levels:
            continue
        if frequencies_hz is not None and meta["frequency_hz"] not in frequencies_hz:
            continue
        paths.append(hea)

    return sorted(paths, key=lambda path: (
        parse_earndb_average_name(path)["level_db_pespl"],
        parse_earndb_average_name(path)["frequency_hz"],
        parse_earndb_average_name(path)["rep"],
    ))


def preprocessed_average_path(preprocessed_root, meta):
    return (
        Path(preprocessed_root)
        / f"earndb_{meta['subject']}_{meta['frequency_hz']}Hz_{meta['level_db_pespl']}dB.npz"
    )


def compare_published_earndb_averages(
    average_root,
    preprocessed_average_root,
    out_root,
    subject="N1",
    levels=(100, 90, 80, 70, 60, 50, 40, 30, 20, 10),
    frequencies_hz=None,
):
    average_root = Path(average_root)
    preprocessed_average_root = Path(preprocessed_average_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not average_root.exists():
        print(f"Skip published-average comparison: missing {average_root}")
        return pd.DataFrame()

    average_paths = collect_earndb_average_paths(
        average_root,
        subject,
        levels=levels,
        frequencies_hz=frequencies_hz,
    )
    if not average_paths:
        print(f"Skip published-average comparison: no records for {subject}")
        return pd.DataFrame()

    summary_rows = []

    for path in average_paths:
        meta = parse_earndb_average_name(path)
        rec = wfdb.rdrecord(str(path.with_suffix("")), physical=False)

        if rec.d_signal is None:
            raise ValueError(f"No digital signal in {path}")

        published = np.asarray(rec.d_signal, dtype=float)
        published_t_ms = np.arange(published.shape[0]) / float(rec.fs) * 1000.0
        new_path = preprocessed_average_path(preprocessed_average_root, meta)
        new_average = np.load(new_path) if new_path.exists() else None
        new_t_ms = (
            new_average["time_ms"]
            if new_average is not None and "time_ms" in new_average.files
            else None
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 6.0), sharex="col")
        fig.suptitle(
            f"{path.stem} | {meta['frequency_hz']} Hz | "
            f"{meta['level_db_pespl']} dB | R{meta['rep']}",
            y=1.03,
        )

        for ch, role in enumerate(["abr", "oae"]):
            if ch >= published.shape[1]:
                continue

            y_pub, pub_norm = normalized_trace(published[:, ch])
            axes[0, ch].plot(published_t_ms, y_pub, linewidth=1.5)
            axes[0, ch].axhline(0, color="0.75", linewidth=0.8)
            axes[0, ch].set_title(f"published {role.upper()}")
            axes[0, ch].set_ylabel("normalized")
            axes[0, ch].grid(alpha=0.25)

            summary_rows.append({
                "record": path.stem,
                "level_db_pespl": meta["level_db_pespl"],
                "frequency_hz": meta["frequency_hz"],
                "rep": meta["rep"],
                "role": role,
                "source": "published_earndb_average",
                "norm_scale": pub_norm,
                "ptp": float(np.nanmax(published[:, ch]) - np.nanmin(published[:, ch])),
                "rms": float(np.sqrt(np.nanmean(
                    (published[:, ch] - np.nanmean(published[:, ch])) ** 2
                ))),
            })

            new_key = f"{role}_all"
            if new_average is None or new_key not in new_average.files:
                axes[1, ch].text(
                    0.5,
                    0.5,
                    f"missing\n{new_path.name}",
                    ha="center",
                    va="center",
                    transform=axes[1, ch].transAxes,
                )
                axes[1, ch].set_title(f"computed {role.upper()}")
                axes[1, ch].grid(alpha=0.25)
                continue

            new_y = np.asarray(new_average[new_key], dtype=float)
            new_plot, new_norm = normalized_trace(new_y)
            if new_t_ms is None:
                new_t_ms = np.arange(len(new_y)) / float(rec.fs) * 1000.0

            axes[1, ch].plot(new_t_ms, new_plot, linewidth=1.5, color="C1")
            axes[1, ch].axhline(0, color="0.75", linewidth=0.8)
            axes[1, ch].set_title(f"computed {role.upper()}")
            axes[1, ch].set_xlabel("Time [ms]")
            axes[1, ch].set_ylabel("normalized")
            axes[1, ch].grid(alpha=0.25)

            new_centered = new_y - np.nanmean(new_y)
            summary_rows.append({
                "record": new_path.stem,
                "level_db_pespl": meta["level_db_pespl"],
                "frequency_hz": meta["frequency_hz"],
                "rep": meta["rep"],
                "role": role,
                "source": "new_preprocessed_average",
                "norm_scale": new_norm,
                "ptp": float(np.nanmax(new_centered) - np.nanmin(new_centered)),
                "rms": float(np.sqrt(np.nanmean(new_centered ** 2))),
            })

        if new_average is not None:
            new_average.close()

        fig.tight_layout()
        fig.savefig(out_root / f"{path.stem}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values([
            "level_db_pespl",
            "frequency_hz",
            "rep",
            "role",
            "source",
        ])
        summary.to_csv(out_root / f"{subject}_published_average_comparison.csv", index=False)

    print(f"Saved {len(summary_rows)} comparison rows to {out_root}")
    return summary
