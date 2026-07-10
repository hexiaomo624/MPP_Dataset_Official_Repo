"""Read-only PNG quality audit.

Flags unreadable, nearly black, or textureless PNG slices. The script does not
delete or move data. Optionally copy a small sample of flagged images to a
review folder.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_GROUPS = ("HH", "Normal_MPP", "Severe_SMPP", "Viral")


def process_image(args: tuple[str, float, float]) -> tuple[str, str, float, float] | None:
    img_path, min_mean, min_std = args
    try:
        with Image.open(img_path) as img:
            arr = np.array(img)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            if mean_val < min_mean:
                return (img_path, "extreme_black", mean_val, std_val)
            if std_val < min_std:
                return (img_path, "no_texture", mean_val, std_val)
            return None
    except Exception as e:
        return (img_path, f"read_error:{e}", 0.0, 0.0)


def chunk_files(files: list[str], chunk_size: int):
    for i in range(0, len(files), chunk_size):
        yield files[i : i + chunk_size]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only PNG quality audit.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    p.add_argument("--out-csv", default="png_quality_audit.csv")
    p.add_argument("--out-json", default="")
    p.add_argument("--sample-dir", default="", help="Optional folder for copies of flagged review samples.")
    p.add_argument("--sample-size", type=int, default=10)
    p.add_argument("--min-mean", type=float, default=2.0)
    p.add_argument("--min-std", type=float, default=1.0)
    p.add_argument("--chunk-size", type=int, default=50000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    all_pngs: list[str] = []
    group_totals = {g: 0 for g in args.groups}
    for group in args.groups:
        group_path = data_root / group
        if group_path.exists():
            pngs = glob.glob(str(group_path / "**" / "*.png"), recursive=True)
            all_pngs.extend(pngs)
            group_totals[group] = len(pngs)

    start = time.time()
    flagged: list[tuple[str, str, float, float]] = []
    worker_args = [(p, args.min_mean, args.min_std) for p in all_pngs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        processed = 0
        for chunk in chunk_files(worker_args, args.chunk_size):
            for res in executor.map(process_image, chunk):
                if res is not None:
                    flagged.append(res)
            processed += len(chunk)
            print(f"Processed {processed}/{len(all_pngs)} PNGs")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "issue", "mean", "std"])
        writer.writerows(flagged)

    flagged_stats = {g: 0 for g in args.groups}
    for path, _issue, _mean, _std in flagged:
        parts = Path(path).parts
        for g in args.groups:
            if g in parts:
                flagged_stats[g] += 1
                break

    copied_samples = 0
    if args.sample_dir and flagged:
        sample_dir = Path(args.sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i, (path, issue, mean, std) in enumerate(random.sample(flagged, min(args.sample_size, len(flagged)))):
            safe_issue = issue.replace(":", "_").replace(os.sep, "_")
            dst = sample_dir / f"{i:03d}_{safe_issue}_m{mean:.1f}_s{std:.1f}_{Path(path).name}"
            shutil.copy2(path, dst)
            copied_samples += 1

    summary = {
        "data_root": str(data_root),
        "groups": args.groups,
        "total_pngs": len(all_pngs),
        "flagged_pngs": len(flagged),
        "flagged_by_group": flagged_stats,
        "group_totals": group_totals,
        "min_mean": args.min_mean,
        "min_std": args.min_std,
        "sample_dir": args.sample_dir,
        "copied_samples": copied_samples,
        "elapsed_seconds": time.time() - start,
    }
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
