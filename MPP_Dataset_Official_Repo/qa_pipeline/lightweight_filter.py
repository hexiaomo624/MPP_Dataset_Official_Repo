"""Conservative lightweight post-filter for PNG CT slices.

This is a supplemental, conservative cleanup utility. It is not the direct
implementation of the manuscript Otsu statement. The script removes only:
  - the first/last N edge slices in each series
  - slices with dark lung-candidate area below a relaxed threshold

By default this script is a dry run and writes CSV/JSON audit files. Add
``--execute --archive-root`` to quarantine files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_GROUPS = ("HH", "Normal_MPP", "Severe_SMPP", "Viral")


@dataclass(frozen=True)
class SliceDecision:
    group: str
    series_id: str
    path: str
    remove: bool
    reason: str
    dark_area_ratio: float
    move_status: str = "not_requested"
    destination: str = ""


def read_image_cv2(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def dark_area_ratio(path: Path, threshold: int) -> float | None:
    img = read_image_cv2(path)
    if img is None:
        return None
    total_pixels = img.shape[0] * img.shape[1]
    if total_pixels <= 0:
        return None
    return float(np.sum(img < threshold) / total_pixels)


def analyze_series(task: tuple[str, str, str, int, int, float]) -> list[SliceDecision]:
    group, series_id, series_dir_text, trim_slices, dark_threshold, min_dark_area_ratio = task
    series_dir = Path(series_dir_text)
    pngs = sorted(p for p in series_dir.glob("*.png"))
    decisions: list[SliceDecision] = []
    trim_set: set[Path] = set()

    if trim_slices > 0 and len(pngs) > trim_slices * 2:
        trim_set.update(pngs[:trim_slices])
        trim_set.update(pngs[-trim_slices:])

    for path in pngs:
        if path in trim_set:
            decisions.append(SliceDecision(group, series_id, str(path), True, "edge_trim", 0.0))
            continue

        ratio = dark_area_ratio(path, dark_threshold)
        if ratio is None:
            decisions.append(SliceDecision(group, series_id, str(path), True, "unreadable", 0.0))
        elif ratio < min_dark_area_ratio:
            decisions.append(SliceDecision(group, series_id, str(path), True, "dark_area_below_threshold", ratio))
        else:
            decisions.append(SliceDecision(group, series_id, str(path), False, "valid", ratio))

    return decisions


def discover_series(data_root: Path, groups: list[str]) -> list[tuple[str, str, Path]]:
    tasks = []
    for group in groups:
        group_dir = data_root / group
        if not group_dir.is_dir():
            continue
        for series_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            tasks.append((group, series_dir.name, series_dir))
    return tasks


def safe_relative_path(src: Path, data_root: Path) -> Path:
    try:
        return src.resolve().relative_to(data_root.resolve())
    except Exception:
        return Path(src.name)


def quarantine(decisions: list[SliceDecision], data_root: Path, archive_root: Path) -> list[SliceDecision]:
    archive_root.mkdir(parents=True, exist_ok=True)
    updated: list[SliceDecision] = []
    for d in decisions:
        if not d.remove:
            updated.append(d)
            continue
        src = Path(d.path)
        dst = archive_root / safe_relative_path(src, data_root)
        status = "not_found"
        if src.is_file():
            if dst.exists():
                status = "destination_exists"
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                status = "moved"
        updated.append(SliceDecision(d.group, d.series_id, d.path, d.remove, d.reason, d.dark_area_ratio, status, str(dst)))
    return updated


def write_csv(decisions: list[SliceDecision], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SliceDecision.__dataclass_fields__.keys())
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in decisions:
            writer.writerow(asdict(d))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Conservative lightweight PNG slice filter.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    p.add_argument("--out-csv", default="lightweight_filter_audit.csv")
    p.add_argument("--out-json", default="")
    p.add_argument("--trim-slices", type=int, default=3)
    p.add_argument("--dark-threshold", type=int, default=40)
    p.add_argument("--min-dark-area-ratio", type=float, default=0.015)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--execute", action="store_true", help="Move flagged slices to --archive-root.")
    p.add_argument("--archive-root", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if args.execute and not args.archive_root:
        raise ValueError("--archive-root is required with --execute")

    series = discover_series(data_root, args.groups)
    worker_tasks = [(g, sid, str(p), args.trim_slices, args.dark_threshold, args.min_dark_area_ratio) for g, sid, p in series]
    decisions: list[SliceDecision] = []
    if worker_tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for part in executor.map(analyze_series, worker_tasks):
                decisions.extend(part)

    if args.execute:
        decisions = quarantine(decisions, data_root, Path(args.archive_root))

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    write_csv(decisions, out_csv)
    summary: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "data_root": str(data_root),
        "series_scanned": len(series),
        "slices_scanned": len(decisions),
        "flagged_slices": sum(1 for d in decisions if d.remove),
        "moved_slices": sum(1 for d in decisions if d.move_status == "moved"),
        "trim_slices": args.trim_slices,
        "dark_threshold": args.dark_threshold,
        "min_dark_area_ratio": args.min_dark_area_ratio,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Dry run only. No files were moved.")


if __name__ == "__main__":
    main()
