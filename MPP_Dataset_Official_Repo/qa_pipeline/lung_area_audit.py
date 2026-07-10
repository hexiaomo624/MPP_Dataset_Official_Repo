"""Audit and optionally quarantine non-lung PNG slices by dark-area CC rules.

This script is the formalized version of the historical
``04_Tools_And_Utilities/qa_tools/lung_area_audit.py`` utility. It evaluates
8-bit lung-window PNG slices with the rules used by that utility:

  - dark lung candidate mask: pixel value < 40
  - total dark-area ratio must be >= 8%
  - at least one connected component must occupy > 1% of the image

The method is a fixed-threshold dark-area and connected-component audit. It is
not an Otsu-thresholding implementation. By default the script is a dry run and
only writes audit reports. Use ``--execute --archive-root`` to move invalid PNG
files into a quarantine folder while preserving group/series structure.

Example dry run:
  python qa_pipeline/lung_area_audit.py \
    --data-root /path/to/png_dataset \
    --out-csv reports/lung_area_audit.csv

Example execute mode:
  python qa_pipeline/lung_area_audit.py \
    --data-root /path/to/png_dataset \
    --archive-root /path/to/quarantine/lung_area \
    --execute
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
class SliceAudit:
    group: str
    series_id: str
    path: str
    is_invalid: bool
    reason: str
    dark_area_ratio: float
    largest_component_ratio: float
    valid_component_count: int
    move_status: str = "not_requested"
    destination: str = ""


def read_image_cv2(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def analyze_slice(
    path: Path,
    *,
    dark_threshold: int,
    min_dark_area_ratio: float,
    min_component_ratio: float,
) -> tuple[bool, str, float, float, int]:
    img = read_image_cv2(path)
    if img is None:
        return True, "unreadable", 0.0, 0.0, 0

    total_pixels = int(img.shape[0] * img.shape[1])
    if total_pixels <= 0:
        return True, "empty_image", 0.0, 0.0, 0

    binary = (img < dark_threshold).astype(np.uint8)
    dark_pixels = int(np.sum(binary))
    dark_area_ratio = dark_pixels / total_pixels

    if dark_area_ratio < min_dark_area_ratio:
        return True, "dark_area_below_threshold", dark_area_ratio, 0.0, 0

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest_component_ratio = 0.0
    valid_component_count = 0

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        component_ratio = area / total_pixels
        largest_component_ratio = max(largest_component_ratio, component_ratio)
        if component_ratio > min_component_ratio:
            valid_component_count += 1

    if valid_component_count == 0:
        return True, "no_large_connected_component", dark_area_ratio, largest_component_ratio, 0

    return False, "valid", dark_area_ratio, largest_component_ratio, valid_component_count


def discover_series(data_root: Path, groups: list[str]) -> list[tuple[str, str, Path]]:
    tasks: list[tuple[str, str, Path]] = []
    for group in groups:
        group_dir = data_root / group
        if not group_dir.is_dir():
            continue
        for series_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            tasks.append((group, series_dir.name, series_dir))
    return tasks


def analyze_series(task: tuple[str, str, str, int, float, float]) -> list[SliceAudit]:
    group, series_id, series_dir_text, dark_threshold, min_dark_area_ratio, min_component_ratio = task
    series_dir = Path(series_dir_text)
    audits: list[SliceAudit] = []

    for path in sorted(series_dir.glob("*.png")):
        is_invalid, reason, dark_ratio, largest_ratio, valid_count = analyze_slice(
            path,
            dark_threshold=dark_threshold,
            min_dark_area_ratio=min_dark_area_ratio,
            min_component_ratio=min_component_ratio,
        )
        audits.append(
            SliceAudit(
                group=group,
                series_id=series_id,
                path=str(path),
                is_invalid=is_invalid,
                reason=reason,
                dark_area_ratio=dark_ratio,
                largest_component_ratio=largest_ratio,
                valid_component_count=valid_count,
            )
        )

    return audits


def safe_relative_path(src: Path, data_root: Path) -> Path:
    try:
        return src.resolve().relative_to(data_root.resolve())
    except Exception:
        return Path(src.name)


def quarantine_invalid_slices(
    audits: list[SliceAudit],
    *,
    data_root: Path,
    archive_root: Path,
) -> list[SliceAudit]:
    archive_root.mkdir(parents=True, exist_ok=True)
    updated: list[SliceAudit] = []

    for audit in audits:
        if not audit.is_invalid:
            updated.append(audit)
            continue

        src = Path(audit.path)
        rel = safe_relative_path(src, data_root)
        dst = archive_root / rel
        move_status = "not_found"
        destination = str(dst)

        if src.is_file():
            if dst.exists():
                move_status = "destination_exists"
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                move_status = "moved"

        updated.append(
            SliceAudit(
                group=audit.group,
                series_id=audit.series_id,
                path=audit.path,
                is_invalid=audit.is_invalid,
                reason=audit.reason,
                dark_area_ratio=audit.dark_area_ratio,
                largest_component_ratio=audit.largest_component_ratio,
                valid_component_count=audit.valid_component_count,
                move_status=move_status,
                destination=destination,
            )
        )

    return updated


def write_csv(audits: list[SliceAudit], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(audits[0]).keys()) if audits else list(SliceAudit.__dataclass_fields__.keys())
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            writer.writerow(asdict(audit))


def write_summary(summary: dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit lung-window PNG slices using dark-area and connected-component rules."
    )
    parser.add_argument("--data-root", required=True, help="Root containing class/group folders of PNG series.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Group folders to scan under --data-root.",
    )
    parser.add_argument("--out-csv", default="lung_area_audit.csv", help="Per-slice audit CSV output.")
    parser.add_argument("--out-json", default="", help="Optional summary JSON output.")
    parser.add_argument("--dark-threshold", type=int, default=40, help="Pixel threshold for dark lung candidate mask.")
    parser.add_argument("--min-dark-area-ratio", type=float, default=0.08, help="Minimum dark area ratio per slice.")
    parser.add_argument(
        "--min-component-ratio",
        type=float,
        default=0.01,
        help="Minimum ratio for at least one connected component.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Move invalid PNGs to --archive-root. Omit for dry-run audit only.",
    )
    parser.add_argument("--archive-root", default="", help="Quarantine root required when --execute is set.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if args.execute and not args.archive_root:
        raise ValueError("--archive-root is required when --execute is set")

    series_tasks = discover_series(data_root, args.groups)
    worker_tasks = [
        (
            group,
            series_id,
            str(series_dir),
            args.dark_threshold,
            args.min_dark_area_ratio,
            args.min_component_ratio,
        )
        for group, series_id, series_dir in series_tasks
    ]

    audits: list[SliceAudit] = []
    if worker_tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for series_audits in executor.map(analyze_series, worker_tasks):
                audits.extend(series_audits)

    if args.execute:
        audits = quarantine_invalid_slices(audits, data_root=data_root, archive_root=Path(args.archive_root))

    invalid_count = sum(1 for audit in audits if audit.is_invalid)
    moved_count = sum(1 for audit in audits if audit.move_status == "moved")
    summary = {
        "mode": "execute" if args.execute else "dry_run",
        "data_root": str(data_root),
        "groups": args.groups,
        "series_scanned": len(series_tasks),
        "slices_scanned": len(audits),
        "invalid_slices": invalid_count,
        "moved_slices": moved_count,
        "dark_threshold": args.dark_threshold,
        "min_dark_area_ratio": args.min_dark_area_ratio,
        "min_component_ratio": args.min_component_ratio,
        "method": "fixed dark-area threshold plus connected-component analysis",
    }

    write_csv(audits, out_csv)
    write_summary(summary, out_json)

    print(f"Mode: {summary['mode']}")
    print(f"Series scanned: {len(series_tasks)}")
    print(f"Slices scanned: {len(audits)}")
    print(f"Invalid slices: {invalid_count}")
    if args.execute:
        print(f"Moved slices: {moved_count}")
    else:
        print("Dry run only. No files were moved. Add --execute and --archive-root to quarantine slices.")
    print(f"Audit CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")


if __name__ == "__main__":
    main()
