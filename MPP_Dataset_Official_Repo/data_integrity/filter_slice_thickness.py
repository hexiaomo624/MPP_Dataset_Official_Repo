"""Audit and optionally quarantine CT series with slice thickness > 1.5 mm.

This script supports the manuscript technical-validation statement that series
with reconstructed slice thickness greater than 1.5 mm were excluded. It reads a
mapping/metadata CSV containing a ``SliceThickness`` column and writes an audit
CSV/JSON report. By default it is a dry run and does not move any files.

Expected useful CSV columns:
  - SliceThickness: numeric reconstructed slice thickness in mm
  - Matched_PNG_Path: optional path to the processed PNG series folder
  - SeriesInstanceUID, PatientID, Original_Folder, Series_Type: optional context

Example dry run:
  python data_integrity/filter_slice_thickness.py \
    --mapping-csv metadata/core_tables/mapping_list_final_v5_png_uid_recon.csv \
    --out-csv reports/removed_over_1.5mm.csv

Example execute mode:
  python data_integrity/filter_slice_thickness.py \
    --mapping-csv metadata/core_tables/mapping_list_final_v5_png_uid_recon.csv \
    --data-root /path/to/png_dataset \
    --archive-root /path/to/quarantine/over_1.5mm \
    --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_THICKNESS_MM = 1.5


@dataclass(frozen=True)
class ThicknessRecord:
    row_index: int
    slice_thickness: float
    matched_png_path: str
    action: str
    move_status: str
    destination: str
    context: dict[str, str]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def resolve_series_path(path_text: str, data_root: Path | None) -> Path | None:
    path_text = (path_text or "").strip()
    if not path_text or path_text == "PNG_Already_Removed":
        return None

    p = Path(path_text)
    if p.is_absolute():
        return p
    if data_root is not None:
        return data_root / p
    return p


def safe_relative_path(src: Path, data_root: Path | None) -> Path:
    if data_root is not None:
        try:
            return src.resolve().relative_to(data_root.resolve())
        except Exception:
            pass
    return Path(src.name)


def collect_records(
    mapping_csv: Path,
    *,
    max_thickness_mm: float,
    data_root: Path | None,
) -> tuple[list[ThicknessRecord], dict[str, Any]]:
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    records: list[ThicknessRecord] = []
    total_rows = 0
    missing_thickness = 0
    invalid_thickness = 0

    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Mapping CSV has no header row")
        if "SliceThickness" not in reader.fieldnames:
            raise ValueError("Mapping CSV must contain a SliceThickness column")

        for row_index, row in enumerate(reader, start=2):
            total_rows += 1
            raw_thickness = row.get("SliceThickness")
            if raw_thickness is None or str(raw_thickness).strip() == "":
                missing_thickness += 1
                continue

            thickness = parse_float(raw_thickness)
            if thickness is None:
                invalid_thickness += 1
                continue

            if thickness <= max_thickness_mm:
                continue

            matched = (row.get("Matched_PNG_Path") or "").strip()
            src = resolve_series_path(matched, data_root)
            action = "exclude_over_thickness"
            move_status = "not_requested"
            destination = ""
            if src is None:
                move_status = "no_matched_png_path"

            context_keys = [
                "PatientID",
                "AnonPID",
                "SeriesInstanceUID",
                "Original_Folder",
                "Series_Type",
                "Class",
                "Group",
                "Overlap_Status",
            ]
            context = {k: (row.get(k) or "") for k in context_keys if k in row}

            records.append(
                ThicknessRecord(
                    row_index=row_index,
                    slice_thickness=thickness,
                    matched_png_path=matched,
                    action=action,
                    move_status=move_status,
                    destination=destination,
                    context=context,
                )
            )

    summary = {
        "mapping_csv": str(mapping_csv),
        "max_thickness_mm": max_thickness_mm,
        "total_rows": total_rows,
        "missing_thickness_rows": missing_thickness,
        "invalid_thickness_rows": invalid_thickness,
        "over_thickness_rows": len(records),
    }
    return records, summary


def move_records(
    records: list[ThicknessRecord],
    *,
    data_root: Path | None,
    archive_root: Path,
) -> list[ThicknessRecord]:
    moved: list[ThicknessRecord] = []
    archive_root.mkdir(parents=True, exist_ok=True)

    for record in records:
        src = resolve_series_path(record.matched_png_path, data_root)
        if src is None:
            moved.append(record)
            continue

        destination = ""
        move_status = "not_found"
        if src.exists() and src.is_dir():
            rel = safe_relative_path(src, data_root)
            dst = archive_root / rel
            destination = str(dst)
            if dst.exists():
                move_status = "destination_exists"
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                move_status = "moved"
        elif src.exists():
            move_status = "not_directory"

        moved.append(
            ThicknessRecord(
                row_index=record.row_index,
                slice_thickness=record.slice_thickness,
                matched_png_path=record.matched_png_path,
                action=record.action,
                move_status=move_status,
                destination=destination,
                context=record.context,
            )
        )
    return moved


def write_csv(records: list[ThicknessRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    context_keys = sorted({k for r in records for k in r.context.keys()})
    fieldnames = [
        "row_index",
        "slice_thickness",
        "matched_png_path",
        "action",
        "move_status",
        "destination",
        *context_keys,
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "row_index": r.row_index,
                "slice_thickness": r.slice_thickness,
                "matched_png_path": r.matched_png_path,
                "action": r.action,
                "move_status": r.move_status,
                "destination": r.destination,
            }
            row.update(r.context)
            writer.writerow(row)


def write_json(summary: dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and optionally quarantine CT series with SliceThickness > 1.5 mm."
    )
    parser.add_argument("--mapping-csv", required=True, help="CSV containing a SliceThickness column.")
    parser.add_argument("--data-root", default="", help="Dataset root used to resolve relative Matched_PNG_Path values.")
    parser.add_argument(
        "--archive-root",
        default="",
        help="Destination root for over-thickness series when --execute is set.",
    )
    parser.add_argument("--out-csv", default="removed_over_1.5mm.csv", help="Audit CSV output path.")
    parser.add_argument("--out-json", default="", help="Optional summary JSON path.")
    parser.add_argument("--max-thickness-mm", type=float, default=DEFAULT_MAX_THICKNESS_MM)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Move over-thickness series to --archive-root. Omit for dry-run audit only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mapping_csv = Path(args.mapping_csv)
    data_root = Path(args.data_root) if args.data_root else None
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")

    records, summary = collect_records(
        mapping_csv,
        max_thickness_mm=args.max_thickness_mm,
        data_root=data_root,
    )

    if args.execute:
        if not args.archive_root:
            raise ValueError("--archive-root is required when --execute is set")
        records = move_records(records, data_root=data_root, archive_root=Path(args.archive_root))
        summary["mode"] = "execute"
        summary["moved_rows"] = sum(1 for r in records if r.move_status == "moved")
        summary["not_found_rows"] = sum(1 for r in records if r.move_status == "not_found")
        summary["destination_exists_rows"] = sum(1 for r in records if r.move_status == "destination_exists")
    else:
        summary["mode"] = "dry_run"

    write_csv(records, out_csv)
    write_json(summary, out_json)

    print(f"Mode: {summary['mode']}")
    print(f"Mapping CSV: {mapping_csv}")
    print(f"Rows with SliceThickness > {args.max_thickness_mm:g} mm: {len(records)}")
    print(f"Audit CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")
    if not args.execute:
        print("Dry run only. No files were moved. Add --execute and --archive-root to quarantine folders.")


if __name__ == "__main__":
    main()
