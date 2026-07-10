"""Audit and optionally quarantine blurry CT PNG slices.

The script estimates sharpness with Laplacian variance. It flags edge slices,
low-sharpness slices, and heavily degraded series. By default it only writes
audit CSV/JSON files. Add ``--execute --archive-root`` to quarantine flagged
files.
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
class MotionDecision:
    group: str
    series_id: str
    path: str
    remove: bool
    reason: str
    sharpness: float
    move_status: str = "not_requested"
    destination: str = ""


def read_image_cv2(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def sharpness_score(path: Path) -> float:
    img = read_image_cv2(path)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def discover_series(data_root: Path, groups: list[str]) -> list[tuple[str, str, Path]]:
    series = []
    for group in groups:
        group_dir = data_root / group
        if not group_dir.is_dir():
            continue
        for series_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            series.append((group, series_dir.name, series_dir))
    return series


def score_series(task: tuple[str, str, str, int]) -> dict[str, Any]:
    group, series_id, series_dir_text, trim_slices = task
    series_dir = Path(series_dir_text)
    pngs = sorted(series_dir.glob("*.png"))
    if len(pngs) <= trim_slices * 2:
        return {"group": group, "series_id": series_id, "series_dir": str(series_dir), "status": "too_short", "pngs": [str(p) for p in pngs], "scores": {}}

    edge = pngs[:trim_slices] + pngs[-trim_slices:] if trim_slices > 0 else []
    eval_pngs = pngs[trim_slices:-trim_slices] if trim_slices > 0 else pngs
    scores = {str(p): sharpness_score(p) for p in eval_pngs}
    return {
        "group": group,
        "series_id": series_id,
        "series_dir": str(series_dir),
        "status": "ok",
        "edge": [str(p) for p in edge],
        "eval": [str(p) for p in eval_pngs],
        "scores": scores,
    }


def build_decisions(
    results: list[dict[str, Any]],
    *,
    threshold: float,
    max_blurry_fraction: float,
    min_remaining_slices: int,
) -> list[MotionDecision]:
    decisions: list[MotionDecision] = []
    for res in results:
        group = res["group"]
        series_id = res["series_id"]
        if res["status"] == "too_short":
            for path in res["pngs"]:
                decisions.append(MotionDecision(group, series_id, path, True, "reject_series_too_short", 0.0))
            continue

        blurry = [p for p in res["eval"] if res["scores"].get(p, 0.0) < threshold]
        eval_count = max(1, len(res["eval"]))
        reject_series = (len(blurry) / eval_count) > max_blurry_fraction
        remaining = len(res["edge"]) + len(res["eval"]) - len(res["edge"]) - len(blurry)
        if remaining < min_remaining_slices:
            reject_series = True

        if reject_series:
            for path in res["edge"] + res["eval"]:
                decisions.append(MotionDecision(group, series_id, path, True, "reject_series_motion_or_short", res["scores"].get(path, 0.0)))
            continue

        edge_set = set(res["edge"])
        blurry_set = set(blurry)
        for path in res["edge"] + res["eval"]:
            if path in edge_set:
                decisions.append(MotionDecision(group, series_id, path, True, "edge_trim", res["scores"].get(path, 0.0)))
            elif path in blurry_set:
                decisions.append(MotionDecision(group, series_id, path, True, "sharpness_below_threshold", res["scores"].get(path, 0.0)))
            else:
                decisions.append(MotionDecision(group, series_id, path, False, "valid", res["scores"].get(path, 0.0)))
    return decisions


def safe_relative_path(src: Path, data_root: Path) -> Path:
    try:
        return src.resolve().relative_to(data_root.resolve())
    except Exception:
        return Path(src.name)


def quarantine(decisions: list[MotionDecision], data_root: Path, archive_root: Path) -> list[MotionDecision]:
    archive_root.mkdir(parents=True, exist_ok=True)
    updated: list[MotionDecision] = []
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
        updated.append(MotionDecision(d.group, d.series_id, d.path, d.remove, d.reason, d.sharpness, status, str(dst)))
    return updated


def write_csv(decisions: list[MotionDecision], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MotionDecision.__dataclass_fields__.keys())
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in decisions:
            writer.writerow(asdict(d))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit motion artifact and blur in PNG CT slices.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    p.add_argument("--out-csv", default="motion_artifact_audit.csv")
    p.add_argument("--out-json", default="")
    p.add_argument("--trim-slices", type=int, default=5)
    p.add_argument("--percentile", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=None, help="Fixed sharpness threshold. Overrides --percentile.")
    p.add_argument("--max-blurry-fraction", type=float, default=0.30)
    p.add_argument("--min-remaining-slices", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--execute", action="store_true")
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
    worker_tasks = [(g, sid, str(p), args.trim_slices) for g, sid, p in series]
    results: list[dict[str, Any]] = []
    if worker_tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            results = list(executor.map(score_series, worker_tasks))

    all_scores = [score for res in results if res["status"] == "ok" for score in res["scores"].values()]
    threshold = float(args.threshold) if args.threshold is not None else float(np.percentile(all_scores, args.percentile)) if all_scores else 0.0
    decisions = build_decisions(results, threshold=threshold, max_blurry_fraction=args.max_blurry_fraction, min_remaining_slices=args.min_remaining_slices)
    if args.execute:
        decisions = quarantine(decisions, data_root, Path(args.archive_root))

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    write_csv(decisions, out_csv)
    summary = {
        "mode": "execute" if args.execute else "dry_run",
        "data_root": str(data_root),
        "series_scanned": len(series),
        "slices_scanned": len(decisions),
        "flagged_slices": sum(1 for d in decisions if d.remove),
        "moved_slices": sum(1 for d in decisions if d.move_status == "moved"),
        "threshold": threshold,
        "threshold_source": "fixed" if args.threshold is not None else f"percentile_{args.percentile:g}",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Dry run only. No files were moved.")


if __name__ == "__main__":
    main()
