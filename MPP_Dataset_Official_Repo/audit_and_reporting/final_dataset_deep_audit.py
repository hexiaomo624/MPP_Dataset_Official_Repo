"""Generate dataset-level audit statistics for PNG CT slices."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_GROUPS = ("HH", "Normal_MPP", "Severe_SMPP", "Viral")


def extract_patient_id(dir_name: str) -> str:
    if dir_name.startswith("Patient_"):
        parts = dir_name.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    parts = dir_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return dir_name


def calculate_image_stats(img_path: str) -> tuple[float, float, int]:
    try:
        with Image.open(img_path) as img:
            arr = np.array(img, dtype=np.float32) / 255.0
            return float(np.sum(arr)), float(np.sum(arr**2)), int(arr.size)
    except Exception:
        return 0.0, 0.0, 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate dataset-level audit statistics.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    p.add_argument("--out-json", default="dataset_global_stats.json")
    p.add_argument("--intensity-samples", type=int, default=5000)
    p.add_argument("--per-series-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    stats: dict[str, Any] = {
        "data_root": str(data_root),
        "groups": {},
        "global": {"unique_patients": 0, "total_series": 0, "total_pngs": 0, "empty_dirs": 0, "zero_byte_files": 0},
        "physical": {
            "resolutions": defaultdict(int),
            "slice_counts": [],
            "max_slices_case": {"id": "", "count": 0},
            "min_slices_case": {"id": "", "count": None},
        },
        "intensity": {"mean": 0.0, "std": 0.0},
    }

    global_patients: set[str] = set()
    intensity_pool: list[str] = []

    for group in args.groups:
        group_dir = data_root / group
        group_patients: set[str] = set()
        group_series = 0
        group_pngs = 0
        if group_dir.exists():
            for series_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                pngs = sorted(series_dir.glob("*.png"))
                num_pngs = len(pngs)
                if num_pngs == 0:
                    stats["global"]["empty_dirs"] += 1
                    continue

                patient_id = extract_patient_id(series_dir.name)
                group_patients.add(patient_id)
                global_patients.add(patient_id)
                group_series += 1
                group_pngs += num_pngs
                stats["physical"]["slice_counts"].append(num_pngs)

                if num_pngs > stats["physical"]["max_slices_case"]["count"]:
                    stats["physical"]["max_slices_case"] = {"id": series_dir.name, "count": num_pngs}
                min_case = stats["physical"]["min_slices_case"]
                if min_case["count"] is None or num_pngs < min_case["count"]:
                    stats["physical"]["min_slices_case"] = {"id": series_dir.name, "count": num_pngs}

                sample_img = pngs[0]
                if sample_img.stat().st_size == 0:
                    stats["global"]["zero_byte_files"] += 1
                try:
                    with Image.open(sample_img) as img:
                        w, h = img.size
                        stats["physical"]["resolutions"][f"{w}x{h}"] += 1
                except Exception:
                    pass

                sample_count = min(args.per_series_samples, len(pngs))
                intensity_pool.extend(str(p) for p in random.sample(pngs, sample_count))

        stats["groups"][group] = {
            "unique_patients": len(group_patients),
            "total_series": group_series,
            "total_pngs": group_pngs,
        }

    stats["global"]["unique_patients"] = len(global_patients)
    stats["global"]["total_series"] = sum(g["total_series"] for g in stats["groups"].values())
    stats["global"]["total_pngs"] = sum(g["total_pngs"] for g in stats["groups"].values())

    sample_size = min(args.intensity_samples, len(intensity_pool))
    intensity_samples = random.sample(intensity_pool, sample_size) if sample_size else []
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for s, sq_s, cnt in executor.map(calculate_image_stats, intensity_samples):
            total_sum += s
            total_sq_sum += sq_s
            total_pixels += cnt

    if total_pixels > 0:
        mean = total_sum / total_pixels
        variance = max(0.0, (total_sq_sum / total_pixels) - (mean**2))
        stats["intensity"]["mean"] = float(mean)
        stats["intensity"]["std"] = float(np.sqrt(variance))

    stats["physical"]["resolutions"] = dict(stats["physical"]["resolutions"])
    if stats["physical"]["min_slices_case"]["count"] is None:
        stats["physical"]["min_slices_case"] = {"id": "", "count": 0}

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit JSON: {out_json}")
    print(f"Patients: {stats['global']['unique_patients']}")
    print(f"Series: {stats['global']['total_series']}")
    print(f"PNGs: {stats['global']['total_pngs']}")


if __name__ == "__main__":
    main()
