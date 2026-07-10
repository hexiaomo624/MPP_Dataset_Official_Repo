"""Convert DICOM CT series to standardized lung-window PNG slices.

This is the manuscript preprocessing pipeline:
  - read DICOM slices and convert stored values to HU using RescaleSlope and
    RescaleIntercept
  - resample each 3D series to 1.0 mm isotropic spacing with B-Spline
    interpolation
  - identify the axial lung range using HU in [-1000, -400] with a 5% rule
  - apply lung window WL=-600, WW=1500 and save 8-bit PNG slices

The script has no site-specific paths. Provide source directories using either
``--source-config`` JSON or one or more ``--source Group=/path/to/dicom_root``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image


def read_dicom_series(dir_path: Path) -> sitk.Image:
    files = [
        p
        for p in dir_path.iterdir()
        if p.is_file() and (p.suffix.lower() == ".dcm" or ("." not in p.name and "DICOMDIR" not in p.name.upper()))
    ]
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            pass

    if not slices:
        raise ValueError(f"No valid DICOMs found in {dir_path}")
    if len(slices) < 3:
        raise ValueError("SKIP_SCOUT: fewer than 3 slices")

    slices.sort(key=lambda x: float(getattr(x, "InstanceNumber", 0)))
    ref_ds = slices[0]
    spacing = [
        float(ref_ds.PixelSpacing[0]),
        float(ref_ds.PixelSpacing[1]),
        float(getattr(ref_ds, "SliceThickness", 1.0)),
    ]

    try:
        z1 = float(slices[0].ImagePositionPatient[2])
        z2 = float(slices[1].ImagePositionPatient[2])
        z_spacing = abs(z1 - z2)
        if z_spacing > 0:
            spacing[2] = z_spacing
    except Exception:
        pass

    origin = [0.0, 0.0, 0.0]
    if hasattr(ref_ds, "ImagePositionPatient"):
        origin = [float(x) for x in ref_ds.ImagePositionPatient]

    direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if hasattr(ref_ds, "ImageOrientationPatient"):
        iop = ref_ds.ImageOrientationPatient
        direction = [float(iop[0]), float(iop[1]), float(iop[2]), float(iop[3]), float(iop[4]), float(iop[5]), 0.0, 0.0, 1.0]

    img_list = []
    for ds in slices:
        image = ds.pixel_array.astype(np.float32)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        img_list.append(image * slope + intercept)

    sitk_img = sitk.GetImageFromArray(np.stack(img_list))
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)
    return sitk_img


def resample_image(image: sitk.Image, new_spacing: list[float]) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        max(1, int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(image)


def apply_window(image_arr: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    windowed = np.clip(image_arr, img_min, img_max)
    windowed = (windowed - img_min) / window_width * 255.0
    return windowed.astype(np.uint8)


def process_case(args: tuple[str, str, str, str, list[float], float, float, float, float, float]) -> dict[str, Any]:
    source_dir, output_dir, group_name, patient_id, spacing, wc, ww, min_hu, max_hu, min_lung_ratio = args
    try:
        image = read_dicom_series(Path(source_dir))
        resampled_img = resample_image(image, new_spacing=spacing)
        img_arr = sitk.GetArrayFromImage(resampled_img)

        z_sum = np.sum((img_arr >= min_hu) & (img_arr <= max_hu), axis=(1, 2))
        total_pixels_per_slice = img_arr.shape[1] * img_arr.shape[2]
        valid_slices = np.where(z_sum > total_pixels_per_slice * min_lung_ratio)[0]
        if len(valid_slices) > 0:
            start_z, end_z = int(valid_slices[0]), int(valid_slices[-1])
        else:
            start_z, end_z = 0, len(img_arr) - 1

        png_arr = apply_window(img_arr[start_z : end_z + 1], wc, ww)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for z in range(png_arr.shape[0]):
            Image.fromarray(png_arr[z]).save(out / f"slice_{z:04d}.png")

        return {"status": "success", "group": group_name, "patient_id": patient_id, "slices": int(png_arr.shape[0]), "error": ""}
    except Exception as e:
        status = "skip_scout" if "SKIP_SCOUT" in str(e) else "error"
        return {"status": status, "group": group_name, "patient_id": patient_id, "slices": 0, "error": str(e)}


def find_leaf_dicom_dirs(root_dir: Path) -> list[Path]:
    dicom_dirs: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if any(f.lower().endswith(".dcm") or ("." not in f and "DICOMDIR" not in f.upper()) for f in filenames):
            dicom_dirs.append(Path(dirpath))
    return dicom_dirs


def parse_sources(args: argparse.Namespace) -> dict[str, str]:
    sources: dict[str, str] = {}
    if args.source_config:
        with Path(args.source_config).open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--source-config must contain a JSON object mapping group names to paths")
        sources.update({str(k): str(v) for k, v in loaded.items()})

    for item in args.source or []:
        if "=" not in item:
            raise ValueError("--source entries must use Group=/path/to/root")
        group, path = item.split("=", 1)
        sources[group.strip()] = path.strip()

    if not sources:
        raise ValueError("Provide --source-config or at least one --source Group=/path/to/root")
    return sources


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert DICOM CT series to lung-window PNG slices.")
    p.add_argument("--source-config", default="", help="JSON object mapping output group names to DICOM roots.")
    p.add_argument("--source", action="append", help="Source mapping in the form Group=/path/to/dicom_root. Repeatable.")
    p.add_argument("--output-root", required=True, help="Destination root for PNG series.")
    p.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Output spacing in mm.")
    p.add_argument("--window-center", type=float, default=-600.0)
    p.add_argument("--window-width", type=float, default=1500.0)
    p.add_argument("--min-lung-hu", type=float, default=-1000.0)
    p.add_argument("--max-lung-hu", type=float, default=-400.0)
    p.add_argument("--min-lung-ratio", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--force", action="store_true", help="Reprocess series even when PNG output already exists.")
    p.add_argument("--audit-json", default="", help="Audit JSON path. Defaults to <output-root>/audit_report.json.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    source_libs = parse_sources(args)
    output_root = Path(args.output_root)
    audit_json = Path(args.audit_json) if args.audit_json else output_root / "audit_report.json"

    tasks = []
    print("Scanning source directories...")
    for group_name, source_path in source_libs.items():
        root = Path(source_path)
        if not root.exists():
            print(f"Warning: source path not found: {root}")
            continue
        for leaf in find_leaf_dicom_dirs(root):
            rel_path = leaf.relative_to(root)
            patient_id = "_".join(rel_path.parts)
            out_dir = output_root / group_name / patient_id
            if not args.force and out_dir.exists() and any(p.suffix.lower() == ".png" for p in out_dir.iterdir()):
                continue
            tasks.append((str(leaf), str(out_dir), group_name, patient_id, args.spacing, args.window_center, args.window_width, args.min_lung_hu, args.max_lung_hu, args.min_lung_ratio))

    print(f"Found {len(tasks)} DICOM series to process.")
    results = []
    start_time = time.time()
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for i, res in enumerate(executor.map(process_case, tasks), start=1):
                results.append(res)
                if i % 10 == 0 or i == len(tasks):
                    print(f"Processed {i}/{len(tasks)} cases...")

    audit: dict[str, Any] = defaultdict(lambda: {"cases": 0, "total_slices": 0, "failed": 0, "skipped_scout": 0, "errors": []})
    for r in results:
        g = r["group"]
        if r["status"] == "success":
            audit[g]["cases"] += 1
            audit[g]["total_slices"] += r["slices"]
        elif r["status"] == "skip_scout":
            audit[g]["skipped_scout"] += 1
        else:
            audit[g]["failed"] += 1
            audit[g]["errors"].append({"patient_id": r["patient_id"], "error": r["error"]})

    summary = {
        "output_root": str(output_root),
        "sources": source_libs,
        "spacing": args.spacing,
        "window_center": args.window_center,
        "window_width": args.window_width,
        "min_lung_hu": args.min_lung_hu,
        "max_lung_hu": args.max_lung_hu,
        "min_lung_ratio": args.min_lung_ratio,
        "elapsed_minutes": (time.time() - start_time) / 60,
        "groups": dict(audit),
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit JSON: {audit_json}")


if __name__ == "__main__":
    main()
