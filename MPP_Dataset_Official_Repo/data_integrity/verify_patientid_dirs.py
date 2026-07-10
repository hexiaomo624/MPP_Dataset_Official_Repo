import argparse
import csv
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Issue:
    group: str
    patient_folder: str
    sample_file: str
    patientid_raw: str
    patientid_digits: str
    issue_type: str


@dataclass(frozen=True)
class PatientRecord:
    category: str
    library_root: str
    patient_folder: str
    patient_id: str
    sample_file: str


SKIP_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".md",
    ".zip",
}
ACCEPT_EXT = {".dcm", ".dicom", ".ima", ".cdm"}
DIGITS_RE = re.compile(r"(\d+)")


def _is_dicom_candidate_file_name(name: str) -> bool:
    if not name:
        return False
    upper = name.upper()
    if upper in {"DICOMDIR", "VERSION"}:
        return False
    ext = Path(name).suffix.lower()
    if ext in SKIP_EXT:
        return False
    if ext in ACCEPT_EXT:
        return True
    return "." not in name


def _pick_one_dicom_like_file(folder: Path) -> Path | None:
    try:
        with os.scandir(folder) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if _is_dicom_candidate_file_name(name):
                    return Path(e.path)
    except Exception:
        return None
    return None


def _pick_one_file_under_patient(patient_dir: Path) -> Path | None:
    f = _pick_one_dicom_like_file(patient_dir)
    if f is not None:
        return f

    try:
        with os.scandir(patient_dir) as it1:
            for e1 in it1:
                if not e1.is_dir():
                    continue
                d1 = Path(e1.path)
                f = _pick_one_dicom_like_file(d1)
                if f is not None:
                    return f
                try:
                    with os.scandir(d1) as it2:
                        for e2 in it2:
                            if not e2.is_dir():
                                continue
                            d2 = Path(e2.path)
                            f = _pick_one_dicom_like_file(d2)
                            if f is not None:
                                return f
                except Exception:
                    continue
    except Exception:
        return None

    return None


def _digits_only(s: str) -> str:
    if not s:
        return ""
    m = DIGITS_RE.search(s)
    return m.group(1) if m else ""


def _expected_patient_id_from_folder(*, group: str, patient_folder: str) -> str:
    if not patient_folder:
        return ""
    prefix = group.upper() + "_"
    if patient_folder.upper().startswith(prefix):
        return patient_folder.split("_", 1)[1].strip()
    return patient_folder.strip()


def _read_patient_id_from_file(path: Path) -> str:
    try:
        import pydicom  # type: ignore
    except Exception:
        return ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=["PatientID"],
            )
        pid = str(getattr(ds, "PatientID", "") or "").strip()
        return pid
    except Exception:
        return ""


def scan_dicom_integrity(
    root: Path,
    *,
    report_csv_path: Path,
    progress_every: int = 50_000,
) -> dict[str, int]:
    counters = {
        "candidates_total": 0,
        "ok_headers": 0,
        "failed_headers": 0,
        "zero_bytes": 0,
        "non_dicom": 0,
    }

    report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with report_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File_Path", "Size_Bytes", "Error_Type", "Error_Message"])

        try:
            import pydicom  # type: ignore
        except Exception as e:
            w.writerow(["", "", "PYDICOM_IMPORT_FAILED", str(e)])
            return counters

        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not _is_dicom_candidate_file_name(name):
                    continue

                counters["candidates_total"] += 1
                if counters["candidates_total"] % progress_every == 0:
                    print("candidates_checked", counters["candidates_total"], flush=True)

                fpath = Path(dirpath) / name
                try:
                    size = fpath.stat().st_size
                except Exception:
                    size = -1

                if size == 0:
                    counters["zero_bytes"] += 1
                    w.writerow([str(fpath), size, "ZERO_BYTES", ""])
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        pydicom.dcmread(
                            str(fpath),
                            stop_before_pixels=True,
                            force=True,
                            specific_tags=["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
                        )
                    counters["ok_headers"] += 1
                except Exception as e:
                    msg = str(e).replace("\r", " ").replace("\n", " ")
                    err_type = type(e).__name__
                    lowered = msg.lower()
                    if "file is missing dicom preamble" in lowered and "dicm" in lowered:
                        counters["non_dicom"] += 1
                    else:
                        counters["failed_headers"] += 1
                    w.writerow([str(fpath), size, err_type, msg[:500]])

    return counters


def _iter_patient_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.is_dir():
                    out.append(Path(e.path))
    except Exception:
        return []
    out.sort(key=lambda p: p.name)
    return out


def _read_patient_id_for_patient_dir(patient_dir: Path) -> tuple[str, str]:
    sample = _pick_one_file_under_patient(patient_dir)
    if sample is None:
        return "", ""
    pid = _read_patient_id_from_file(sample)
    return pid, str(sample)


def verify(root: Path, groups: list[str]) -> tuple[dict[str, dict[str, int]], list[Issue]]:
    summary: dict[str, dict[str, int]] = {}
    issues: list[Issue] = []

    for g in groups:
        gpath = root / g
        counters = {
            "patients_total": 0,
            "match_exact": 0,
            "mismatch": 0,
            "no_sample_file": 0,
            "empty_patientid": 0,
            "read_failed": 0,
        }

        if not gpath.exists():
            summary[g] = {"patients_total": 0, "missing_group_dir": 1}
            continue

        with os.scandir(gpath) as it:
            for e in it:
                if not e.is_dir():
                    continue
                counters["patients_total"] += 1
                patient_dir = Path(e.path)
                folder_name = patient_dir.name

                sample = _pick_one_file_under_patient(patient_dir)
                if sample is None:
                    counters["no_sample_file"] += 1
                    issues.append(
                        Issue(
                            group=g,
                            patient_folder=folder_name,
                            sample_file="",
                            patientid_raw="",
                            patientid_digits="",
                            issue_type="NO_SAMPLE_FILE",
                        )
                    )
                    continue

                pid = _read_patient_id_from_file(sample)
                if pid == "":
                    counters["empty_patientid"] += 1
                    issues.append(
                        Issue(
                            group=g,
                            patient_folder=folder_name,
                            sample_file=str(sample),
                            patientid_raw="",
                            patientid_digits="",
                            issue_type="EMPTY_PATIENT_ID",
                        )
                    )
                    continue

                expected = _expected_patient_id_from_folder(group=g, patient_folder=folder_name)
                if pid == expected:
                    counters["match_exact"] += 1
                else:
                    counters["mismatch"] += 1
                    issues.append(
                        Issue(
                            group=g,
                            patient_folder=folder_name,
                            sample_file=str(sample),
                            patientid_raw=pid,
                            patientid_digits=_digits_only(pid),
                            issue_type="MISMATCH",
                        )
                    )

        summary[g] = counters

    return summary, issues


def crosslib_counts_and_overlaps(base_root: Path) -> tuple[
    dict[str, int],
    dict[str, list[str]],
    list[tuple[str, list[str]]],
    dict[str, list[PatientRecord]],
]:
    base = base_root
    raw_root = base / "病例收集_已整理"

    candidates = [
        ("hh", raw_root / "HH", "HH"),
        ("mpp", raw_root / "MPP", "MPP"),
        ("smpp", raw_root / "SMPP", "SMPP"),
        ("viral", raw_root / "Viral", "Viral"),
        ("viral", raw_root / "VIRAL", "Viral"),
        ("mpp", base / "Organized_Normal_MPP", "Organized_Normal_MPP"),
        ("smpp", base / "Organized_Severe_SMPP", "Organized_Severe_SMPP"),
        ("viral", base / "Organized_Viral_Pneumonia", "Organized_Viral_Pneumonia"),
    ]

    sources = [(c, p, n) for (c, p, n) in candidates if p.exists()]

    category_to_pids: dict[str, set[str]] = {"hh": set(), "mpp": set(), "smpp": set(), "viral": set()}
    pid_to_categories: dict[str, set[str]] = {}
    pid_to_records: dict[str, list[PatientRecord]] = {}

    for category, lib_root, lib_name in sources:
        for pdir in _iter_patient_dirs(lib_root):
            pid, sample = _read_patient_id_for_patient_dir(pdir)
            if pid == "":
                if lib_name in {"HH", "MPP", "SMPP"}:
                    pid = _expected_patient_id_from_folder(group=lib_name, patient_folder=pdir.name)
                else:
                    pid = _digits_only(pdir.name) or pdir.name

            if pid == "":
                continue

            category_to_pids[category].add(pid)
            pid_to_categories.setdefault(pid, set()).add(category)
            pid_to_records.setdefault(pid, []).append(
                PatientRecord(
                    category=category,
                    library_root=str(lib_root),
                    patient_folder=pdir.name,
                    patient_id=pid,
                    sample_file=sample,
                )
            )

    counts = {k: len(v) for k, v in category_to_pids.items()}

    overlaps: dict[str, list[str]] = {}
    mpp_smpp = sorted(category_to_pids["mpp"].intersection(category_to_pids["smpp"]))
    overlaps["mpp&smpp"] = mpp_smpp

    multi_category = sorted((pid, sorted(list(cats))) for pid, cats in pid_to_categories.items() if len(cats) > 1)
    return counts, overlaps, multi_category, pid_to_records


def verify_all_files(
    root: Path,
    groups: list[str],
    *,
    report_csv_path: Path,
    progress_every: int = 50_000,
) -> tuple[dict[str, dict[str, int]], list[Issue], Path | None]:
    summary: dict[str, dict[str, int]] = {}
    first_issues: list[Issue] = []

    csv_file = None
    csv_writer: csv.writer | None = None
    report_written: Path | None = None

    try:
        for g in groups:
            gpath = root / g
            counters = {
                "files_checked": 0,
                "match_exact": 0,
                "mismatch": 0,
                "empty_patientid": 0,
                "read_failed": 0,
                "outside_patient_dir": 0,
            }

            if not gpath.exists():
                summary[g] = {"files_checked": 0, "missing_group_dir": 1}
                continue

            for dirpath, _, filenames in os.walk(gpath):
                try:
                    rel = Path(dirpath).relative_to(gpath)
                except Exception:
                    rel = Path(dirpath)

                patient_folder = rel.parts[0] if rel.parts else ""
                if not patient_folder:
                    patient_folder = ""

                for name in filenames:
                    if not _is_dicom_candidate_file_name(name):
                        continue

                    counters["files_checked"] += 1
                    if counters["files_checked"] % progress_every == 0:
                        print(g, "files_checked", counters["files_checked"], flush=True)

                    if not patient_folder:
                        counters["outside_patient_dir"] += 1
                        continue

                    fpath = Path(dirpath) / name
                    pid = _read_patient_id_from_file(fpath)
                    if pid == "":
                        counters["empty_patientid"] += 1
                        issue = Issue(
                            group=g,
                            patient_folder=patient_folder,
                            sample_file=str(fpath),
                            patientid_raw="",
                            patientid_digits="",
                            issue_type="EMPTY_PATIENT_ID",
                        )
                        if len(first_issues) < 200:
                            first_issues.append(issue)
                        if csv_writer is None:
                            report_csv_path.parent.mkdir(parents=True, exist_ok=True)
                            csv_file = report_csv_path.open("w", encoding="utf-8-sig", newline="")
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(
                                [
                                    "Group",
                                    "Patient_Folder",
                                    "File_Path",
                                    "PatientID_Raw",
                                    "PatientID_Digits",
                                    "Issue_Type",
                                ]
                            )
                            report_written = report_csv_path
                        csv_writer.writerow(
                            [
                                issue.group,
                                issue.patient_folder,
                                issue.sample_file,
                                issue.patientid_raw,
                                issue.patientid_digits,
                                issue.issue_type,
                            ]
                        )
                        continue

                    expected = _expected_patient_id_from_folder(group=g, patient_folder=patient_folder)
                    if pid == expected:
                        counters["match_exact"] += 1
                        continue

                    counters["mismatch"] += 1
                    issue = Issue(
                        group=g,
                        patient_folder=patient_folder,
                        sample_file=str(fpath),
                        patientid_raw=pid,
                        patientid_digits=_digits_only(pid),
                        issue_type="MISMATCH",
                    )
                    if len(first_issues) < 200:
                        first_issues.append(issue)
                    if csv_writer is None:
                        report_csv_path.parent.mkdir(parents=True, exist_ok=True)
                        csv_file = report_csv_path.open("w", encoding="utf-8-sig", newline="")
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(
                            [
                                "Group",
                                "Patient_Folder",
                                "File_Path",
                                "PatientID_Raw",
                                "PatientID_Digits",
                                "Issue_Type",
                            ]
                        )
                        report_written = report_csv_path
                    csv_writer.writerow(
                        [
                            issue.group,
                            issue.patient_folder,
                            issue.sample_file,
                            issue.patientid_raw,
                            issue.patientid_digits,
                            issue.issue_type,
                        ]
                    )

            summary[g] = counters
    finally:
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass

    return summary, first_issues, report_written


def write_csv_report(path: Path, issues: list[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Group",
                "Patient_Folder",
                "Sample_File",
                "PatientID_Raw",
                "PatientID_Digits",
                "Issue_Type",
            ]
        )
        for it in issues:
            w.writerow(
                [
                    it.group,
                    it.patient_folder,
                    it.sample_file,
                    it.patientid_raw,
                    it.patientid_digits,
                    it.issue_type,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DICOM PatientID consistency and integrity.")
    parser.add_argument("--root", required=False, help="Root containing group folders for sample/all/integrity modes.")
    parser.add_argument("--groups", nargs="+", default=["HH", "MPP", "SMPP"])
    parser.add_argument("--mode", default="sample", choices=["sample", "all", "integrity", "corrupt", "check", "crosslib", "dedup", "overlap"])
    parser.add_argument("--base-root", default="", help="Base root for cross-library overlap mode.")
    parser.add_argument("--out-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--out-dir", default=".", help="Directory for default CSV outputs.")
    parser.add_argument("--all", action="store_true", help="Alias for --mode all.")
    args = parser.parse_args()

    mode = "all" if args.all else args.mode
    root = Path(args.root) if args.root else None
    groups = args.groups
    out_dir = Path(args.out_dir)

    if mode in {"crosslib", "dedup", "overlap"}:
        if not args.base_root:
            raise ValueError("--base-root is required for crosslib/dedup/overlap mode")
        counts, overlaps, multi_category, pid_to_records = crosslib_counts_and_overlaps(Path(args.base_root))
        intra_dups: dict[str, list[str]] = {"hh": [], "mpp": [], "smpp": [], "viral": []}
        for pid, recs in pid_to_records.items():
            by_cat: dict[str, set[str]] = {}
            for r in recs:
                by_cat.setdefault(r.category, set()).add(r.patient_folder)
            for cat, folders in by_cat.items():
                if len(folders) > 1:
                    intra_dups.setdefault(cat, []).append(pid)
        for cat in intra_dups.keys():
            intra_dups[cat] = sorted(set(intra_dups[cat]))

        print("=== UNIQUE PATIENT COUNTS (dedup by PatientID) ===")
        for k in ["hh", "mpp", "smpp", "viral"]:
            print(k, counts.get(k, 0))
        print("total_unique_all", len(pid_to_records))
        print("mpp&smpp_overlap", len(overlaps.get("mpp&smpp", [])))

        dup_any = [pid for pid, cats in multi_category]
        print("overlap_any_categories", len(dup_any))
        print(
            "intra_category_duplicate_patientid_counts",
            {k: len(v) for k, v in intra_dups.items()},
        )

        if overlaps.get("mpp&smpp"):
            print("mpp&smpp_ids", ",".join(overlaps["mpp&smpp"][:200]))

        if dup_any:
            out_csv = Path(args.out_csv) if args.out_csv else out_dir / "patient_overlap_hh_mpp_smpp_viral.csv"
            with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["PatientID", "Categories", "Records"])
                for pid, cats in multi_category:
                    recs = pid_to_records.get(pid, [])
                    rec_text = " | ".join(f"{r.category}@{r.library_root}\\{r.patient_folder}" for r in recs)
                    w.writerow([pid, "+".join(cats), rec_text])
            print("overlap_csv", out_csv)
        elif any(intra_dups.values()):
            out_csv = Path(args.out_csv) if args.out_csv else out_dir / "patient_duplicate_within_category.csv"
            with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Category", "PatientID", "Folders"])
                for cat in ["hh", "mpp", "smpp", "viral"]:
                    for pid in intra_dups.get(cat, []):
                        folders = sorted({r.patient_folder for r in pid_to_records.get(pid, []) if r.category == cat})
                        w.writerow([cat, pid, " | ".join(folders)])
            print("intra_dup_csv", out_csv)
        return

    if mode in {"integrity", "corrupt", "check"}:
        if root is None:
            raise ValueError("--root is required for integrity/corrupt/check mode")
        target = root
        out_csv = Path(args.out_csv) if args.out_csv else out_dir / "dicom_integrity.csv"
        counters = scan_dicom_integrity(target, report_csv_path=out_csv)
        print("=== DICOM INTEGRITY SCAN ===")
        print("ROOT:", target)
        print("counters:", counters)
        print("report_csv:", out_csv)
        return

    if mode == "all":
        if root is None:
            raise ValueError("--root is required for all mode")
        report_csv = Path(args.out_csv) if args.out_csv else out_dir / "patientid_verify_allfiles.csv"
        summary, issues, written = verify_all_files(root, groups, report_csv_path=report_csv)
    else:
        if root is None:
            raise ValueError("--root is required for sample mode")
        summary, issues = verify(root, groups)
        written = None
        if issues:
            written = Path(args.out_csv) if args.out_csv else out_dir / "patientid_verify.csv"
            write_csv_report(written, issues)

    print("=== VERIFY PatientID vs folder name ===")
    print("ROOT:", root)
    for g in groups:
        print(g, summary.get(g))

    print("issues_total:", len(issues))
    for it in issues[:30]:
        print(
            "ISSUE",
            {
                "group": it.group,
                "folder": it.patient_folder,
                "dicom_patientid": it.patientid_raw,
                "sample": it.sample_file,
                "type": it.issue_type,
            },
        )

    if written is not None:
        print("report_csv:", written)


if __name__ == "__main__":
    sys.exit(main())
