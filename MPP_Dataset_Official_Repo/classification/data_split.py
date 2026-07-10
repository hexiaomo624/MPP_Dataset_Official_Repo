import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


PATIENT_ID_RE = re.compile(r"P\d{6}")


@dataclass(frozen=True)
class PatientRecord:
    anon_pid: str
    original_label: str
    rel_paths: tuple[str, ...]


def _extract_anon_pid(patient_sequence_dirname: str) -> str:
    m = PATIENT_ID_RE.search(patient_sequence_dirname)
    if not m:
        raise ValueError(f"Cannot parse AnonPID from folder name: {patient_sequence_dirname}")
    return m.group(0)


def scan_dataset(data_root: Path) -> list[PatientRecord]:
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    known = {"Normal_MPP", "Severe_SMPP", "HH", "Viral", "MPP", "SMPP"}
    class_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name in known]
    if not class_dirs:
        ignored = {"paper", "splits", "outputs", "__pycache__"}
        candidates = [p for p in data_root.iterdir() if p.is_dir() and p.name not in ignored]
        class_dirs = []
        for c in candidates:
            try:
                next(c.glob("*/*.png"))
                class_dirs.append(c)
            except StopIteration:
                continue
    if not class_dirs:
        raise RuntimeError(f"No class folders with PNG slices found under: {data_root}")

    patient_to_label: dict[str, str] = {}
    patient_to_paths: dict[str, list[str]] = {}

    for class_dir in sorted(class_dirs, key=lambda p: p.name):
        original_label = class_dir.name
        for seq_dir in class_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            anon_pid = _extract_anon_pid(seq_dir.name)
            existing = patient_to_label.get(anon_pid)
            if existing is None:
                patient_to_label[anon_pid] = original_label
            elif existing != original_label:
                raise RuntimeError(
                    f"AnonPID appears in multiple class folders: {anon_pid} -> {existing} and {original_label}"
                )

            png_paths = sorted(seq_dir.glob("*.png"))
            if not png_paths:
                continue

            for p in png_paths:
                rel_path = p.relative_to(data_root).as_posix()
                patient_to_paths.setdefault(anon_pid, []).append(rel_path)

    records: list[PatientRecord] = []
    for anon_pid, original_label in sorted(patient_to_label.items(), key=lambda kv: kv[0]):
        rel_paths = patient_to_paths.get(anon_pid, [])
        if not rel_paths:
            continue
        records.append(PatientRecord(anon_pid=anon_pid, original_label=original_label, rel_paths=tuple(rel_paths)))

    if not records:
        raise RuntimeError("No PNG slices found while scanning dataset.")

    return records


def split_patients(
    records: list[PatientRecord],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[set[str], set[str], set[str]]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")

    anon_pids = [r.anon_pid for r in records]
    labels = [r.original_label for r in records]

    train_pids, temp_pids, train_labels, temp_labels = train_test_split(
        anon_pids,
        labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    val_size_of_temp = val_ratio / (val_ratio + test_ratio)
    val_pids, test_pids = train_test_split(
        temp_pids,
        test_size=(1.0 - val_size_of_temp),
        random_state=seed,
        shuffle=True,
        stratify=temp_labels,
    )

    return set(train_pids), set(val_pids), set(test_pids)


def write_split_csv(
    records: list[PatientRecord],
    out_dir: Path,
    train_pids: set[str],
    val_pids: set[str],
    test_pids: set[str],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    patients_csv = out_dir / "patients_split.csv"
    slices_csv = out_dir / "slices_split.csv"

    with patients_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anon_pid", "original_label", "split"])
        for r in records:
            if r.anon_pid in train_pids:
                split = "train"
            elif r.anon_pid in val_pids:
                split = "val"
            elif r.anon_pid in test_pids:
                split = "test"
            else:
                continue
            w.writerow([r.anon_pid, r.original_label, split])

    with slices_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "anon_pid", "original_label", "split"])
        for r in records:
            if r.anon_pid in train_pids:
                split = "train"
            elif r.anon_pid in val_pids:
                split = "val"
            elif r.anon_pid in test_pids:
                split = "test"
            else:
                continue
            for rel_path in r.rel_paths:
                w.writerow([rel_path, r.anon_pid, r.original_label, split])

    return patients_csv, slices_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root. Structure: /class/patient_sequence/slice_xxx.png",
    )
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "splits"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    records = scan_dataset(data_root)
    train_pids, val_pids, test_pids = split_patients(records, seed=args.seed)
    patients_csv, slices_csv = write_split_csv(records, out_dir, train_pids, val_pids, test_pids)

    train_n = sum(1 for r in records if r.anon_pid in train_pids)
    val_n = sum(1 for r in records if r.anon_pid in val_pids)
    test_n = sum(1 for r in records if r.anon_pid in test_pids)

    print(f"Patients: train={train_n}, val={val_n}, test={test_n}")
    print(f"Wrote: {patients_csv}")
    print(f"Wrote: {slices_csv}")


if __name__ == "__main__":
    main()
