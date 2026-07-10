import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TaskName = Literal["A", "B", "C"]


TASK_CLASSES: dict[TaskName, list[str]] = {
    "A": ["Normal_MPP", "Severe_SMPP"],
    "B": ["MPP", "HH", "Viral"],
    "C": ["HH", "MPP", "SMPP", "Viral"],
}


def canonicalize_label(original_label: str) -> str:
    mapping = {
        "Normal_MPP": "MPP",
        "Severe_SMPP": "SMPP",
    }
    return mapping.get(original_label, original_label)


def task_label(task: TaskName, original_label: str) -> str | None:
    if task == "A":
        label = original_label
    else:
        label = canonicalize_label(original_label)
    if label in TASK_CLASSES[task]:
        return label
    return None


def build_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


@dataclass(frozen=True)
class Sample:
    abs_path: str
    y: int
    anon_pid: str
    label_name: str


class CTSliceDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: Literal["train", "val", "test"],
        task: TaskName,
        image_size: int = 224,
        return_meta: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv)
        self.split = split
        self.task = task
        self.return_meta = return_meta
        self.transform = build_transforms(image_size=image_size)

        if not self.split_csv.exists():
            raise FileNotFoundError(f"split_csv not found: {self.split_csv}")

        classes = TASK_CLASSES[task]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        samples: list[Sample] = []
        with self.split_csv.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            required = {"rel_path", "anon_pid", "original_label", "split"}
            if not required.issubset(set(r.fieldnames or [])):
                raise ValueError(f"split_csv must contain columns: {sorted(required)}")
            for row in r:
                if row["split"] != split:
                    continue
                label_name = task_label(task, row["original_label"])
                if label_name is None:
                    continue
                y = class_to_idx[label_name]
                abs_path = str((self.data_root / row["rel_path"]).resolve())
                samples.append(
                    Sample(
                        abs_path=abs_path,
                        y=y,
                        anon_pid=row["anon_pid"],
                        label_name=label_name,
                    )
                )

        if not samples:
            raise RuntimeError(f"No samples found for task={task}, split={split}.")

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.abs_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(s.y, dtype=torch.long)
        if self.return_meta:
            return x, y, {"anon_pid": s.anon_pid, "path": s.abs_path, "label_name": s.label_name}
        return x, y
