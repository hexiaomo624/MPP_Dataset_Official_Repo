import argparse
import random
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D


ROOT_DEFAULT = Path(".")
GROUPS = ["HH", "Normal_MPP", "Severe_SMPP", "Viral"]


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def iter_pngs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.png"):
        if p.is_file():
            yield p


def read_png_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img, dtype=np.float32)
    return arr


def robust_window(img: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> Tuple[float, float]:
    lo = float(np.percentile(img, low_q))
    hi = float(np.percentile(img, high_q))
    if hi <= lo + 1e-6:
        lo = float(np.min(img))
        hi = float(np.max(img))
        if hi <= lo + 1e-6:
            hi = lo + 1.0
    return lo, hi


def extract_slice_num(p: Path) -> Optional[int]:
    m = re.findall(r"\d+", p.stem)
    if not m:
        return None
    return int(m[-1])


def fig2_random_2x2(root: Path, out_path: Path, seed: Optional[int] = None) -> None:
    rng = random.Random(seed)

    picks: List[Tuple[str, Path]] = []
    for g in GROUPS:
        gdir = root / g
        pngs = list(iter_pngs(gdir))
        if not pngs:
            raise RuntimeError(f"No PNG found under: {gdir}")
        picks.append((g, rng.choice(pngs)))

    imgs = [read_png_grayscale(p) for _, p in picks]
    vmins, vmaxs = zip(*(robust_window(im) for im in imgs))
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0))
    for ax, (label, _), im in zip(axes.ravel(), picks, imgs):
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(label, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    fig.suptitle("Random CT slice examples (2×2)", y=0.98, fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_final_pathology_comparison(
    *,
    hh_path: Path,
    normal_mpp_path: Path,
    severe_smpp_path: Path,
    viral_path: Path,
    out_path: Path,
) -> None:
    inputs = [
        ("(a) HH", hh_path),
        ("(b) Normal MPP", normal_mpp_path),
        ("(c) Severe SMPP", severe_smpp_path),
        ("(d) Viral", viral_path),
    ]

    for _, p in inputs:
        if not p.exists():
            raise FileNotFoundError(str(p))

    imgs = [read_png_grayscale(p) for _, p in inputs]
    vmins, vmaxs = zip(*(robust_window(im) for im in imgs))
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    max_h = max(int(im.shape[0]) for im in imgs)
    max_w = max(int(im.shape[1]) for im in imgs)
    base_dpi = 300.0
    fig_w = max(6.0, (2.0 * max_w) / base_dpi)
    fig_h = max(6.0, (2.0 * max_h) / base_dpi + 0.6)

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=base_dpi)
    fig.patch.set_facecolor("black")

    title_font = {"family": ["Arial", "Times New Roman", "DejaVu Sans"], "weight": "bold", "size": 14}

    for ax, (title, _), im in zip(axes.ravel(), inputs, imgs):
        ax.set_facecolor("black")
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")
        ax.set_title(title, color="white", pad=8, fontdict=title_font)

    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def fig2_standard_style_pathology_comparison(
    *,
    hh_path: Path,
    normal_mpp_path: Path,
    severe_smpp_path: Path,
    viral_path: Path,
    out_path: Path,
) -> None:
    inputs = [
        ("a", "HH", hh_path),
        ("b", "Normal MPP", normal_mpp_path),
        ("c", "Severe SMPP", severe_smpp_path),
        ("d", "Viral", viral_path),
    ]

    for _, _, p in inputs:
        if not p.exists():
            raise FileNotFoundError(str(p))

    imgs = [read_png_grayscale(p) for _, _, p in inputs]
    vmins, vmaxs = zip(*(robust_window(im) for im in imgs))
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    max_h = max(int(im.shape[0]) for im in imgs)
    max_w = max(int(im.shape[1]) for im in imgs)
    base_dpi = 300.0
    fig_w = max(6.5, (2.0 * max_w) / base_dpi + 0.8)
    fig_h = max(6.5, (2.0 * max_h) / base_dpi + 0.8)

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=base_dpi)
    fig.patch.set_facecolor("white")

    label_font = {"family": ["Arial", "DejaVu Sans"], "weight": "bold", "size": 14}
    name_font = {"family": ["Arial", "DejaVu Sans"], "weight": "normal", "size": 13}

    for ax, (letter, name, _), im in zip(axes.ravel(), inputs, imgs):
        ax.set_facecolor("white")
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")

        ax.text(
            0.01,
            0.99,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            fontdict=label_font,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85),
        )
        ax.text(
            0.5,
            -0.08,
            name,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="black",
            fontdict=name_font,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#B0B0B0")

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
def find_6_consecutive_slices_in_series(series_dir: Path) -> Optional[List[Path]]:
    pngs = [p for p in series_dir.glob("*.png") if p.is_file()]
    items: List[Tuple[int, Path]] = []
    for p in pngs:
        n = extract_slice_num(p)
        if n is None:
            continue
        items.append((n, p))
    if len(items) < 6:
        return None

    items.sort(key=lambda x: x[0])
    nums = [n for n, _ in items]
    paths = [p for _, p in items]

    index_by_num = {n: i for i, n in enumerate(nums)}
    for start in nums:
        ok = True
        idxs: List[int] = []
        for k in range(6):
            idx = index_by_num.get(start + k)
            if idx is None:
                ok = False
                break
            idxs.append(idx)
        if ok:
            return [paths[i] for i in idxs]

    return paths[:6]


def fig3_severe_6_slices(root: Path, out_path: Path, seed: Optional[int] = None) -> None:
    rng = random.Random(seed)
    severe_dir = root / "Severe_SMPP"
    if not severe_dir.exists():
        raise RuntimeError(f"Folder not found: {severe_dir}")

    series_dirs = [p for p in severe_dir.iterdir() if p.is_dir()]
    rng.shuffle(series_dirs)

    chosen_paths: Optional[List[Path]] = None
    chosen_series: Optional[Path] = None
    for sd in series_dirs:
        cand = find_6_consecutive_slices_in_series(sd)
        if cand is not None:
            chosen_paths = cand
            chosen_series = sd
            break

    if chosen_paths is None or chosen_series is None:
        raise RuntimeError("No series folder in Severe_SMPP contains 6 (near-)consecutive PNG slices.")

    imgs = [read_png_grayscale(p) for p in chosen_paths]
    vmins, vmaxs = zip(*(robust_window(im) for im in imgs))
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    fig, axes = plt.subplots(1, 6, figsize=(16.0, 3.2))
    for ax, p, im in zip(axes, chosen_paths, imgs):
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        n = extract_slice_num(p)
        ax.set_title(f"{n:04d}" if n is not None else p.stem, fontsize=10, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    fig.suptitle(
        f"Severe_SMPP: 6 consecutive slices | {chosen_series.name}",
        y=1.03,
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_sequence_grid(
    *,
    series_dir: Path,
    start_slice: int,
    end_slice: int,
    out_path: Path,
) -> None:
    if end_slice < start_slice:
        raise ValueError("end_slice must be >= start_slice")
    if not series_dir.exists():
        raise FileNotFoundError(str(series_dir))

    paths: List[Path] = []
    for i in range(start_slice, end_slice + 1):
        paths.append(series_dir / f"slice_{i:04d}.png")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing slice files:\n" + "\n".join(missing))

    imgs = [read_png_grayscale(p) for p in paths]
    vmins, vmaxs = zip(*(robust_window(im) for im in imgs))
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    base_dpi = 300.0
    max_h = max(int(im.shape[0]) for im in imgs)
    max_w = max(int(im.shape[1]) for im in imgs)
    fig_w = max(10.0, (5.0 * max_w) / base_dpi + 0.8)
    fig_h = max(4.5, (2.0 * max_h) / base_dpi + 0.8)

    fig, axes = plt.subplots(2, 5, figsize=(fig_w, fig_h), dpi=base_dpi)
    fig.patch.set_facecolor("white")

    sans = ["Arial", "DejaVu Sans"]
    label_font = {"family": sans, "weight": "bold", "size": 13}
    idx_font = {"family": sans, "weight": "normal", "size": 9}

    letters = [chr(ord("a") + k) for k in range(len(paths))]

    for ax, im, letter, p in zip(axes.ravel(), imgs, letters, paths):
        ax.set_facecolor("white")
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")

        ax.text(
            0.01,
            0.99,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            fontdict=label_font,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
        )

        s = extract_slice_num(p)
        idx_text = f"#{s}" if s is not None else f"#{p.stem}"
        ax.text(
            0.985,
            0.02,
            idx_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#E6E6E6",
            fontdict=idx_font,
            bbox=dict(facecolor="black", edgecolor="none", pad=1.0, alpha=0.20),
        )

    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_spatial_dimension_flow(
    *,
    series_dir: Path,
    start_slice: int,
    end_slice: int,
    out_path: Path,
    stack_count: int = 4,
) -> None:
    if end_slice < start_slice:
        raise ValueError("end_slice must be >= start_slice")
    if not series_dir.exists():
        raise FileNotFoundError(str(series_dir))

    paths: List[Path] = []
    for i in range(start_slice, end_slice + 1):
        paths.append(series_dir / f"slice_{i:04d}.png")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing slice files:\n" + "\n".join(missing))

    imgs = [read_png_grayscale(p) for p in paths]
    vmins, vmaxs = zip(*(robust_window(im) for im in imgs))
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    base_dpi = 300.0
    max_h = max(int(im.shape[0]) for im in imgs)
    max_w = max(int(im.shape[1]) for im in imgs)

    fig_w = max(12.5, (5.0 * max_w) / base_dpi + 4.5)
    fig_h = max(5.2, (2.0 * max_h) / base_dpi + 1.6)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=base_dpi)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.35, 3.7], wspace=0.02)

    ax_stack = fig.add_subplot(gs[0, 0])
    ax_arrow = fig.add_subplot(gs[0, 1])

    right = gs[0, 2].subgridspec(2, 5, wspace=0.01, hspace=0.01)
    axes_grid = [fig.add_subplot(right[r, c]) for r in range(2) for c in range(5)]

    sans = ["Arial", "DejaVu Sans"]
    label_font = {"family": sans, "weight": "bold", "size": 13}
    idx_font = {"family": sans, "weight": "normal", "size": 9}
    caption_font = {"family": sans, "weight": "bold", "size": 13}

    stack_n = max(3, min(int(stack_count), len(imgs)))
    stack_imgs = imgs[:stack_n]
    offset = int(round(0.06 * max(min(max_w, max_h), 1), 0))
    offset = max(10, min(offset, 26))

    stack_total_w = max_w + (stack_n - 1) * offset
    stack_total_h = max_h + (stack_n - 1) * offset

    ax_stack.set_facecolor("white")
    ax_stack.set_xlim(-0.5, stack_total_w + 0.5)
    ax_stack.set_ylim(stack_total_h + 0.5, -0.5)
    ax_stack.axis("off")

    edge = "#808080"
    for i in range(stack_n):
        im = stack_imgs[stack_n - 1 - i]
        x0 = i * offset
        y0 = i * offset
        ax_stack.imshow(
            im,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(x0, x0 + max_w, y0 + max_h, y0),
        )
        ax_stack.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                max_w,
                max_h,
                linewidth=0.8,
                edgecolor=edge,
                facecolor="none",
            )
        )

    ax_stack.text(
        0.5,
        -0.08,
        "Input CT Sequence (T₁)",
        transform=ax_stack.transAxes,
        ha="center",
        va="top",
        color="black",
        fontdict=caption_font,
    )

    ax_arrow.set_facecolor("white")
    ax_arrow.axis("off")
    ax_arrow.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#1E6BD6", linewidth=6, mutation_scale=28),
    )

    letters = [chr(ord("a") + k) for k in range(len(paths))]
    for ax, im, letter, p in zip(axes_grid, imgs, letters, paths):
        ax.set_facecolor("white")
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")

        ax.text(
            0.01,
            0.99,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            fontdict=label_font,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
        )

        s = extract_slice_num(p)
        idx_text = f"#{s}" if s is not None else f"#{p.stem}"
        ax.text(
            0.99,
            0.02,
            idx_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#B8B8B8",
            fontdict=idx_font,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.55),
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#C0C0C0")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_volume_3d_flow(
    *,
    series_dir: Path,
    start_slice: int,
    end_slice: int,
    out_path: Path,
    stack_count: int = 4,
    elev: float = 25.0,
    azim: float = -35.0,
) -> None:
    if end_slice < start_slice:
        raise ValueError("end_slice must be >= start_slice")
    if not series_dir.exists():
        raise FileNotFoundError(str(series_dir))

    paths: List[Path] = []
    for i in range(start_slice, end_slice + 1):
        paths.append(series_dir / f"slice_{i:04d}.png")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing slice files:\n" + "\n".join(missing))

    imgs = [read_png_grayscale(p) for p in paths]
    vmins, vmaxs = zip(*(robust_window(im) for im in imgs))
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    base_dpi = 300.0
    max_h = max(int(im.shape[0]) for im in imgs)
    max_w = max(int(im.shape[1]) for im in imgs)

    fig_w = max(13.0, (5.0 * max_w) / base_dpi + 5.2)
    fig_h = max(5.6, (2.0 * max_h) / base_dpi + 1.9)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=base_dpi)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.38, 3.7], wspace=0.02)
    ax_stack = fig.add_subplot(gs[0, 0], projection="3d")
    ax_arrow = fig.add_subplot(gs[0, 1])

    right = gs[0, 2].subgridspec(2, 5, wspace=0.01, hspace=0.01)
    axes_grid = [fig.add_subplot(right[r, c]) for r in range(2) for c in range(5)]

    sans = ["Arial", "DejaVu Sans"]
    label_font = {"family": sans, "weight": "bold", "size": 13}
    idx_font = {"family": sans, "weight": "normal", "size": 9}
    caption_font = {"family": sans, "weight": "bold", "size": 13}

    stack_n = max(3, min(int(stack_count), len(imgs)))
    stack_imgs = imgs[:stack_n]

    step = max(1, int(round(max(max_h, max_w) / 180.0)))
    xs = np.arange(0, max_w, step, dtype=np.float32)
    ys = np.arange(0, max_h, step, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    z_gap = float(0.16 * max(max_h, max_w))
    xy_shift = float(0.06 * max(max_h, max_w))
    cmap = plt.get_cmap("gray")
    ls = LightSource(azdeg=315, altdeg=45)
    denom = max(1e-6, (vmax - vmin))

    ax_stack.set_facecolor("white")
    ax_stack.set_proj_type("persp")
    ax_stack.view_init(elev=elev, azim=azim)
    ax_stack.grid(False)

    for axis in (ax_stack.xaxis, ax_stack.yaxis, ax_stack.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass

    ax_stack.set_xticks([])
    ax_stack.set_yticks([])
    ax_stack.set_zticks([])
    ax_stack.set_axis_off()

    edge = "#9A9A9A"
    for i in range(stack_n):
        im = stack_imgs[i]
        im_ds = im[::step, ::step]
        norm = np.clip((im_ds - vmin) / denom, 0.0, 1.0)
        rgb = np.dstack([norm, norm, norm]).astype(np.float64)
        relief = (norm - 0.5).astype(np.float64)
        shaded_rgb = ls.shade_rgb(rgb, relief, fraction=1.0)

        slice_alpha = 0.95 - 0.12 * float(i)
        slice_alpha = float(np.clip(slice_alpha, 0.55, 0.95))
        alpha_mask = 0.10 + 0.90 * np.power(norm, 0.85)
        alpha = (slice_alpha * alpha_mask).astype(np.float64)
        facecolors = np.dstack([shaded_rgb, alpha]).astype(np.float64)
        z = float(i) * z_gap
        x0 = float(i) * xy_shift
        y0 = -float(i) * xy_shift
        z_relief = z + float(0.06 * z_gap) * relief

        ax_stack.plot_surface(
            X + x0,
            Y + y0,
            z_relief,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

        ax_stack.plot(
            [x0, x0 + max_w, x0 + max_w, x0, x0],
            [y0, y0, y0 + max_h, y0 + max_h, y0],
            [z, z, z, z, z],
            color=edge,
            linewidth=1.0,
        )

    x_max = max_w + (stack_n - 1) * xy_shift
    y_min = -float(stack_n - 1) * xy_shift
    y_max = max_h
    z_max = (stack_n - 1) * z_gap
    ax_stack.set_xlim(0.0, float(x_max))
    ax_stack.set_ylim(float(y_min), float(y_max))
    ax_stack.set_zlim(-0.5 * z_gap, float(z_max) + 0.5 * z_gap)
    try:
        ax_stack.set_box_aspect((float(x_max), float(max_h), float(z_max + z_gap)))
    except Exception:
        pass

    ax_stack.text2D(
        0.5,
        -0.08,
        "Input CT Sequence (T₁)",
        transform=ax_stack.transAxes,
        ha="center",
        va="top",
        color="black",
        fontdict=caption_font,
    )

    ax_arrow.set_facecolor("white")
    ax_arrow.axis("off")
    ax_arrow.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#1E6BD6", linewidth=6, mutation_scale=28),
    )

    letters = [chr(ord("a") + k) for k in range(len(paths))]
    for ax, im, letter, p in zip(axes_grid, imgs, letters, paths):
        ax.set_facecolor("white")
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")

        ax.text(
            0.01,
            0.99,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            fontdict=label_font,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
        )

        s = extract_slice_num(p)
        idx_text = f"#{s}" if s is not None else f"#{p.stem}"
        ax.text(
            0.99,
            0.02,
            idx_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#B8B8B8",
            fontdict=idx_font,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.55),
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#C0C0C0")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
def scan_dataset_counts(root: Path):
    info = {}
    for g in GROUPS:
        gdir = root / g
        if not gdir.exists():
            info[g] = {"patients": 0, "pngs": 0}
            continue
        patient_dirs = [p for p in gdir.iterdir() if p.is_dir()]
        png_count = sum(1 for _ in iter_pngs(gdir))
        info[g] = {"patients": len(patient_dirs), "pngs": png_count}
    return info


def draw_box(ax, x: float, y: float, text: str, fc: str = "white", ec: str = "black") -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.4,rounding_size=0.15",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.0,
        ),
    )


def fig4_structure_tree(root: Path, out_path: Path) -> None:
    info = scan_dataset_counts(root)

    fig = plt.figure(figsize=(10.5, 5.8))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    root_x, root_y = 0.5, 0.88
    draw_box(ax, root_x, root_y, f"Dataset root\n{root}", fc="#F5F5F5")

    xs = np.linspace(0.12, 0.88, 4)
    y_group = 0.55
    y_leaf = 0.24

    for x, g in zip(xs, GROUPS):
        draw_box(ax, x, y_group, g, fc="white")
        ax.plot([root_x, x], [root_y - 0.06, y_group + 0.06], color="black", linewidth=1.0)

        patients = info[g]["patients"]
        pngs = info[g]["pngs"]
        draw_box(ax, x, y_leaf, f"{patients} series folders\n{pngs} PNG slices", fc="white")
        ax.plot([x, x], [y_group - 0.06, y_leaf + 0.06], color="black", linewidth=1.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle("Dataset hierarchy overview", y=0.98, fontsize=16, fontweight="semibold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_paper_style()

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Dataset root containing PNG group folders.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--outdir", type=str, default="figures_out")
    p.add_argument("--figure2-final", action="store_true")
    p.add_argument("--figure2-standard", action="store_true")
    p.add_argument("--figure3-seq", action="store_true")
    p.add_argument("--figure3-flow", action="store_true")
    p.add_argument("--figure3-3d-flow", action="store_true")
    p.add_argument("--series-dir", type=str, default="", help="Series folder for sequence/flow figures.")
    p.add_argument("--start-slice", type=int, default=59)
    p.add_argument("--end-slice", type=int, default=68)
    p.add_argument("--stack-count", type=int, default=4)
    p.add_argument("--hh", type=str, default="image_0b0324.png")
    p.add_argument("--normal-mpp", type=str, default="image_0b035c.png")
    p.add_argument("--severe-smpp", type=str, default="image_0b039f.png")
    p.add_argument("--viral", type=str, default="image_0b03de.png")
    args = p.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.figure3_3d_flow:
        if not args.series_dir:
            raise ValueError("--series-dir is required for --figure3-3d-flow")
        series_dir = Path(args.series_dir)
        out = outdir / "Figure_3_Volume_Spatial_Flow.png"
        fig3_volume_3d_flow(
            series_dir=series_dir,
            start_slice=args.start_slice,
            end_slice=args.end_slice,
            out_path=out,
            stack_count=args.stack_count,
        )
        print("Saved:")
        print(out)
        return

    if args.figure3_flow:
        if not args.series_dir:
            raise ValueError("--series-dir is required for --figure3-flow")
        series_dir = Path(args.series_dir)
        out = outdir / "Figure_3_Spatial_Dimension_Flow.png"
        fig3_spatial_dimension_flow(
            series_dir=series_dir,
            start_slice=args.start_slice,
            end_slice=args.end_slice,
            out_path=out,
            stack_count=args.stack_count,
        )
        print("Saved:")
        print(out)
        return

    if args.figure3_seq:
        if not args.series_dir:
            raise ValueError("--series-dir is required for --figure3-seq")
        series_dir = Path(args.series_dir)
        out = outdir / "Figure_3_SMPP_Sequence_Final.png"
        fig3_sequence_grid(
            series_dir=series_dir,
            start_slice=args.start_slice,
            end_slice=args.end_slice,
            out_path=out,
        )
        print("Saved:")
        print(out)
        return

    if args.figure2_final or args.figure2_standard:
        hh_path = Path(args.hh)
        normal_mpp_path = Path(args.normal_mpp)
        severe_smpp_path = Path(args.severe_smpp)
        viral_path = Path(args.viral)

        if not hh_path.is_absolute():
            hh_path = outdir / hh_path
        if not normal_mpp_path.is_absolute():
            normal_mpp_path = outdir / normal_mpp_path
        if not severe_smpp_path.is_absolute():
            severe_smpp_path = outdir / severe_smpp_path
        if not viral_path.is_absolute():
            viral_path = outdir / viral_path

        if args.figure2_standard:
            out = outdir / "Figure_2_Standard_Style.png"
            fig2_standard_style_pathology_comparison(
                hh_path=hh_path,
                normal_mpp_path=normal_mpp_path,
                severe_smpp_path=severe_smpp_path,
                viral_path=viral_path,
                out_path=out,
            )
            print("Saved:")
            print(out)
            return

        out = outdir / "Figure_2_Final.png"
        fig2_final_pathology_comparison(
            hh_path=hh_path,
            normal_mpp_path=normal_mpp_path,
            severe_smpp_path=severe_smpp_path,
            viral_path=viral_path,
            out_path=out,
        )
        print("Saved:")
        print(out)
        return

    fig2_random_2x2(root, outdir / "Fig2_style_random_2x2.png", seed=args.seed)
    fig3_severe_6_slices(root, outdir / "Fig3_style_severe_6_slices.png", seed=args.seed)
    fig4_structure_tree(root, outdir / "Fig4_style_dataset_tree.png")

    print("Saved:")
    print(outdir / "Fig2_style_random_2x2.png")
    print(outdir / "Fig3_style_severe_6_slices.png")
    print(outdir / "Fig4_style_dataset_tree.png")


if __name__ == "__main__":
    main()
