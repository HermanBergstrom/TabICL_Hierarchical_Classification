"""
Unified confusion matrix plotting: taxonomy-ordered labels with per-parent block highlights.

Labels are grouped by their top-level (L1) parent in the order they appear in the
taxonomy, so all mistakes within the same L1 category cluster near the diagonal.
A coloured rectangle is drawn around each L1 block to make cross-parent vs within-parent
errors immediately visible.

Can be imported as a module or run standalone:
    python plot_confusion_matrix.py \\
        --dataset bugs \\
        --true-labels y_true_fold_0.npy \\
        --pred-labels y_pred_fold_0.npy \\
        --output cm_fold_0.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Make the repo importable when this module is loaded from anywhere
_here = Path(__file__).resolve().parent
_repo_root = _here / "dsi-nlp-publib" / "htc-survey-24"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# ---------------------------------------------------------------------------
# Label ordering
# ---------------------------------------------------------------------------

def get_taxonomy_ordered_labels(
    taxonomy: Dict[str, List[str]],
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """
    Walk the taxonomy top-down to produce leaf labels grouped by L1 (root-child) parent.

    Returns:
        ordered_leaves: leaf label strings in taxonomy-grouped order
        blocks:         list of (start_idx, end_inclusive_idx, l1_name) for each L1 group,
                        suitable for drawing block-highlight rectangles on the diagonal
    """
    l1_nodes = taxonomy.get("root", [])

    ordered_leaves: List[str] = []
    blocks: List[Tuple[int, int, str]] = []

    for l1 in l1_nodes:
        start = len(ordered_leaves)
        leaves: List[str] = []
        # BFS: collect all leaf descendants of this L1 node in sorted order
        queue = [l1]
        while queue:
            node = queue.pop(0)
            children = taxonomy.get(node, [])
            for child in sorted(children):
                if child in taxonomy:   # internal node — recurse
                    queue.append(child)
                else:                   # leaf
                    leaves.append(child)
        ordered_leaves.extend(leaves)
        if leaves:
            blocks.append((start, len(ordered_leaves) - 1, l1))

    return ordered_leaves, blocks


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def save_cm_with_taxonomy_blocks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taxonomy: Dict[str, List[str]],
    output_path: Path,
    title: str = "",
    extra_labels: Optional[List[str]] = None,
) -> None:
    """
    Save a confusion matrix PNG where:
      - rows / columns are ordered by L1 taxonomy group (within each group: sorted)
      - a coloured rectangle highlights the diagonal block for each L1 group

    Args:
        y_true:       1-D string array of true labels.
        y_pred:       1-D string array of predicted labels.
        taxonomy:     {parent: [children]} dict from load_taxonomy().
        output_path:  Destination PNG path (parent directory is created if needed).
        title:        Plot title.
        extra_labels: Additional labels appended after the taxonomy-ordered ones
                      (e.g. ["__none__"] when some samples have no prediction).
    """
    ordered_labels, blocks = get_taxonomy_ordered_labels(taxonomy)
    if extra_labels:
        ordered_labels = ordered_labels + list(extra_labels)

    cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)
    n = len(ordered_labels)

    # Figure and font sizing: keep it square, cap at 36 in
    fig_size = max(8.0, min(36.0, 0.4 * n + 2.0))
    fontsize  = max(8, min(14, 400 // n))

    label_fontsize = 30

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    img = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=label_fontsize)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ordered_labels, fontsize=fontsize)
    ax.set_xlabel("Predicted label", fontsize=label_fontsize)
    ax.set_ylabel("True label", fontsize=label_fontsize)

    # Block-highlight rectangles: one per L1 parent, cycling through tab20
    if blocks:
        cmap_blocks = plt.cm.tab20
        for i, (start, end, _l1_name) in enumerate(blocks):
            size = end - start + 1
            color = cmap_blocks(i / max(len(blocks), 1))
            rect = mpatches.Rectangle(
                (start - 0.5, start - 0.5), size, size,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
                clip_on=False,
            )
            ax.add_patch(rect)

    fig.tight_layout(pad=0.3)
    output_path = Path(output_path).with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-block zoomed confusion matrices
# ---------------------------------------------------------------------------

def save_block_cms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taxonomy: Dict[str, List[str]],
    output_dir: Path,
    title_prefix: str = "",
    extra_labels: Optional[List[str]] = None,
) -> None:
    """
    Save one zoomed-in confusion matrix PNG per L1 block, with count annotations
    for cells with value >= 10.

    Files are written to {output_dir}/blocks/{l1_name}.png.

    Args:
        y_true:       1-D string array of true labels.
        y_pred:       1-D string array of predicted labels.
        taxonomy:     {parent: [children]} dict from load_taxonomy().
        output_dir:   Parent directory (same as the full CM output directory).
        title_prefix: Prepended to each block's plot title.
        extra_labels: Same extra_labels passed to save_cm_with_taxonomy_blocks;
                      they are excluded from block CMs (no L1 parent).
    """
    ordered_labels, blocks = get_taxonomy_ordered_labels(taxonomy)
    if extra_labels:
        ordered_labels = ordered_labels + list(extra_labels)

    cm_full = confusion_matrix(y_true, y_pred, labels=ordered_labels)

    blocks_dir = Path(output_dir) / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    cmap_blocks = plt.cm.tab20

    for i, (start, end, l1_name) in enumerate(blocks):
        block_labels = ordered_labels[start : end + 1]
        n = len(block_labels)
        cm_block = cm_full[start : end + 1, start : end + 1]

        # Strip the L1 prefix (e.g. "ACPI_BIOS" → "BIOS") for tighter tick labels
        prefix = l1_name + "_"
        display_labels = [
            lbl[len(prefix):] if lbl.startswith(prefix) else lbl
            for lbl in block_labels
        ]

        # Scale figure and fonts with block size (larger block → more space → bigger font)
        fig_size     = max(4.0, min(16.0, 0.8 * n + 2.0))
        fontsize     = max(8,  min(16, 7  + n // 2))
        ann_fontsize = max(9,  min(18, 8  + n // 2))
        label_fontsize = max(10, min(18, 9 + n // 2))

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        color = cmap_blocks(i / max(len(blocks), 1))

        img = ax.imshow(cm_block, interpolation="nearest", cmap=plt.cm.Blues)
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=label_fontsize)

        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=fontsize)
        ax.set_yticks(ticks)
        ax.set_yticklabels(display_labels, fontsize=fontsize)
        ax.set_xlabel("Predicted label", fontsize=label_fontsize)
        ax.set_ylabel("True label", fontsize=label_fontsize)
        ax.set_title(l1_name, fontsize=label_fontsize)

        # Highlight the border in the block's tab20 color
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

        # Per-cell count annotations (skip cells with value < 10)
        thresh = float(cm_block.max()) / 2.0 if cm_block.size else 0.0
        for r, c in np.ndindex(cm_block.shape):
            v = int(cm_block[r, c])
            if v < 10:
                continue
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if cm_block[r, c] > thresh else "black",
                    fontsize=ann_fontsize,
                    bbox=dict(pad=0, linewidth=0, facecolor="none"))

        fig.tight_layout(pad=0.3)
        fig.savefig(blocks_dir / f"{l1_name}.pdf", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a taxonomy-ordered confusion matrix from saved .npy prediction arrays.\n"
            "Both model files save predictions when --save-predictions-dir is passed."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["wos", "bugs", "amazon", "bgc"])
    parser.add_argument("--true-labels", required=True,
                        help="Path to .npy file with true leaf-label strings (1-D).")
    parser.add_argument("--pred-labels", required=True,
                        help="Path to .npy file with predicted leaf-label strings (1-D).")
    parser.add_argument("--output",  required=True, help="Output PNG path.")
    parser.add_argument("--title",   default="",   help="Plot title.")
    args = parser.parse_args()

    # chdir so load_taxonomy can find the data files
    os.chdir(str(_repo_root))
    from preprocess_taxonomy import load_taxonomy  # noqa: PLC0415

    taxonomy = load_taxonomy(args.dataset)
    y_true   = np.load(args.true_labels, allow_pickle=True).astype(str)
    y_pred   = np.load(args.pred_labels, allow_pickle=True).astype(str)

    save_cm_with_taxonomy_blocks(y_true, y_pred, taxonomy, Path(args.output), title=args.title)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    _main()
