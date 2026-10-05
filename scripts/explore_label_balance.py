"""
Compute per-node label balance for datasets used by hierarchical_tabicl_model.py.

Usage:
  python scripts/explore_label_balance.py --dataset wos
  python scripts/explore_label_balance.py --dataset bgc --csv out.csv

Outputs a per-node breakdown of how many samples at that node have each
immediate child label (counts and percentages). Optionally writes a CSV.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import os
import sys
import csv

# Resolve the repo root (dsi-nlp-publib/htc-survey-24) by searching upward
repo_root = None
base = Path(__file__).resolve().parent
for _ in range(5):
    candidate = base / "dsi-nlp-publib" / "htc-survey-24"
    if candidate.exists():
        repo_root = candidate
        break
    base = base.parent
if repo_root is None:
    raise FileNotFoundError(
        "Could not locate dsi-nlp-publib/htc-survey-24."
        " Run this script from within the htc_survey workspace or ensure the repository layout matches expectations."
    )

top_level = Path(__file__).resolve().parents[1]
# Ensure imports work whether preprocess_taxonomy.py lives at the top level
# (htc_survey/preprocess_taxonomy.py) or inside the dsi-nlp-publib tree.
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(top_level))
os.chdir(top_level)

from preprocess_taxonomy import load_full_labels, load_taxonomy, build_parent_map, get_immediate_children_at_node, TAX_PATHS


def _compute_node_depths(taxonomy):
    depths = {"root": 0}
    queue = ["root"]
    while queue:
        node = queue.pop(0)
        for child in taxonomy.get(node, []):
            depths[child] = depths[node] + 1
            if child in taxonomy:
                queue.append(child)
    return depths


def compute_balance(dataset_name: str):
    taxonomy = load_taxonomy(dataset_name)
    parent_map = build_parent_map(taxonomy)
    full_labels = load_full_labels(dataset_name)

    n_samples = len(full_labels)

    # For each node -> counts of samples that have each immediate child
    node_child_counts = defaultdict(lambda: defaultdict(int))
    node_total_samples = defaultdict(int)

    nodes = sorted(list(taxonomy.keys()))
    for idx in range(n_samples):
        sample_path = full_labels[idx]
        # For every node, check its immediate children for this sample
        for node in nodes:
            true_children = get_immediate_children_at_node(sample_path, node, parent_map)
            if true_children:
                node_total_samples[node] += 1
                for child in true_children:
                    node_child_counts[node][child] += 1

    node_depths = _compute_node_depths(taxonomy)

    # Prepare a sorted list of nodes to report (by depth, then name)
    report_nodes = sorted(node_total_samples.keys(), key=lambda n: (node_depths.get(n, -1), n))

    rows = []
    for node in report_nodes:
        total = node_total_samples[node]
        child_counts = node_child_counts[node]
        # sort children by descending count
        sorted_children = sorted(child_counts.items(), key=lambda x: -x[1])
        distribution = [(child, cnt, cnt / total) for child, cnt in sorted_children]
        rows.append((node, node_depths.get(node, -1), total, distribution))

    return rows


def print_report(rows):
    for node, depth, total, distribution in rows:
        print(f"Node: {node}  depth={depth}  samples_at_node={total}")
        for child, cnt, frac in distribution:
            print(f"  - {child}: {cnt} ({frac:.2%})")
        print()


def write_csv(rows, csv_path: str):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "depth", "samples_at_node", "child", "count", "fraction"])
        for node, depth, total, distribution in rows:
            for child, cnt, frac in distribution:
                writer.writerow([node, depth, total, child, cnt, f"{frac:.6f}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["wos", "bugs", "amazon", "bgc"])    
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--min-samples", type=int, default=1, help="Only show nodes with at least this many samples")
    args = parser.parse_args()

    rows = compute_balance(args.dataset)
    # filter
    rows = [r for r in rows if r[2] >= args.min_samples]

    print_report(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
