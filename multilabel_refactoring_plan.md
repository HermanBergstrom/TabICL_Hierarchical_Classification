# Multi-Label HTC Refactoring Plan

## Problem Statement

The current `TabICLHierarchical` implementation routes each sample down a single path
(one predicted child per node) and evaluates with single-label accuracy. This is
incorrect for BGC, where samples can have multiple leaf labels from different branches.
The BERT baseline uses independent sigmoid outputs (one per class) with a 0.5 threshold,
which is equivalent to Binary Relevance (one-vs-rest). We need to match this.

---

## Approach: Per-Node Auto-Detection

The classifier type used at each node is **determined automatically from the precomputed
support data**, not from a dataset-level flag. At any internal node:

- If **every** sample passing through that node has exactly **one** true immediate child
  → use standard **multi-class TabICL** (faster, existing logic)
- If **any** sample has **more than one** true immediate child at this node
  → use **Binary Relevance** (C independent binary TabICLs, one per child)

This is detected at load time by checking whether any global sample index appears more
than once in `support_indices.npy` (duplicate index = sample maps to multiple children).

```python
counts = np.bincount(support_indices, minlength=len(self.embeddings))
node_is_multilabel = bool((counts[support_indices] > 1).any())
```

Consequences:
- WOS / Bugs / Amazon: every node auto-detects as single-label → zero overhead, existing
  path used throughout
- BGC: most internal nodes will be single-label (each sample has one genre path); only
  nodes where samples genuinely branch to multiple children use binary relevance
- No `--multilabel` flag required; behavior is fully data-driven and correct by default

---

## Approach: Binary Relevance at Multi-Label Nodes

At an internal node where `node_is_multilabel=True`, with `C` candidate children, we fit
**C independent binary TabICL classifiers** — one per child. For child `c`:
- Positive samples: training samples at this node that have child `c`
- Negative samples: all other training samples at this node

At inference, each binary classifier outputs a probability. Samples with prob > 0.5
are routed to that child. If **all children score below 0.5**, the sample is routed
to the single highest-scoring child (avoids dead-end termination). If **multiple
children score above 0.5**, the sample fans out to all of them (correct multilabel
behavior).

---

## Design Decisions (resolved)

| Issue | Decision |
|---|---|
| When to use binary relevance | Auto-detected per node from support data; no user flag |
| Class imbalance (1:C-1 ratio) | Optional `--min-class-balance-ratio` (default None) |
| Computational overhead (×C passes per node) | Accepted; only applies at genuinely multi-label nodes |
| Zero-prediction routing | Fallback to single argmax child |
| Metric alignment | Record all visited nodes per sample; use ancestor extension for hF1 |

---

## Files to Change

### 1. `hierarchical_tabicl_model.py` — major rework

#### 1a. New class: `BinaryRelevanceNodeClassifier`

Add a small container class (before `TabICLHierarchical`) that wraps C binary TabICL
classifiers for one node:

```python
@dataclass-ish class BinaryRelevanceNodeClassifier:
    child_labels: List[str]        # ordered list of child names
    clfs: List[TabICLClassifier]   # one per child (same order)
    pca: object                    # single shared PCA (fitted on full node support)

    def predict_proba(self, X_pca: np.ndarray) -> np.ndarray:
        # Returns (n_samples, n_children) array of P(positive) for each child
        # Each clf.predict_proba gives [[P(neg), P(pos)], ...] — take col 1
        ...

    def route(self, X_pca: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        # Returns list of predicted children per sample.
        # Fallback: if all probs < threshold for a sample, return [argmax child].
        proba = self.predict_proba(X_pca)  # (n, C)
        result = []
        for row in proba:
            chosen = [child_labels[i] for i, p in enumerate(row) if p >= threshold]
            if not chosen:
                chosen = [child_labels[np.argmax(row)]]  # fallback
            result.append(chosen)
        return result
```

For consistency, the existing multi-class path also needs a `route()` interface that
returns `List[List[str]]` (always a single-element list), so `_bfs_pass` can use one
unified calling convention.

#### 1b. Wrap existing multi-class logic: `MultiClassNodeClassifier`

Thin wrapper around the existing `(TabICLClassifier, pca)` tuple, adding `route()`:

```python
class MultiClassNodeClassifier:
    def route(self, X_pca) -> List[List[str]]:
        return [[pred] for pred in self.clf.predict(X_pca)]
```

#### 1c. `TabICLHierarchical.__init__` — new parameter

Add:
- `min_class_balance_ratio: Optional[float] = None` — if set, subsample the majority
  class in binary support sets so the minority makes up at least this fraction

Drop the old `multilabel` flag entirely (detection is automatic now).

#### 1d. `_get_classifier` — auto-dispatch

```python
def _get_classifier(self, node, fold_idx):
    # ... load support_indices, support_y (same as now) ...
    counts = np.bincount(support_indices, minlength=len(self.embeddings))
    node_is_multilabel = bool((counts[support_indices] > 1).any())

    if node_is_multilabel:
        return self._build_binary_relevance_node(support_indices, support_y)
    else:
        return self._build_multiclass_node(support_indices, support_y)  # existing logic
```

`_build_binary_relevance_node(support_indices, support_y)`:
1. Apply `max_support_per_node` subsampling if configured (same as existing)
2. Fit **one shared PCA** on the full support set (same as now)
3. For each unique child label `c` in `support_y`:
   a. Build binary `y_bin`: 1 where `support_y == c`, 0 elsewhere
   b. Apply `_balance_binary_support(X_pca, y_bin, self.min_class_balance_ratio)`
   c. Fit `TabICLClassifier` on `(X_balanced, y_bin_balanced)`
4. Return `BinaryRelevanceNodeClassifier(children, clfs, pca)`

`_balance_binary_support(X, y_bin, ratio)`:
- If `ratio is None`: return X, y_bin unchanged
- Compute `pos_frac = y_bin.sum() / len(y_bin)`
- If `pos_frac < ratio`: subsample negatives so `n_neg = n_pos * (1 - ratio) / ratio`
  (capped at actual n_neg to avoid over-subsampling)
- If `pos_frac > 1 - ratio` (negatives too rare): subsample positives analogously
- Shuffle and return

#### 1e. `_bfs_pass` — unified fan-out routing

Both node classifier types now expose `route()` → `List[List[str]]`, so the BFS loop
becomes unified:

```python
# Works for both MultiClassNodeClassifier and BinaryRelevanceNodeClassifier
routed = clf.route(X_pca)           # List[List[str]], length = n_samples
for idx, predicted_children in zip(indices, routed):
    all_visited[idx].update(predicted_children)
    for child in predicted_children:
        if child in self.taxonomy:
            next_level[child].append(idx)
```

Replace `last_pred: Dict[int, str]` with `all_visited: Dict[int, Set[str]]`.

#### 1f. `evaluate_fold` — new output format

After BFS, build per-sample predicted label lists:

```python
predicted_labels: List[List[str]] = [
    list(all_visited.get(idx, set())) for idx in test_idx
]
true_labels: List[List[str]] = [
    self.full_labels[idx] for idx in test_idx
]
```

Then:
- **hF1 / AHC**: call `compute_hierarchical_metrics(true_labels, predicted_labels, ...)`
  with `encoder_dump_or_mapping=None` (passes label string lists directly; the function
  then calls `_extend_label_set` to add ancestors before computing F1)
- **Flat accuracy & macro F1**: build a `MultiLabelBinarizer` fitted on all taxonomy
  node names, binarize both predicted and true lists, compute `accuracy_score` (exact
  match) and `precision_recall_fscore_support(average='macro')`

#### 1g. `print_summary` — update for hF1 reporting

Add hF1 and AHC to the summary table alongside leaf accuracy.

---

### 2. `hierarchical_tabicl_model.py` — CLI args

Remove `--multilabel`. Add:

```
--min-class-balance-ratio  float, default None; min fraction for positive class
                           in binary support sets at multi-label nodes
                           (e.g. 0.2 = at least 20% positive after subsampling)
```

Example:
```bash
# BGC — binary relevance auto-enabled where needed
python hierarchical_tabicl_model.py --dataset bgc --min-class-balance-ratio 0.2

# WOS — same command works, binary relevance never triggered
python hierarchical_tabicl_model.py --dataset wos
```

---

### 3. `preprocess_taxonomy.py` — no structural changes required

The existing `_build_node_support` already handles BGC multilabel correctly: a single
sample that visits multiple children of a node produces multiple `(index, child)` entries
in `support_y`. The saved `support_indices.npy / support_y.npy` format is fully
compatible with binary relevance — we just re-interpret each unique child label as one
binary classification problem at load time.

---

## Metric Alignment with the Baseline

The baseline (`BERT_flat`) evaluates a `(n_samples, n_all_classes)` binary matrix
where "all classes" includes every leaf label (and implicit parent labels through hF1
ancestor extension). Our hierarchical approach must produce the same format.

| Metric | Baseline computation | Our equivalent |
|---|---|---|
| **Acc** | `accuracy_score(y_true_multihot, y_pred_multihot)` — exact subset match | Binarize `predicted_labels` lists with `MultiLabelBinarizer(classes=all_nodes)`, then same |
| **Macro F1** | `precision_recall_fscore_support(average='macro')` on multihot | Same after binarization |
| **hF1** | `h_multilabel_precision_recall_fscore` after `_extend_label_set` (adds ancestors) | Call `compute_hierarchical_metrics(true_labels_str, pred_labels_str, taxonomy_path, None)` |
| **AHC** | Average shortest-path distance per sample | Same function handles it |

Key: `compute_hierarchical_metrics(..., encoder_dump_or_mapping=None)` accepts
`List[List[str]]` directly — no MultiLabelBinarizer needed for hF1.

---

## Implementation Order

1. **`BinaryRelevanceNodeClassifier` class** — self-contained, testable in isolation
2. **`MultiClassNodeClassifier` wrapper** — thin wrapper, adds `route()` to existing logic
3. **`_balance_binary_support`** — pure function, easy to unit-test
4. **`_get_classifier` auto-dispatch** — plugs both classifiers into existing cache
5. **`_bfs_pass` fan-out routing** — unified `route()` call, `all_visited` dict change
6. **`evaluate_fold` metric computation** — new multi-hot + h-metric output
7. **CLI args** — remove `--multilabel`, add `--min-class-balance-ratio`
8. **`print_summary` update** — cosmetic

Steps 1–4 can be developed and tested (on a small node) without touching BFS.
Steps 5–6 are the integration point.

---

## Edge Cases to Handle

- **Node with only one child class in training set**: current code already skips these
  (`< 2 classes` check). In binary relevance mode a single-child node always predicts
  1 (trivial), so we can skip or route directly without a classifier.
- **Sample already visited child via another branch**: `all_visited` is a `Set`, so
  duplicates are collapsed automatically.
- **Cache key**: both classifier types are cached under the same `(node, fold_idx)` key;
  no cache structure changes needed.

---

## Non-Goals / Out of Scope

- Changing `preprocess_taxonomy.py` data format (fully backward-compatible)
- Modifying the BERT baseline or benchmark repo
- Supporting calibrated thresholds per class (threshold is always 0.5)
- Probabilistic routing beyond the 0.5-or-argmax fallback rule
