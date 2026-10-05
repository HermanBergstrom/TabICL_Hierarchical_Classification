# Baseline Evaluation Metrics in dsi-nlp-publib/htc-survey-24

This document outlines how the baselines from the HTC-survey evaluate multi-label text classification metrics, particularly focusing on how these evaluate parent nodes along with leaf nodes.

### 1. Flat Accuracy (Acc)
The script relies on `sklearn.metrics.accuracy_score` applied directly to the binary thresholded multi-hot encoded arrays of flat labels. 
*   **Mechanism:** For multi-label inputs, scikit-learn defines this as the subset accuracy or "Exact Match Ratio". 
*   **Implication:** A sample is considered correct *if and only if* the complete predicted set of labels stringently matches the entire true set of labels. Partial matches yield an accuracy of 0 for that sample.

### 2. Flat Macro-F1 (F1)
The script uses `sklearn.metrics.precision_recall_fscore_support(average="macro")`.
*   **Mechanism:** Using the thresholded multi-hot predictions (typically > 0.5), it evaluates the F1 score for each possible label independently. It then computes the unweighted arithmetic mean (macro-average) across all labels in the dataset.
*   **Implication:** Does not explicitly comprehend hierarchy. However, because parent labels are typically tracked explicitly as targets alongside leaf labels in this codebase, their individual F1 scores implicitly weigh equally alongside leaf-level scores in the final average.

### 3. Hierarchical Macro-F1 (h-F1)
Calculated via `h_multilabel_precision_recall_fscore` inside `src/utils/metrics.py`.
*   **Mechanism:** 
    1.  **Label Extension:** For both the ground-truth set and the predicted set of labels, the code performs a graph augmentation: for every label present, all of its ancestral nodes (derived via `nx.ancestors` on the taxonomy tree) up to the root are injected into the set.
    2.  **Removal:** The dummy "root" (or "Root") string is cropped from these augmented sets so as not to artificially inflate scores.
    3.  **Scoring:** Standard multi-hot `precision_recall_fscore_support(average="macro")` is then evaluated on these "extended" label sets. 
*   **Implication:** This heavily rewards models that predict deep in the correct sub-lineage, since predicting a single correct leaf implicitly gifts the model multiple correct true positive predictions for the ancestors along the branch. Reaping the reward for the parent layers requires explicitly or implicitly adding them through the augmentation.

### 4. Average Hierarchical Cost (AHC)
Calculated via `compute_hierarchical_metrics` in `src/utils/metrics.py`. This measures the shortest-path discrepancy (in edges) between the predicted categories and the true classes on the taxonomy graph structure.
*   **Mechanism:** 
    1.  For a single evaluation sample, iterate over every label $p$ that the model predicted.
    2.  Calculate the shortest path length on the undirected taxonomy tree from $p$ to the closest true label $t$ in the ground truth set ($d = \min_{t \in T_{sample}} \text{dist}(p, t)$).
    3.  Sum up these minimal edge distances across all predictions for that sample, and divide them by the total count of *true* labels for the sample ($|T_{sample}|$).
    4.  Global AHC is the unweighted average of these per-sample AHC scores across the test set.
*   **Math:** $AHC_{Global} = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{|T_i|} \sum_{p \in P_i} \min_{t \in T_i} \operatorname{dist}(p, t) \right)$
*   **Implication:** Lower is better. A value of 0.0 implies that every prediction exactly matches a ground truth label (distance = 0) with a correct number of predictions. Over-predicting (or missing the mark widely in the graph) generates massive penalties scaled by the graph distance connecting the wrongly predicted neighborhood to the true lineage.

