# Hierarchical TabICL Refactoring Plan

## Overview
Transition the current flat TabICL classifier into a generalized, arbitrary-depth hierarchical classification method. The method will navigate the taxonomy tree top-down. At each node, a dynamically instantiated TabICL classifier will predict the next subcategory using a tailored support set restricted solely to that node's branch, with local Principal Component Analysis (PCA) performed to maximize relevant feature variance.

## Step 1: Offline Taxonomy Preprocessing Script
- Create a standalone Python script (e.g., `preprocess_taxonomy.py`) strictly outside of `dsi-nlp-publib/htc-survey-24` to avoid modifying the benchmark repository.
- Reuse standard taxonomy parsing methods already demonstrated in the benchmark (e.g., parsing `.txt` files using `networkx.read_adjlist` generally, or `read_edgelist` for particular datasets like BGC).
- Map leaf labels (present in the cached embeddings) backward through all their ancestors up to the semantic root.

## Step 2: Precomputing Node-Specific PCAs & Support Subsets
- Instead of computing filtering and PCA dynamically at runtime, compute them offline in the preprocessing script:
  1. For every node in the taxonomy tree, collect all cached training embeddings belonging to its children and map them to their immediate subcategory labels.
  2. Implement subsampling for each node's support set to ensure it fits within the TabICL transformer's maximum sequence length and memory limits.
  3. Fit a localized Principal Component Analysis (PCA) model utilizing *only* this node-specific support set to maximize feature variance capable of distinguishing those specific subcategories.
- Serialize and save each node's filtered support set, its corresponding PCA model parameters, and its immediate child label mappings to a structured directory (e.g., `embeddings_cache/node_data/`).

## Step 3: Lazy-Loading Predictor Initialization
- Avoid instantiating or loading all possible TabICL classifiers and data subsets upfront to conserve memory.
- Implement a lazy-loading factory in the inference loop: when prediction reaches a specific Category Node for the first time, dynamically initialize a new TabICL classifier for it.
- During initialization, merely load the pre-computed:
  1. Node-specific support set.
  2. Fitted node-specific PCA parameters.

## Step 4: Recursive Inference Pipeline
- Implement a recursive or iterative `predict` method that traverses the taxonomy tree:
  1. **Start at Root:** Query the root-level classifier.
  2. **Predict & Route:** Receive the predicted subcategory.
  3. **Early Evaluation Stopping (Optional):** If true labels are available (e.g., during testing/evaluation), check if the predicted category is correct. If it is wrong at the current level, **halt inference** for this sample immediately to save compute, as all downstream predictions will inherently be incorrect.
  4. **Traverse:** Load/initialize the classifier for the predicted subcategory.
  5. **Repeat:** Repeat the prediction on the subcategory classifier until a leaf node (final fine-grained label) is reached.

## Step 5: Integration, Fallbacks & Evaluation
- Implement robust multi-level fallback behavior (e.g., if a classifier fails to predict a valid child due to missing support, halt at the current node or map to a default backing label).
- Ensure execution state doesn't memory leak through excessive loaded classifiers by limiting the cache size of active TabICL instances if necessary.
- Update evaluation logic to track **per-level accuracy**. By measuring accuracy at each individual depth, we can pinpoint at which level traversing the taxonomy introduces the most errors.
- Ensure that if a sample is early-stopped (due to a wrong upper-level prediction), it is automatically counted as an incorrect prediction for the current level and all subsequent lower levels.