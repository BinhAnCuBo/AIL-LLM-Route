"""
Data preprocessing for Matrix Factorization Router.
Loads LLMRouterBench data, builds score matrices, generates embeddings, creates splits.
"""

import json
import os
import random
import glob

import numpy as np
import torch
from tqdm import tqdm

import config


def load_dataset_records(dataset_name, split, models):
    """
    Load records for a given dataset and list of models.

    Returns:
        queries: list of (index, origin_query) tuples
        score_matrix: np.ndarray of shape (num_queries, num_models), values in [0, 1]
        cost_matrix: np.ndarray of shape (num_queries, num_models), cost per query
    """
    dataset_dir = os.path.join(config.DATA_DIR, dataset_name, split)
    if not os.path.isdir(dataset_dir):
        print(f"  [WARN] Directory not found: {dataset_dir}")
        return None, None, None

    # Discover which models are available
    if models is None:
        models = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
    available_models = []
    model_data = {}
    for model_name in models:
        model_dir = os.path.join(dataset_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        json_files = glob.glob(os.path.join(model_dir, "*.json"))
        if not json_files:
            continue
        # Load the first (and typically only) JSON file
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        if not records:
            continue
        available_models.append(model_name)
        model_data[model_name] = {r["index"]: r for r in records}

    if not available_models:
        print(f"  [WARN] No model data found for {dataset_name}/{split}")
        return None, None, None

    # Find common query indices across all models
    all_indices = None
    for model_name in available_models:
        indices = set(model_data[model_name].keys())
        if all_indices is None:
            all_indices = indices
        else:
            all_indices = all_indices & indices

    if not all_indices:
        print(f"  [WARN] No common queries for {dataset_name}/{split}")
        return None, None, None

    sorted_indices = sorted(all_indices)

    # Build queries list and score/cost matrices
    queries = []
    score_matrix = np.zeros((len(sorted_indices), len(available_models)), dtype=np.float32)
    cost_matrix = np.zeros((len(sorted_indices), len(available_models)), dtype=np.float32)

    for qi, idx in enumerate(sorted_indices):
        # Use origin_query from first available model
        first_model = available_models[0]
        record = model_data[first_model][idx]
        queries.append((idx, record.get("origin_query", record.get("prompt", ""))))

        for mi, model_name in enumerate(available_models):
            r = model_data[model_name][idx]
            score = r.get("score")
            score_matrix[qi, mi] = float(score) if score is not None else 0.0
            cost = r.get("cost")
            cost_matrix[qi, mi] = float(cost) if cost is not None else 0.0

    return queries, score_matrix, cost_matrix, available_models


def load_all_data(datasets, split, model_pool):
    """
    Load data from all datasets for a given split and model pool.

    Returns:
        all_data: dict mapping dataset_name -> {
            'queries': list of (idx, text),
            'scores': np.ndarray (num_queries x num_models),
            'costs':  np.ndarray (num_queries x num_models),
            'models': list of model names
        }
    """
    all_data = {}
    for ds in tqdm(datasets, desc=f"Loading datasets ({split})"):
        result = load_dataset_records(ds, split, model_pool)
        if result[0] is None:
            continue
        queries, scores, costs, models = result
        all_data[ds] = {
            "queries": queries,
            "scores": scores,
            "costs": costs,
            "models": models,
        }
        print(f"  {ds}: {len(queries)} queries, {len(models)} models, "
              f"avg_score={scores.mean():.3f}")
    return all_data


def generate_embeddings(queries_by_dataset, cache_path=None):
    """
    Generate query embeddings using sentence-transformers.

    Args:
        queries_by_dataset: dict mapping dataset_name -> list of (idx, query_text)
        cache_path: if provided, load/save embeddings from/to this path

    Returns:
        embeddings_by_dataset: dict mapping dataset_name -> np.ndarray (num_queries x embed_dim)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    embeddings_by_dataset = {}
    for ds_name, queries in tqdm(queries_by_dataset.items(), desc="Generating embeddings"):
        texts = [q[1] for q in queries]
        emb = model.encode(texts, show_progress_bar=True, batch_size=64,
                           convert_to_numpy=True)
        embeddings_by_dataset[ds_name] = emb
        print(f"  {ds_name}: {emb.shape}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(embeddings_by_dataset, cache_path)
        print(f"Saved embeddings to {cache_path}")

    return embeddings_by_dataset


def create_train_test_split(num_samples, train_ratio=0.7, seed=42):
    """Create reproducible train/test split indices."""
    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    return indices[:split_point], indices[split_point:]


# ==========================================================================
# Binary Routing Data (RouteLLM-style: strong vs. weak)
# ==========================================================================

def build_binary_routing_data(all_data, embeddings, strong_model, weak_model, seed=42):
    """
    Build training data for binary routing (strong vs. weak model).

    For each query, the label is:
      1 if strong_model scores higher (or equal) than weak_model
      0 if weak_model scores higher

    Returns:
        train_data, test_data: dicts with keys:
            'embeddings': torch.Tensor (N x embed_dim)
            'labels': torch.Tensor (N,) — 1=route to strong, 0=route to weak
            'dataset_ids': list of dataset names per query
            'strong_scores': np.ndarray (N,)
            'weak_scores':   np.ndarray (N,)
    """
    all_embeddings = []
    all_labels = []
    all_ds_ids = []
    all_strong_scores = []
    all_weak_scores = []

    for ds_name, ds_data in all_data.items():
        models = ds_data["models"]
        if strong_model not in models or weak_model not in models:
            continue

        strong_idx = models.index(strong_model)
        weak_idx = models.index(weak_model)
        scores = ds_data["scores"]
        emb = embeddings[ds_name]

        for qi in range(scores.shape[0]):
            s_score = scores[qi, strong_idx]
            w_score = scores[qi, weak_idx]

            # Label: 1 if strong is better or equal, 0 if weak is better
            label = 1.0 if s_score >= w_score else 0.0

            all_embeddings.append(emb[qi])
            all_labels.append(label)
            all_ds_ids.append(ds_name)
            all_strong_scores.append(s_score)
            all_weak_scores.append(w_score)

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    all_strong_scores = np.array(all_strong_scores, dtype=np.float32)
    all_weak_scores = np.array(all_weak_scores, dtype=np.float32)

    # Split
    train_idx, test_idx = create_train_test_split(
        len(all_labels), config.TRAIN_RATIO, seed
    )

    def _subset(indices):
        return {
            "embeddings": torch.tensor(all_embeddings[indices]),
            "labels": torch.tensor(all_labels[indices]),
            "dataset_ids": [all_ds_ids[i] for i in indices],
            "strong_scores": all_strong_scores[indices],
            "weak_scores": all_weak_scores[indices],
        }

    return _subset(train_idx), _subset(test_idx)


# ==========================================================================
# Multi-Model Routing Data
# ==========================================================================

def build_multimodel_routing_data(all_data, embeddings, seed=42):
    """
    Build training data for multi-model routing.

    For each query, the target is the full score vector across all models.
    The router should learn to predict which model scores highest.

    Returns:
        train_data, test_data: dicts with keys:
            'embeddings': torch.Tensor (N x embed_dim)
            'scores': torch.Tensor (N x num_models) — ground-truth scores
            'best_model_ids': torch.LongTensor (N,) — index of best model
            'dataset_ids': list of dataset names per query
        models: list of model names (consistent ordering)
    """
    # First, establish a consistent model ordering from the first dataset
    # that has data, then only use queries where all models have data
    # Find union of available models
    model_set = set()
    for ds_data in all_data.values():
        model_set.update(ds_data["models"])
    models = sorted(model_set)
    model_to_idx = {m: i for i, m in enumerate(models)}

    all_embeddings = []
    all_scores = []
    all_ds_ids = []

    for ds_name, ds_data in all_data.items():
        ds_models = ds_data["models"]
        scores = ds_data["scores"]
        emb = embeddings[ds_name]

        for qi in range(scores.shape[0]):
            score_vec = np.full(len(models), -1.0, dtype=np.float32)  # -1 = unavailable
            for mi, m in enumerate(ds_models):
                if m in model_to_idx:
                    score_vec[model_to_idx[m]] = scores[qi, mi]

            all_embeddings.append(emb[qi])
            all_scores.append(score_vec)
            all_ds_ids.append(ds_name)

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    all_scores = np.array(all_scores, dtype=np.float32)

    # Best model per query (among available models, i.e. score >= 0)
    masked_scores = np.where(all_scores >= 0, all_scores, -np.inf)
    best_model_ids = np.argmax(masked_scores, axis=1).astype(np.int64)

    # Split
    train_idx, test_idx = create_train_test_split(
        len(all_scores), config.TRAIN_RATIO, seed
    )

    def _subset(indices):
        return {
            "embeddings": torch.tensor(all_embeddings[indices]),
            "scores": torch.tensor(all_scores[indices]),
            "best_model_ids": torch.tensor(best_model_ids[indices]),
            "dataset_ids": [all_ds_ids[i] for i in indices],
        }

    return _subset(train_idx), _subset(test_idx), models


# ==========================================================================
# Main preprocessing
# ==========================================================================

def preprocess_performance_setting(seed=42):
    """Full preprocessing pipeline for performance-oriented setting (small models)."""
    print("=" * 60)
    print("PREPROCESSING: Performance-Oriented Setting (Small Models)")
    print("=" * 60)

    all_data = load_all_data(config.PERF_DATASETS, "test", config.SMALL_MODELS)

    # Generate embeddings
    queries_map = {ds: data["queries"] for ds, data in all_data.items()}
    cache_path = os.path.join(config.CACHE_DIR, f"embeddings_perf_seed{seed}.pt")
    embeddings = generate_embeddings(queries_map, cache_path=cache_path)

    # Build multi-model routing data
    train_data, test_data, models = build_multimodel_routing_data(
        all_data, embeddings, seed=seed
    )

    print(f"\nMulti-model routing data:")
    print(f"  Train: {len(train_data['embeddings'])} queries")
    print(f"  Test:  {len(test_data['embeddings'])} queries")
    print(f"  Models: {len(models)}")

    return train_data, test_data, models, all_data


def preprocess_cost_setting(seed=42):
    """Full preprocessing pipeline for performance-cost setting (flagship models)."""
    print("=" * 60)
    print("PREPROCESSING: Performance-Cost Setting (Flagship Models)")
    print("=" * 60)

    # For cost setting, we look in the "hybrid" split
    # Also check "test" as fallback
    all_data = {}
    for ds in tqdm(config.COST_DATASETS, desc="Loading cost datasets"):
        # Try "hybrid" first, then "test"
        for split in ["hybrid", "test"]:
            result = load_dataset_records(ds, split, None)
            if result[0] is not None:
                queries, scores, costs, models = result
                all_data[ds] = {
                    "queries": queries,
                    "scores": scores,
                    "costs": costs,
                    "models": models,
                }
                print(f"  {ds} ({split}): {len(queries)} queries, {len(models)} models")
                break

    if not all_data:
        print("No data found for cost setting. Trying with all available models...")
        # Load with None model pool to discover all models
        for ds in config.COST_DATASETS:
            for split in ["hybrid", "test"]:
                dataset_dir = os.path.join(config.DATA_DIR, ds, split)
                if not os.path.isdir(dataset_dir):
                    continue
                available = [d for d in os.listdir(dataset_dir)
                             if os.path.isdir(os.path.join(dataset_dir, d))]
                result = load_dataset_records(ds, split, available)
                if result[0] is not None:
                    queries, scores, costs, models = result
                    all_data[ds] = {
                        "queries": queries, "scores": scores,
                        "costs": costs, "models": models,
                    }
                    print(f"  {ds} ({split}): {len(queries)} queries, {len(models)} models")
                    break

    # Generate embeddings
    queries_map = {ds: data["queries"] for ds, data in all_data.items()}
    cache_path = os.path.join(config.CACHE_DIR, f"embeddings_cost_seed{seed}.pt")
    embeddings = generate_embeddings(queries_map, cache_path=cache_path)

    # Find strong/weak models
    strong = config.STRONG_MODEL
    weak = config.WEAK_MODEL

    # Check if strong/weak are available; if not, pick best/worst by avg score
    all_models_in_data = set()
    for ds_data in all_data.values():
        all_models_in_data.update(ds_data["models"])

    if strong not in all_models_in_data or weak not in all_models_in_data:
        print(f"\n[INFO] Strong ({strong}) or weak ({weak}) not in data.")
        print(f"  Available models: {sorted(all_models_in_data)}")
        # Pick the best and worst performing models
        model_avg = {}
        for ds_data in all_data.values():
            for mi, m in enumerate(ds_data["models"]):
                if m not in model_avg:
                    model_avg[m] = []
                model_avg[m].append(ds_data["scores"][:, mi].mean())
        model_avg = {m: np.mean(v) for m, v in model_avg.items()}
        sorted_models = sorted(model_avg.items(), key=lambda x: x[1], reverse=True)
        strong = sorted_models[0][0]
        weak = sorted_models[-1][0]
        print(f"  Using strong={strong} (avg={model_avg[strong]:.3f}), "
              f"weak={weak} (avg={model_avg[weak]:.3f})")

    # Build binary routing data
    train_bin, test_bin = build_binary_routing_data(
        all_data, embeddings, strong, weak, seed=seed
    )

    # Build multi-model routing data too
    train_multi, test_multi, models = build_multimodel_routing_data(
        all_data, embeddings, seed=seed
    )

    print(f"\nBinary routing data (strong={strong}, weak={weak}):")
    print(f"  Train: {len(train_bin['embeddings'])} queries")
    print(f"  Test:  {len(test_bin['embeddings'])} queries")
    print(f"  Label balance: {train_bin['labels'].mean():.3f} route-to-strong")

    print(f"\nMulti-model routing data:")
    print(f"  Train: {len(train_multi['embeddings'])} queries")
    print(f"  Test:  {len(test_multi['embeddings'])} queries")
    print(f"  Models: {len(models)}")

    return {
        "binary": {"train": train_bin, "test": test_bin,
                   "strong": strong, "weak": weak},
        "multi": {"train": train_multi, "test": test_multi, "models": models},
        "all_data": all_data,
    }


if __name__ == "__main__":
    print("Running preprocessing...")
    perf_data = preprocess_performance_setting(seed=config.RANDOM_SEED)
    print("\n\n")
    cost_data = preprocess_cost_setting(seed=config.RANDOM_SEED)
    print("\nDone!")
