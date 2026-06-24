"""
Evaluation module for Matrix Factorization Router.
Computes all metrics from the LLMRouterBench paper.
"""

import numpy as np
import torch
from collections import defaultdict

import config


# ==========================================================================
# Paper Metrics
# ==========================================================================

def compute_dataset_accuracy(selected_model_ids, score_matrix, available_mask=None):
    """
    Compute accuracy for a routing method on a single dataset.

    Args:
        selected_model_ids: (N,) — index of selected model per query
        score_matrix: (N, M) — ground-truth scores
        available_mask: (N, M) — boolean, True if model is available

    Returns:
        accuracy: float
    """
    N = len(selected_model_ids)
    correct = 0
    for i in range(N):
        mid = selected_model_ids[i]
        if available_mask is not None and not available_mask[i, mid]:
            continue
        correct += score_matrix[i, mid]
    return correct / N if N > 0 else 0.0


def compute_random_accuracy(score_matrix, available_mask=None):
    """Random router: average score across all available models per query."""
    if available_mask is None:
        return score_matrix.mean()
    masked = np.where(available_mask, score_matrix, np.nan)
    return float(np.nanmean(masked))


def compute_best_single_accuracy(score_matrix, available_mask=None):
    """Best Single: model with highest average accuracy across the dataset."""
    if available_mask is None:
        model_accs = score_matrix.mean(axis=0)
    else:
        masked = np.where(available_mask, score_matrix, np.nan)
        model_accs = np.nanmean(masked, axis=0)
    best_model_idx = np.nanargmax(model_accs)
    return float(model_accs[best_model_idx]), int(best_model_idx)


def compute_oracle_accuracy(score_matrix, available_mask=None):
    """Oracle: selects best model per query (upper bound)."""
    if available_mask is None:
        return float(score_matrix.max(axis=1).mean())
    masked = np.where(available_mask, score_matrix, -np.inf)
    return float(masked.max(axis=1).mean())


def compute_all_metrics(router_acc_by_dataset, baseline_accs_by_dataset):
    """
    Compute all paper metrics from per-dataset accuracies.

    Args:
        router_acc_by_dataset: dict[dataset_name -> float]
        baseline_accs_by_dataset: dict[dataset_name -> {
            'random': float, 'best_single': float, 'oracle': float
        }]

    Returns:
        metrics: dict with AvgAcc, Gain@R, Gain@B, Gap@O
    """
    datasets = sorted(router_acc_by_dataset.keys())
    D = len(datasets)

    avg_acc = np.mean([router_acc_by_dataset[d] for d in datasets])

    # Gain@R = (1/|D|) * sum( Acc(a,d)/Acc(R,d) - 1 )
    gain_r = np.mean([
        router_acc_by_dataset[d] / max(baseline_accs_by_dataset[d]["random"], 1e-8) - 1
        for d in datasets
    ])

    # Gain@B = (1/|D|) * sum( Acc(a,d)/Acc(B,d) - 1 )
    gain_b = np.mean([
        router_acc_by_dataset[d] / max(baseline_accs_by_dataset[d]["best_single"], 1e-8) - 1
        for d in datasets
    ])

    # Gap@O = (1/|D|) * sum( 1 - Acc(a,d)/Acc(O,d) )
    gap_o = np.mean([
        1 - router_acc_by_dataset[d] / max(baseline_accs_by_dataset[d]["oracle"], 1e-8)
        for d in datasets
    ])

    return {
        "AvgAcc": avg_acc,
        "Gain@R": gain_r,
        "Gain@B": gain_b,
        "Gap@O": gap_o,
    }


# ==========================================================================
# Evaluation Functions
# ==========================================================================

def evaluate_binary_router(model, test_data, all_data, threshold=0.5, device=None):
    """
    Full evaluation of binary router with per-dataset metrics.

    Returns:
        results: dict with overall and per-dataset metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        emb = test_data["embeddings"].to(device)
        decisions = model.route(emb, threshold=threshold).cpu().numpy()

    labels = test_data["labels"].numpy()
    strong_scores = test_data["strong_scores"]
    weak_scores = test_data["weak_scores"]
    dataset_ids = test_data["dataset_ids"]

    # Compute routing accuracy
    routing_acc = (decisions == labels).mean()

    # Compute actual performance: what score does the routed model achieve?
    routed_scores = np.where(decisions == 1, strong_scores, weak_scores)
    actual_acc = routed_scores.mean()

    # Per-dataset breakdown
    ds_results = defaultdict(lambda: {"routed_scores": [], "strong_scores": [],
                                      "weak_scores": [], "decisions": []})
    for i in range(len(decisions)):
        ds = dataset_ids[i]
        ds_results[ds]["routed_scores"].append(routed_scores[i])
        ds_results[ds]["strong_scores"].append(strong_scores[i])
        ds_results[ds]["weak_scores"].append(weak_scores[i])
        ds_results[ds]["decisions"].append(decisions[i])

    per_dataset = {}
    baseline_accs = {}
    router_accs = {}
    for ds in sorted(ds_results.keys()):
        r = ds_results[ds]
        ds_routed = np.mean(r["routed_scores"])
        ds_strong = np.mean(r["strong_scores"])
        ds_weak = np.mean(r["weak_scores"])
        ds_random = (ds_strong + ds_weak) / 2
        ds_oracle = np.mean(np.maximum(r["strong_scores"], r["weak_scores"]))

        per_dataset[ds] = {
            "router_acc": ds_routed,
            "strong_acc": ds_strong,
            "weak_acc": ds_weak,
            "random_acc": ds_random,
            "oracle_acc": ds_oracle,
            "pct_strong": np.mean(r["decisions"]),
            "num_queries": len(r["decisions"]),
        }
        router_accs[ds] = ds_routed
        baseline_accs[ds] = {
            "random": ds_random,
            "best_single": max(ds_strong, ds_weak),
            "oracle": ds_oracle,
        }

    metrics = compute_all_metrics(router_accs, baseline_accs)

    return {
        "routing_accuracy": routing_acc,
        "actual_accuracy": actual_acc,
        "metrics": metrics,
        "per_dataset": per_dataset,
        "threshold": threshold,
    }


def evaluate_multimodel_router(model, test_data, models, all_data=None, device=None):
    """
    Full evaluation of multi-model router with per-dataset metrics.

    Returns:
        results: dict with overall and per-dataset metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        emb = test_data["embeddings"].to(device)
        scores = test_data["scores"]
        available_mask = (scores >= 0)

        pred_scores = model(emb).cpu()
        # Mask unavailable models
        pred_scores_masked = pred_scores.masked_fill(
            ~torch.tensor(available_mask), -float("inf")
        )
        selected = pred_scores_masked.argmax(dim=-1).numpy()

    best_ids = test_data["best_model_ids"].numpy()
    gt_scores = scores.numpy()
    dataset_ids = test_data["dataset_ids"]

    # Best-model selection accuracy
    selection_acc = (selected == best_ids).mean()

    # Actual performance: score achieved by selected model
    routed_scores = np.array([
        gt_scores[i, selected[i]] if available_mask[i, selected[i]] else 0.0
        for i in range(len(selected))
    ])
    actual_acc = routed_scores.mean()

    # Per-dataset breakdown
    ds_groups = defaultdict(list)
    for i, ds in enumerate(dataset_ids):
        ds_groups[ds].append(i)

    per_dataset = {}
    router_accs = {}
    baseline_accs = {}

    for ds in sorted(ds_groups.keys()):
        indices = ds_groups[ds]
        ds_scores = gt_scores[indices]
        ds_avail = available_mask[indices] if available_mask is not None else None
        ds_selected = selected[indices]
        ds_routed = np.array([
            ds_scores[j, ds_selected[j]] if ds_avail[j, ds_selected[j]] else 0.0
            for j in range(len(indices))
        ])

        ds_router_acc = ds_routed.mean()
        ds_random = compute_random_accuracy(ds_scores, ds_avail)
        ds_best_single, ds_best_idx = compute_best_single_accuracy(ds_scores, ds_avail)
        ds_oracle = compute_oracle_accuracy(ds_scores, ds_avail)

        per_dataset[ds] = {
            "router_acc": ds_router_acc,
            "random_acc": ds_random,
            "best_single_acc": ds_best_single,
            "best_single_model": models[ds_best_idx] if ds_best_idx < len(models) else "?",
            "oracle_acc": ds_oracle,
            "num_queries": len(indices),
        }
        router_accs[ds] = ds_router_acc
        baseline_accs[ds] = {
            "random": ds_random,
            "best_single": ds_best_single,
            "oracle": ds_oracle,
        }

    metrics = compute_all_metrics(router_accs, baseline_accs)

    return {
        "selection_accuracy": selection_acc,
        "actual_accuracy": actual_acc,
        "metrics": metrics,
        "per_dataset": per_dataset,
    }


# ==========================================================================
# Pretty Printing
# ==========================================================================

def print_binary_results(results, strong_name, weak_name):
    """Print binary router evaluation results."""
    print("\n" + "=" * 70)
    print(f"BINARY MF ROUTER RESULTS  (strong={strong_name}, weak={weak_name})")
    print("=" * 70)

    m = results["metrics"]
    print(f"\n  Overall Metrics:")
    print(f"    AvgAcc:  {m['AvgAcc']:.4f}")
    print(f"    Gain@R:  {m['Gain@R']:+.4f}")
    print(f"    Gain@B:  {m['Gain@B']:+.4f}")
    print(f"    Gap@O:   {m['Gap@O']:.4f}")
    print(f"    Routing Accuracy: {results['routing_accuracy']:.4f}")
    print(f"    Threshold: {results['threshold']}")

    print(f"\n  Per-Dataset Breakdown:")
    print(f"  {'Dataset':<20} {'Router':>8} {'Strong':>8} {'Weak':>8} "
          f"{'Random':>8} {'Oracle':>8} {'%Strong':>8} {'N':>6}")
    print("  " + "-" * 86)
    for ds, r in sorted(results["per_dataset"].items()):
        print(f"  {ds:<20} {r['router_acc']:>8.4f} {r['strong_acc']:>8.4f} "
              f"{r['weak_acc']:>8.4f} {r['random_acc']:>8.4f} "
              f"{r['oracle_acc']:>8.4f} {r['pct_strong']:>7.1%} {r['num_queries']:>6}")


def print_multimodel_results(results, models):
    """Print multi-model router evaluation results."""
    print("\n" + "=" * 70)
    print("MULTI-MODEL MF ROUTER RESULTS")
    print("=" * 70)

    m = results["metrics"]
    print(f"\n  Overall Metrics:")
    print(f"    AvgAcc:  {m['AvgAcc']:.4f}")
    print(f"    Gain@R:  {m['Gain@R']:+.4f}")
    print(f"    Gain@B:  {m['Gain@B']:+.4f}")
    print(f"    Gap@O:   {m['Gap@O']:.4f}")
    print(f"    Selection Accuracy (best model): {results['selection_accuracy']:.4f}")

    print(f"\n  Per-Dataset Breakdown:")
    print(f"  {'Dataset':<20} {'Router':>8} {'Random':>8} {'BestSgl':>8} "
          f"{'Oracle':>8} {'BestModel':<30} {'N':>6}")
    print("  " + "-" * 100)
    for ds, r in sorted(results["per_dataset"].items()):
        print(f"  {ds:<20} {r['router_acc']:>8.4f} {r['random_acc']:>8.4f} "
              f"{r['best_single_acc']:>8.4f} {r['oracle_acc']:>8.4f} "
              f"{r['best_single_model']:<30} {r['num_queries']:>6}")
