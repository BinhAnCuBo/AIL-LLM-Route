"""
Main entry point for training and evaluating Matrix Factorization Routers.
Runs both Binary (Performance-Cost) and Multi-Model (Performance) settings.
"""

import os
import argparse
import torch

import config
from preprocess import preprocess_performance_setting, preprocess_cost_setting
from train import train_binary_router, train_multimodel_router, save_checkpoint
from evaluate import evaluate_binary_router, evaluate_multimodel_router, print_binary_results, print_multimodel_results

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def run_binary_routing(seed):
    print("\n" + "=" * 80)
    print(f"RUNNING BINARY ROUTING (PERFORMANCE-COST SETTING) - SEED {seed}")
    print("=" * 80)

    device = get_device()
    
    # 1. Preprocess Data
    data_dict = preprocess_cost_setting(seed=seed)
    
    if not data_dict["binary"]["train"]["embeddings"].numel():
        print("No binary training data available. Skipping.")
        return

    train_data = data_dict["binary"]["train"]
    test_data = data_dict["binary"]["test"]
    strong = data_dict["binary"]["strong"]
    weak = data_dict["binary"]["weak"]

    # 2. Train Model
    print(f"\nTraining binary router ({strong} vs {weak})...")
    model, history = train_binary_router(
        train_data, test_data,
        hidden_dim=config.HIDDEN_DIM,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        device=device,
        verbose=True
    )

    # 3. Save Checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"mf_binary_seed{seed}.pt")
    save_checkpoint(model, history, ckpt_path, extra={"strong": strong, "weak": weak})

    # 4. Evaluate
    print("\nEvaluating binary router...")
    results = evaluate_binary_router(model, test_data, data_dict["all_data"], device=device)
    print_binary_results(results, strong, weak)
    
    return results

def run_multimodel_routing(seed):
    print("\n" + "=" * 80)
    print(f"RUNNING MULTI-MODEL ROUTING (PERFORMANCE-ORIENTED SETTING) - SEED {seed}")
    print("=" * 80)

    device = get_device()
    
    # 1. Preprocess Data
    train_data, test_data, models, all_data = preprocess_performance_setting(seed=seed)
    
    if not train_data["embeddings"].numel():
        print("No multi-model training data available. Skipping.")
        return

    # 2. Train Model
    print(f"\nTraining multi-model router ({len(models)} models)...")
    model, history = train_multimodel_router(
        train_data, test_data,
        num_models=len(models),
        hidden_dim=config.HIDDEN_DIM,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        device=device,
        verbose=True
    )

    # 3. Save Checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"mf_multi_seed{seed}.pt")
    save_checkpoint(model, history, ckpt_path, extra={"models": models})

    # 4. Evaluate
    print("\nEvaluating multi-model router...")
    results = evaluate_multimodel_router(model, test_data, models, all_data, device=device)
    print_multimodel_results(results, models)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate MF Routers")
    parser.add_argument("--mode", type=str, choices=["binary", "multi", "both"], default="both",
                        help="Which routing paradigm to run")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Using Device: {get_device()}")
    
    if args.mode in ["binary", "both"]:
        run_binary_routing(args.seed)
        
    if args.mode in ["multi", "both"]:
        run_multimodel_routing(args.seed)

if __name__ == "__main__":
    main()
