"""
Training pipeline for Matrix Factorization Router.
Supports both binary and multi-model training.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from model import MFRouterBinary, MFRouterMulti


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==========================================================================
# Binary Router Training
# ==========================================================================

def train_binary_router(train_data, test_data, hidden_dim=None, num_epochs=None,
                        lr=None, batch_size=None, device=None, verbose=True):
    """
    Train the binary MF router (RouteLLM-style).

    Args:
        train_data: dict with 'embeddings' (N, D) and 'labels' (N,)
        test_data: dict with 'embeddings' and 'labels'

    Returns:
        model: trained MFRouterBinary
        history: dict with training metrics per epoch
    """
    hidden_dim = hidden_dim or config.HIDDEN_DIM
    num_epochs = num_epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE
    batch_size = batch_size or config.BATCH_SIZE
    device = device or get_device()

    embed_dim = train_data["embeddings"].shape[1]

    # Create model
    model = MFRouterBinary(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    if verbose:
        print(f"\nBinary MF Router: {model.get_num_params():,} parameters")
        print(f"  embed_dim={embed_dim}, hidden_dim={hidden_dim}")
        print(f"  device={device}, epochs={num_epochs}, lr={lr}, batch_size={batch_size}")

    # DataLoader
    train_dataset = TensorDataset(train_data["embeddings"], train_data["labels"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=False)

    # Training loop
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for emb_batch, label_batch in train_loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            logits = model(emb_batch, return_logit=True)
            loss = criterion(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label_batch)
            preds = (logits > 0).float()
            correct += (preds == label_batch).sum().item()
            total += len(label_batch)

        train_loss = total_loss / total
        train_acc = correct / total

        # Evaluate on test set
        test_loss, test_acc = evaluate_binary(model, test_data, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}"
                  f"{'*' if test_acc >= best_test_acc else ''}")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n  Best test accuracy: {best_test_acc:.4f}")

    return model, history


def evaluate_binary(model, data, criterion, device):
    """Evaluate binary router on a dataset."""
    model.eval()
    with torch.no_grad():
        emb = data["embeddings"].to(device)
        labels = data["labels"].to(device)
        logits = model(emb, return_logit=True)
        loss = criterion(logits, labels).item()
        preds = (logits > 0).float()
        acc = (preds == labels).float().mean().item()
    return loss, acc


# ==========================================================================
# Multi-Model Router Training
# ==========================================================================

def train_multimodel_router(train_data, test_data, num_models, hidden_dim=None,
                            num_epochs=None, lr=None, batch_size=None,
                            device=None, verbose=True):
    """
    Train the multi-model MF router.

    Uses a combination of:
    1. BCE loss on per-model score predictions
    2. Cross-entropy loss on best-model selection

    Args:
        train_data: dict with 'embeddings', 'scores', 'best_model_ids'
        test_data: same structure
        num_models: number of candidate models

    Returns:
        model: trained MFRouterMulti
        history: dict with training metrics
    """
    hidden_dim = hidden_dim or config.HIDDEN_DIM
    num_epochs = num_epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE
    batch_size = batch_size or config.BATCH_SIZE
    device = device or get_device()

    embed_dim = train_data["embeddings"].shape[1]

    model = MFRouterMulti(
        embed_dim=embed_dim, num_models=num_models, hidden_dim=hidden_dim
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    if verbose:
        print(f"\nMulti-Model MF Router: {model.get_num_params():,} parameters")
        print(f"  embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
              f"num_models={num_models}")
        print(f"  device={device}, epochs={num_epochs}, lr={lr}, batch_size={batch_size}")

    # DataLoader
    train_dataset = TensorDataset(
        train_data["embeddings"],
        train_data["scores"],
        train_data["best_model_ids"],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0
    best_state = None

    bce_loss_fn = nn.BCELoss(reduction="none")
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for emb_batch, score_batch, best_ids_batch in train_loader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            best_ids_batch = best_ids_batch.to(device)

            # Forward pass: predict scores for all models
            pred_scores = model(emb_batch)  # (B, num_models)

            # Mask for available models (score >= 0)
            available_mask = (score_batch >= 0).float()

            # BCE loss on per-model score predictions (only for available models)
            target_scores = torch.clamp(score_batch, 0, 1)
            bce = bce_loss_fn(pred_scores, target_scores)
            bce = (bce * available_mask).sum() / (available_mask.sum() + 1e-8)

            # Cross-entropy loss on best model selection
            # Use logits (before sigmoid) for CE loss
            logits_for_ce = pred_scores.log() - (1 - pred_scores).log()  # inverse sigmoid
            logits_for_ce = logits_for_ce.masked_fill(available_mask == 0, -1e9)
            ce = ce_loss_fn(logits_for_ce, best_ids_batch)

            # Combined loss
            loss = bce + 0.5 * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(emb_batch)

            # Accuracy: did router pick the best model?
            masked_preds = pred_scores.masked_fill(available_mask == 0, -float("inf"))
            selected = masked_preds.argmax(dim=-1)
            correct += (selected == best_ids_batch).sum().item()
            total += len(emb_batch)

        train_loss = total_loss / total
        train_acc = correct / total

        # Evaluate
        test_loss, test_acc = evaluate_multimodel(
            model, test_data, bce_loss_fn, ce_loss_fn, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}"
                  f"{'*' if test_acc >= best_test_acc else ''}")

    if best_state:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n  Best test accuracy (best-model selection): {best_test_acc:.4f}")

    return model, history


def evaluate_multimodel(model, data, bce_loss_fn, ce_loss_fn, device):
    """Evaluate multi-model router."""
    model.eval()
    with torch.no_grad():
        emb = data["embeddings"].to(device)
        scores = data["scores"].to(device)
        best_ids = data["best_model_ids"].to(device)

        pred = model(emb)
        available_mask = (scores >= 0).float()

        target = torch.clamp(scores, 0, 1)
        bce = bce_loss_fn(pred, target)
        bce = (bce * available_mask).sum() / (available_mask.sum() + 1e-8)

        logits = pred.log() - (1 - pred + 1e-8).log()
        logits = logits.masked_fill(available_mask == 0, -1e9)
        ce = ce_loss_fn(logits, best_ids)

        loss = (bce + 0.5 * ce).item()

        masked_pred = pred.masked_fill(available_mask == 0, -float("inf"))
        selected = masked_pred.argmax(dim=-1)
        acc = (selected == best_ids).float().mean().item()

    return loss, acc


def save_checkpoint(model, history, path, extra=None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "model_class": model.__class__.__name__,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, model):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
