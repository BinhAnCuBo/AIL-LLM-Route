"""
Matrix Factorization Router Models.
Implements both binary (RouteLLM-style) and multi-model MF routers.
Based on the MF approach described in LLMRouterBench (arXiv:2601.07206).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFRouterBinary(nn.Module):
    """
    Matrix Factorization Router for Binary Routing (RouteLLM-style).

    Routes queries between a strong model and a weak model by predicting
    the probability that the strong model outperforms the weak model.

    Architecture:
        1. Project query embedding into latent space: q = text_proj(query_embed)
        2. Learn model embeddings: P[strong], P[weak]
        3. Compute interaction: h = q * (P[strong] - P[weak])
        4. Classify: score = sigmoid(classifier(h))
        5. Route: if score > threshold → strong model, else → weak model

    This matches the RouteLLM MF approach where model embeddings capture
    model-specific strengths, and the classifier predicts win probability.
    """

    def __init__(self, embed_dim, hidden_dim=128, num_models=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Model embeddings (strong=0, weak=1)
        self.P = nn.Embedding(num_models, hidden_dim)

        # Project text embeddings into latent space
        self.text_proj = nn.Linear(embed_dim, hidden_dim, bias=False)

        # Classifier head
        self.classifier = nn.Linear(hidden_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.normal_(self.P.weight, std=0.01)

    def forward(self, query_embed, return_logit=False):
        """
        Args:
            query_embed: (batch_size, embed_dim) — query text embeddings
            return_logit: if True, return raw logit instead of probability

        Returns:
            score: (batch_size,) — probability of routing to strong model
        """
        # Project query into latent space
        q = self.text_proj(query_embed)  # (B, hidden_dim)

        # Get model embeddings
        strong_emb = self.P(torch.zeros(1, dtype=torch.long,
                                        device=query_embed.device))  # (1, hidden_dim)
        weak_emb = self.P(torch.ones(1, dtype=torch.long,
                                     device=query_embed.device))   # (1, hidden_dim)

        # Interaction: element-wise product with (strong - weak) difference
        h = q * (strong_emb - weak_emb)  # (B, hidden_dim)

        # Classify
        logit = self.classifier(h).squeeze(-1)  # (B,)

        if return_logit:
            return logit
        return torch.sigmoid(logit)

    def route(self, query_embed, threshold=0.5):
        """
        Route decision: returns 1 for strong model, 0 for weak model.
        """
        with torch.no_grad():
            score = self.forward(query_embed)
            return (score >= threshold).long()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MFRouterMulti(nn.Module):
    """
    Matrix Factorization Router for Multi-Model Routing.

    Extended MF approach that scores all candidate models and selects the best.

    Architecture:
        1. Project query embedding: q = text_proj(query_embed)
        2. For each model m: score_m = classifier(q * P[m])
        3. Route to model with highest score: argmax(scores)

    The model learns to predict per-model correctness probability,
    then routes to the model most likely to answer correctly.
    """

    def __init__(self, embed_dim, num_models, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_models = num_models
        self.hidden_dim = hidden_dim

        # Model embeddings
        self.P = nn.Embedding(num_models, hidden_dim)

        # Project text embeddings into latent space
        self.text_proj = nn.Linear(embed_dim, hidden_dim, bias=False)

        # Per-model classifier (shared across models for generalization)
        self.classifier = nn.Linear(hidden_dim, 1, bias=False)

        # Model-specific bias (captures average model quality)
        self.model_bias = nn.Parameter(torch.zeros(num_models))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.normal_(self.P.weight, std=0.01)

    def forward(self, query_embed, model_ids=None):
        """
        Args:
            query_embed: (batch_size, embed_dim)
            model_ids: optional (batch_size,) — if provided, score only these models
                       if None, score all models

        Returns:
            if model_ids is not None:
                scores: (batch_size,) — predicted score for specified models
            else:
                scores: (batch_size, num_models) — predicted scores for all models
        """
        # Project query
        q = self.text_proj(query_embed)  # (B, hidden_dim)

        if model_ids is not None:
            # Score specific models
            m_emb = self.P(model_ids)  # (B, hidden_dim)
            h = q * m_emb  # (B, hidden_dim)
            logit = self.classifier(h).squeeze(-1) + self.model_bias[model_ids]
            return torch.sigmoid(logit)
        else:
            # Score all models
            all_m_emb = self.P.weight  # (num_models, hidden_dim)
            # q: (B, hidden_dim), all_m_emb: (M, hidden_dim)
            # Interaction: (B, M, hidden_dim)
            h = q.unsqueeze(1) * all_m_emb.unsqueeze(0)
            # Classify: (B, M)
            logits = self.classifier(h).squeeze(-1) + self.model_bias.unsqueeze(0)
            return torch.sigmoid(logits)

    def route(self, query_embed, available_mask=None):
        """
        Route to the best model.

        Args:
            query_embed: (batch_size, embed_dim)
            available_mask: optional (batch_size, num_models) boolean mask
                           True = model is available for this query

        Returns:
            selected_model_ids: (batch_size,) — index of selected model
        """
        with torch.no_grad():
            scores = self.forward(query_embed)  # (B, num_models)
            if available_mask is not None:
                scores = scores.masked_fill(~available_mask, -float("inf"))
            return scores.argmax(dim=-1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
