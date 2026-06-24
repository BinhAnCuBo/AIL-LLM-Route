"""
Configuration for Matrix Factorization Router.
Based on RouteLLM MF approach from LLMRouterBench (arXiv:2601.07206).
"""

import os

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "bench-release")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create dirs
for d in [CACHE_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# Embedding Model
# ==============================================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ==============================================================================
# Model Pools (from LLMRouterBench paper)
# ==============================================================================

# Performance-oriented setting: 20 lightweight ~7B LLMs
SMALL_MODELS = [
    "DeepHermes-3-Llama-3-8B-Preview",
    "DeepSeek-R1-0528-Qwen3-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Fin-R1",
    "GLM-Z1-9B-0414",
    "Intern-S1-mini",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1",
    "MiMo-7B-RL-0530",
    "MiniCPM4.1-8B",
    "NVIDIA-Nemotron-Nano-9B-v2",
    "OpenThinker3-7B",
    "Qwen2.5-Coder-7B-Instruct",
    "Qwen3-8B",
    "cogito-v1-preview-llama-8B",
    "gemma-2-9b-it",
    "glm-4-9b-chat",
    "granite-3.3-8b-instruct",
    "internlm3-8b-instruct",
]

# Performance-cost setting: flagship LLMs with cost info
# (strong_model, weak_model) for binary routing
STRONG_MODEL = "gpt-5"
WEAK_MODEL = "qwen3-235b-a22b-thinking"

# Datasets that use the "test" split (performance-oriented, small models)
PERF_DATASETS = [
    "bbh", "emorynlp", "finqa", "gpqa", "humaneval",
    "korbench", "livecodebench", "livemathbench", "math500",
    "mathbench", "mbpp", "medqa", "meld", "mmlupro", "winogrande",
]

# Datasets that use the "hybrid" split (performance-cost, flagship + small)
COST_DATASETS = [
    "aime", "arenahard", "bbh", "gpqa", "hle",
    "humaneval", "korbench", "simpleqa", "swe-bench", "tau2",
]

# ==============================================================================
# Training Hyperparameters
# ==============================================================================
HIDDEN_DIM = 128          # Latent factor dimension (same as RouteLLM paper)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
NUM_EPOCHS = 100
TRAIN_RATIO = 0.7         # 70/30 train/test split (paper setting)
RANDOM_SEED = 42
WIN_RATE_THRESHOLD = 0.5  # Threshold for binary routing decision

# Multiple seeds for robust evaluation (paper uses 5 seeds)
EVAL_SEEDS = [42, 999, 2024, 2025, 3407]
