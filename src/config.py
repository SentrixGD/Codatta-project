"""
File: config.py

Description:
    Central configuration for paths, model hyperparameters, training settings,
    and task definitions. Single source of truth — all other modules import from here.

Purpose:
    Eliminate magic numbers and duplicated constants across the codebase.

Usage:
    from src.config import ROOT_DIR, DATA_DIR, MODEL_CONFIG
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
STATS_DIR = os.path.join(ROOT_DIR, "stats")
RESULTS_DIR = os.path.join(STATS_DIR, "results")

# ──────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────

MODEL_CONFIG = {
    "heads_ratio": 32,
    "dim": 128,
    "dropout_mha": 0,
    "dropout_swin_mlp": 0,
    "dropout_outer": 0,
    "dropout_shared_mlp": 0.25,
    "dropout_pre_output": 0.1,
    "droppath": 0.1,
    "window_size": 7,
    "input_channels": 3,
    "depths": [2, 2, 18, 2],
    "stage_num": 4,
    "shared_mlp_size": 1024,
    "ingredients_mlp_size": 768,
    "portions_mlp_size": 768,
    "dish_names_mlp_size": 768,
    "food_type_classes": 5,
    "ingredients_classes": 589,
    "portion_size_classes": 437,
    "dish_names_classes": 602,
    "cooking_method_classes": 15,
    "binary_classes": 2,
}

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────

EPOCHS = 15
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
WEIGHT_OPTIMIZER_LR = 1e-2
ETA_MIN = 1e-6
EMA_DECAY = 0.99
WARMUP_RATIO = 0.05
GRAD_CLIP_MAX_NORM = 1.0
GRADIENT_CHECKPOINTING = True

# ──────────────────────────────────────────────
# Stress test / resolution
# ──────────────────────────────────────────────

STRESS_TEST_BATCH_SIZE = MICRO_BATCH_SIZE
STRESS_TEST_HEIGHT = 448
STRESS_TEST_ASPECT_RATIO = 3

# ──────────────────────────────────────────────
# Task definitions (for full multi-task setup)
# ──────────────────────────────────────────────

CLASSIFICATION_TASKS = [
    ("food_type", 1),
    ("dish_name", 5),
]

MULTILABEL_TASKS = [
    "ingredients",
    "portion_presence",
    "cooking_method",
]

REGRESSION_TASKS = [
    ("calories", "calories_kcal"),
    ("fats", "fat_g"),
    ("carbohydrates", "carbohydrate_g"),
    ("proteins", "protein_g"),
]

BINARY_PROB_TASKS = [
    (0, "camera_or_phone_prob", "camera_or_phone"),
    (1, "food_prob", "food"),
]

REGRESSION_METRICS = [
    "calories",
    "fats",
    "carbohydrates",
    "proteins",
    "camera_or_phone",
    "food",
]

# ──────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────

CONFUSION_LABELS = [0, 4, 3, 2, 1]
CONFUSION_CLASS_NAMES = [
    "Homemade food",
    "Restaurant food",
    "Raw vegetables and fruits",
    "Packaged food",
    "Others",
]

# ──────────────────────────────────────────────
# Metric schema
# ──────────────────────────────────────────────

METRIC_SCHEMA = {
    "meta": ["epoch", "model_name", "total_loss"],
    "classification": ["food_type_acc", "precision", "recall", "f1"],
}

HEADERS = sum(METRIC_SCHEMA.values(), [])
