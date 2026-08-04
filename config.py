"""
HemoVisionAI — Central Configuration
======================================
Single source of truth for all hyperparameters, paths, and constants.

Every notebook and module imports from here. Hardcoded values in
individual files are forbidden.
"""

import os

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "mohammadamireshraghi", "blood-cell-cancer-all-4class",
    "versions", "1", "Blood cell Cancer [ALL]",
)

# ─────────────────────────────────────────────────────────────
# Image & Pipeline
# ─────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)    # (height, width) after resizing
BATCH_SIZE = 16
SEED = 42
VAL_SPLIT = 0.2            # fraction reserved for validation

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

NUM_CLASSES = 4
LEARNING_RATE = 0.001
EPOCHS = 50

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

# Project root (one level above this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Checkpoint is saved/loaded relative to notebooks/ working directory,
# but canonical location is under saved_models/.
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "saved_models")
CHECKPOINT_NAME = "hemovisionnet_v1_best.keras"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
