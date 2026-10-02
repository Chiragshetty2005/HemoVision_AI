# HemoVisionAI/training/callbacks.py
"""Standard callbacks for HemoVisionNet training (mirrors notebook 03)."""

import os
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


def get_callbacks(checkpoint_path: str):
    """EarlyStopping + ModelCheckpoint + ReduceLROnPlateau.

    Returns (callbacks_list, dict_of_callbacks) for logging.
    """
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=4, min_lr=1e-6, verbose=1
    )
    return [early_stopping, model_checkpoint, reduce_lr]
