"""
HemoVisionAI — Data Pipeline Module
====================================
Centralised, reusable data-loading and preprocessing utilities.

This module replaces the broken `image_dataset_from_directory(validation_split=...)`
approach with a robust, stratified pipeline:

    1. build_dataframe()       → scan class directories → Pandas DataFrame
    2. stratified_split()      → sklearn train_test_split with stratify
    3. print_split_summary()   → pretty-print class distribution tables
    4. df_to_tf_dataset()      → convert DataFrame → tf.data.Dataset
    5. get_datasets()          → convenience wrapper combining 1-4

Every notebook imports this single module, guaranteeing:
  • identical splits (same random_state)
  • no data leakage
  • no duplicate preprocessing code
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# 1.  Build a DataFrame of (image_path, label, class_name)
# ─────────────────────────────────────────────────────────────

def build_dataframe(dataset_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Walk the class sub-directories inside `dataset_dir` and return a
    DataFrame with columns [image_path, label, class_name] plus the
    ordered list of class names.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing one sub-folder per class.

    Returns
    -------
    df : pd.DataFrame
        Columns: image_path (str), label (int), class_name (str).
    class_names : list[str]
        Sorted list of class directory names.
    """

    # --- Collect every image path and its class ----------------
    records = []

    # Sort class names for deterministic label assignment
    class_names = sorted(
        [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
    )

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)

        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)

            # Only include actual image files
            if os.path.isfile(fpath) and fname.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            ):
                records.append(
                    {
                        "image_path": fpath,
                        "label": label_idx,
                        "class_name": class_name,
                    }
                )

    df = pd.DataFrame(records)

    print(f"✅ Built DataFrame: {len(df)} images across {len(class_names)} classes")

    return df, class_names


# ─────────────────────────────────────────────────────────────
# 2.  Stratified train / validation split
# ─────────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and validation sets with **stratification**
    so that each class is proportionally represented in both sets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'label' column.
    test_size : float
        Fraction of data reserved for validation (default 0.2 = 20 %).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_df, val_df : tuple[pd.DataFrame, pd.DataFrame]
    """

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],          # ← the critical fix
    )

    # Reset indices for clean iteration
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"✅ Stratified split: {len(train_df)} train / {len(val_df)} val")

    return train_df, val_df


# ─────────────────────────────────────────────────────────────
# 3.  Pretty-print class distribution tables
# ─────────────────────────────────────────────────────────────

def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
) -> None:
    """
    Print class-count and percentage tables for train and validation sets.
    """

    print("\n" + "=" * 60)
    print("  TRAINING SET — Class Distribution")
    print("=" * 60)
    _print_distribution(train_df, class_names)

    print("\n" + "=" * 60)
    print("  VALIDATION SET — Class Distribution")
    print("=" * 60)
    _print_distribution(val_df, class_names)

    print()


def _print_distribution(df: pd.DataFrame, class_names: list[str]) -> None:
    """Helper: print a single distribution table."""

    total = len(df)

    print(f"{'Class':<30} {'Count':>8} {'Percentage':>12}")
    print("-" * 52)

    for idx, name in enumerate(class_names):
        count = (df["label"] == idx).sum()
        pct = count / total * 100
        print(f"{name:<30} {count:>8}   {pct:>8.2f} %")

    print("-" * 52)
    print(f"{'TOTAL':<30} {total:>8}   {'100.00':>8} %")


# ─────────────────────────────────────────────────────────────
# 4.  Convert DataFrame → tf.data.Dataset
# ─────────────────────────────────────────────────────────────

def _load_and_preprocess(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow-graph-compatible image loader.

    Reads the file at `path`, decodes it, resizes to `image_size`,
    and casts to float32.

    NOTE: Pixel values are returned as float32 in [0, 255] range.
    Normalization is NOT done here because the HemoVisionNet model
    contains a Rescaling(1/255) layer internally. Normalizing in both
    places causes double-normalization ([0,1] → [0, 0.004]) which
    collapses all features and makes the model predict a single class.
    """

    # Read raw bytes from disk
    raw = tf.io.read_file(path)

    # Decode JPEG into a 3-channel uint8 tensor (fast and thread-safe)
    image = tf.io.decode_jpeg(raw, channels=3)

    # Resize to target dimensions
    image = tf.image.resize(image, image_size)

    # Cast to float32 for TF pipeline compatibility.
    # DO NOT divide by 255 — the model's internal Rescaling layer handles that.
    image = tf.cast(image, tf.float32)

    # Set explicit static shape to prevent kernel crash during model.fit() in Jupyter
    image.set_shape([image_size[0], image_size[1], 3])

    return image, label


def df_to_tf_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
    shuffle: bool = True,
    augment_fn: tf.keras.Sequential | None = None,
) -> tf.data.Dataset:
    """
    Build a performant tf.data.Dataset from a Pandas DataFrame.

    Pipeline: tensor_slices → map(load & preprocess) → [shuffle] →
              batch → [augment] → prefetch.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'image_path' and 'label' columns.
    image_size : tuple[int, int]
        Target (height, width) for resizing.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset (True for training, False for val).
    augment_fn : tf.keras.Sequential or None
        Optional data augmentation pipeline applied to image batches.

    Returns
    -------
    tf.data.Dataset
        Yields (image_batch, label_batch) tuples.
    """

    AUTOTUNE = tf.data.AUTOTUNE

    # Create a dataset of (path, label) pairs
    paths = df["image_path"].values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Load, decode, resize, normalise each image
    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l, image_size),
        num_parallel_calls=AUTOTUNE,
    )

    # Shuffle (training only) with a buffer proportional to dataset size
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)

    # Batch
    ds = ds.batch(batch_size)

    # Apply augmentation (training only — typically passed only for train set)
    if augment_fn is not None:
        ds = ds.map(
            lambda x, y: (augment_fn(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Prefetch for pipeline parallelism
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# ─────────────────────────────────────────────────────────────
# 5.  High-level convenience wrapper
# ─────────────────────────────────────────────────────────────

def get_datasets(
    dataset_dir: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
    augment_fn: tf.keras.Sequential | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    End-to-end pipeline: scan directory → stratified split → tf.data.Datasets.

    Parameters
    ----------
    dataset_dir : str
        Root directory with one sub-folder per class.
    image_size : tuple[int, int]
        Target (height, width).
    batch_size : int
        Batch size for both train and val datasets.
    val_split : float
        Fraction for validation (default 0.2).
    seed : int
        Random state for reproducibility.
    augment_fn : tf.keras.Sequential or None
        Data augmentation applied **only** to the training set.

    Returns
    -------
    train_ds : tf.data.Dataset
    val_ds   : tf.data.Dataset
    class_names : list[str]
    """

    # Step 1 — Build the DataFrame
    df, class_names = build_dataframe(dataset_dir)

    # Step 2 — Stratified split
    train_df, val_df = stratified_split(df, test_size=val_split, random_state=seed)

    # Step 3 — Print distribution summary
    print_split_summary(train_df, val_df, class_names)

    # Step 4 — Convert to tf.data.Datasets
    train_ds = df_to_tf_dataset(
        train_df,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        augment_fn=augment_fn,
    )

    val_ds = df_to_tf_dataset(
        val_df,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,        # never shuffle validation data
        augment_fn=None,      # never augment validation data
    )

    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────────────────────
# 6. Standalone Execution Entrypoint (Self-Test)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Add project root to path for config import when executed directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config import DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, SEED

    print("=" * 60)
    print("  HemoVisionAI Data Pipeline — Standalone Self-Test")
    print("=" * 60)

    train_ds, val_ds, class_names = get_datasets(
        dataset_dir=DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
    )

    print("\n✅ Verification:")
    print(f"  • Class names ({len(class_names)}): {class_names}")
    
    for images, labels in train_ds.take(1):
        pmin = tf.reduce_min(images).numpy()
        pmax = tf.reduce_max(images).numpy()
        print(f"  • Training batch image shape: {images.shape}")
        print(f"  • Training batch label shape: {labels.shape}")
        print(f"  • Pixel range: [{pmin:.2f}, {pmax:.2f}] (float32)")
        assert pmax > 1.0, "Pixel range error: images appear pre-normalized!"

    print("\n🎉 Pipeline self-test complete! All checks passed successfully.")

