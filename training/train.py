# HemoVisionAI/training/train.py
"""
Retrain HemoVisionNet — runs on your laptop OR on Colab free GPU.

Examples:
  # Local quick check (2 epochs, CPU):
  .venv/bin/python training/train.py --version v2 --epochs 2

  # Full retrain on Colab GPU (see COLAB_RETRAIN.md):
  !python training/train.py --version v2 --epochs 50 --download

What it does:
  1. Finds / downloads the dataset (Kaggle blood-cell-cancer-all-4class)
  2. Builds stratified train/val split (seed fixed, same as report)
  3. Builds HemoVisionNet V1 (original) or V2 (brightness-robust)
  4. Trains with EarlyStopping + Checkpoint + ReduceLROnPlateau
  5. Prints val accuracy and where the best model was saved
"""

import argparse
import os
import sys

# Make `config`, `preprocessing`, `models` importable whether we run from
# the project root or from inside training/.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def resolve_dataset(data_dir: str, download: bool) -> str:
    """Return a folder containing one sub-folder per class.

    If the folder is missing/empty and --download is passed, fetch it with
    kagglehub (works on Colab after `!pip install kagglehub` + Kaggle login).
    """
    if os.path.isdir(data_dir) and any(
        os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir)
    ):
        return data_dir
    if not download:
        raise FileNotFoundError(
            f"Dataset not found at: {data_dir}\n"
            "Pass --download to fetch it with kagglehub, or point "
            "--data-dir at your local copy (e.g. dataset/'Blood cell Cancer [ALL]')."
        )
    import kagglehub

    print("Downloading dataset with kagglehub ...")
    root = kagglehub.dataset_download(
        "mohammadamireshraghi/blood-cell-cancer-all-4class"
    )
    inner = os.path.join(root, "Blood cell Cancer [ALL]")
    return inner if os.path.isdir(inner) else root


def main():
    from config import (
        IMAGE_SIZE,
        BATCH_SIZE,
        SEED,
        VAL_SPLIT,
        NUM_CLASSES,
        LEARNING_RATE,
        EPOCHS,
        CHECKPOINT_PATH,
    )

    parser = argparse.ArgumentParser(description="Retrain HemoVisionNet")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2",
                        help="v1 = original, v2 = brightness-robust (recommended)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Defaults to config.DATASET_PATH, falls back to ./dataset copy")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to save best model. Default: hemovisionnet_v2_best.keras")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset with kagglehub if not found")
    args = parser.parse_args()

    import tensorflow as tf

    from preprocessing.data_pipeline import get_datasets
    from models.cnn import build_hemovisionnet
    from training.callbacks import get_callbacks

    # GPU check (just info — training works on CPU too, only slower)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPU: {len(gpus)} found"
          + (f" ({gpus[0].name})" if gpus else " (CPU mode — fine, just slower)"))

    # Resolve dataset
    from config import DATASET_PATH

    candidates = [a for a in [args.data_dir, DATASET_PATH,
                              os.path.join(PROJECT_ROOT, "dataset",
                                           "Blood cell Cancer [ALL]")] if a]
    data_dir = None
    for c in candidates:
        if os.path.isdir(c) and os.listdir(c):
            data_dir = c
            break
    if data_dir is None:
        data_dir = resolve_dataset(candidates[0], download=args.download)
    print(f"Dataset: {data_dir}")

    # Output path
    if args.output:
        out_path = args.output
    elif args.version == "v2":
        out_path = os.path.join(PROJECT_ROOT, "saved_models",
                                "hemovisionnet_v2_best.keras")
    else:
        out_path = CHECKPOINT_PATH
    print(f"Best model will be saved to: {out_path}")

    # Data (augmentation lives INSIDE the model, so augment_fn=None here)
    train_ds, val_ds, class_names = get_datasets(
        dataset_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        val_split=VAL_SPLIT,
        seed=args.seed,
        augment_fn=None,
    )
    print(f"Classes: {class_names}")

    # Model
    model = build_hemovisionnet(
        image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, version=args.version
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    try:
        model.summary()
    except Exception:
        print(f"Params: {model.count_params()}")

    # Train
    callbacks = get_callbacks(out_path)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=callbacks, verbose=1,
    )

    # Quick honest score on val
    loss, acc = model.evaluate(val_ds, verbose=0)
    print("=" * 50)
    print(f"Done. Val accuracy: {acc:.4f} ({100*acc:.2f}%)  loss={loss:.4f}")
    print(f"Best checkpoint: {out_path}")
    print("Bring this .keras file back to saved_models/ and re-run the "
          "evaluation to refresh reports/evaluation_report.md")
    print("=" * 50)


if __name__ == "__main__":
    main()
