# HemoVisionAI/models/cnn.py
"""
HemoVisionNet — custom CNN for blood-cell cancer classification.

V1: exact replica of the model defined in notebooks/03_model_development.ipynb
    (the weights in saved_models/hemovisionnet_v1_best.keras).
V2: same backbone + brightness-robust augmentation (fixes the Section-12
    robustness finding where dark x0.6 -> 31% and bright x1.4 -> 21%).

Pipeline contract (do not break):
  data_pipeline delivers float32 pixels in [0, 255] -> augmentation runs on
  [0, 255] -> internal Rescaling(1/255) normalises. Never divide by 255 outside.
"""

from tensorflow.keras import layers, Model


def get_augmentation(version: str = "v1") -> "tf.keras.Sequential":
    """Training-only augmentation block. Lives INSIDE the model.

    V1 keeps the original 4 ops. V2 adds brightness + stronger contrast +
    small translation to survive real-world lighting / framing changes.
    """
    import tensorflow as tf

    if version == "v2":
        return tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.10),
                layers.RandomZoom(0.10),
                layers.RandomContrast(0.20),  # stronger than V1's 0.10
                layers.RandomBrightness(
                    0.20, value_range=(0, 255)
                ),  # NEW: +/-20% brightness, fixes dark/bright collapse
                layers.RandomTranslation(0.05, 0.05),  # NEW: small shift, helps framing
            ],
            name="DataAugmentation",
        )
    # v1 — original, bit-for-bit compatible with the saved checkpoint
    return __import__("tensorflow").keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="DataAugmentation",
    )


def hemo_stem_block(inputs):
    """Initial conv block for low-level feature extraction."""
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def hemo_feature_block(inputs, filters: int):
    """Double-conv block for hierarchical feature extraction."""
    x = layers.Conv2D(filters, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def build_hemovisionnet(
    image_size: tuple = (224, 224),
    num_classes: int = 4,
    version: str = "v1",
    dropout: float = 0.5,
) -> Model:
    """Assemble HemoVisionNet.

    version="v1": identical to notebooks/03 (loads hemovisionnet_v1_best.keras weights).
    version="v2": same backbone, robust augmentation (retrain to get V2 weights).
    """
    name = "HemoVisionNet_V1" if version == "v1" else "HemoVisionNet_V2"
    inputs = layers.Input(shape=(*image_size, 3), name="Input_Image")

    x = get_augmentation(version)(inputs)

    # Normalise [0, 255] -> [0, 1] INSIDE the model.
    x = layers.Rescaling(1.0 / 255, name="Normalization")(x)

    x = hemo_stem_block(x)
    x = hemo_feature_block(x, 64)
    x = hemo_feature_block(x, 128)
    x = hemo_feature_block(x, 256)

    x = layers.GlobalAveragePooling2D(name="GlobalAveragePooling")(x)
    x = layers.Dense(256, activation="relu", name="Dense_256")(x)
    x = layers.Dropout(dropout, name="Dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="Prediction")(x)

    model = Model(inputs=inputs, outputs=outputs, name=name)
    assert model.output_shape[-1] == num_classes
    return model
