from pathlib import Path

import numpy as np
from tensorflow import keras

from utils.finbert_cnn import BASELINE_SEED, MacroF1Callback, train_baseline
from utils.preprocessing import LABELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models/finbert_cnn_frozen_baseline.keras"
STOP_MODES = {"val_loss": "min", "val_macro_f1": "max"}


def ensure_baseline_checkpoint(path, embedding_matrix, X_train, y_train, X_val, y_val, trainable=False):
    # train the baseline once and save it. If the file is already there we keep it,
    # so every run starts from the same weights
    path = Path(path)
    if path.exists():
        return path
    keras.utils.set_random_seed(BASELINE_SEED)
    model = train_baseline(embedding_matrix, X_train, y_train, X_val, y_val, trainable=trainable,
                           verbose=0, epochs=80 if trainable else 20)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"saved baseline checkpoint to {path}")
    return path


def continue_training(checkpoint_path, X_train, y_train, X_val, y_val, sample_weight=None,
                      learning_rate=1e-5, epochs=20, patience=3, batch_size=32, verbose=0, stop_on="val_loss"):
    if stop_on not in STOP_MODES:
        raise ValueError(f"stop_on must be one of {list(STOP_MODES)}")
    model = keras.models.load_model(checkpoint_path, compile=False)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if len(sample_weight) != len(y_train):
            raise ValueError("sample_weight must have one weight per training sample")
    early_stop = keras.callbacks.EarlyStopping(monitor=stop_on, mode=STOP_MODES[stop_on], patience=patience,
                                               restore_best_weights=True)
    callbacks = [early_stop] if stop_on == "val_loss" else [MacroF1Callback(X_val, y_val), early_stop]
    history = model.fit(X_train, keras.utils.to_categorical(y_train, len(LABELS)), sample_weight=sample_weight,
                        validation_data=(X_val, keras.utils.to_categorical(y_val, len(LABELS))),
                        epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)
    return model, history


def matched_random_weights(target_weights, seed):
    # random control: the same weights as the targeted run, but on randomly chosen sentences
    target_weights = np.asarray(target_weights, dtype=np.float32)
    changed = target_weights[target_weights != 1.0]
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(target_weights), size=len(changed), replace=False)
    control = np.ones_like(target_weights)
    control[rows] = rng.permutation(changed)
    return control
