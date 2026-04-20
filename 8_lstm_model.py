"""
8. LSTM
========
LSTM model factory and user-independent / user-dependent evaluation.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences


def prepare_lstm_data(data: list, labels: list,
                      max_len: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Pad sequences for LSTM input and return (X, y) arrays."""
    padded = pad_sequences(data, maxlen=max_len,
                           dtype="float32", padding="post")
    return padded, np.array(labels)


def _build_lstm_model(timesteps: int, num_classes: int) -> Sequential:
    """Factory that returns a freshly compiled LSTM model."""
    model = Sequential([
        Input(shape=(timesteps, 3)),
        Masking(mask_value=0.0),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def lstm_evaluation(
        data: list, labels: list, users: list
) -> tuple[float, float, list]:
    """
    Evaluate LSTM using leave-one-user-out CV (user-independent setting).
    Epochs = 30, validation_split = 0.1.
    """
    min_label   = int(np.min(labels))
    labels_zi   = np.array(labels) - min_label   # 0-indexed
    num_classes = len(np.unique(labels_zi))

    X, _ = prepare_lstm_data(data, labels_zi)

    unique_users = sorted(set(users))
    accs = []

    for u in unique_users:
        train_idx = [i for i, usr in enumerate(users) if usr != u]
        test_idx  = [i for i, usr in enumerate(users) if usr == u]

        y_train = labels_zi[np.array(train_idx)]
        y_test  = labels_zi[np.array(test_idx)]

        model = _build_lstm_model(X.shape[1], num_classes)
        model.fit(X[train_idx], y_train,
                  epochs=30, batch_size=32,
                  validation_split=0.1, verbose=0)

        _, acc = model.evaluate(X[test_idx], y_test, verbose=0)
        accs.append(float(acc))
        print(f"    LSTM – User {u} → accuracy = {acc:.3f}")

    return float(np.mean(accs)), float(np.std(accs)), accs


def lstm_evaluation_user_dependent(
        data: list, labels: list, users: list
) -> tuple[float, float, list]:
    """
    Evaluate LSTM using leave-one-sample-out CV (user-dependent setting).
    """
    min_label   = int(np.min(labels))
    labels_zi   = np.array(labels) - min_label   # 0-indexed, shape (N,)
    num_classes = len(np.unique(labels_zi))

    X, _ = prepare_lstm_data(data, labels_zi)

    unique_users    = sorted(set(users))
    gesture_classes = sorted(set(labels))
    fold_accs = []

    for u in unique_users:
        gesture_to_indices: dict[int, list] = {g: [] for g in gesture_classes}
        for i, (lbl, usr) in enumerate(zip(labels, users)):
            if usr == u:
                gesture_to_indices[lbl].append(i)

        n_folds = min(len(v) for v in gesture_to_indices.values())

        for fold in range(n_folds):
            test_idx  = [gesture_to_indices[g][fold] for g in gesture_classes]
            train_idx = [
                gesture_to_indices[g][rep]
                for g in gesture_classes
                for rep in range(n_folds) if rep != fold
            ]

            y_train = labels_zi[np.array(train_idx)]
            y_test  = labels_zi[np.array(test_idx)]

            model = _build_lstm_model(X.shape[1], num_classes)
            model.fit(X[train_idx], y_train,
                      epochs=30, batch_size=16, verbose=0)

            _, acc = model.evaluate(X[test_idx], y_test, verbose=0)
            fold_accs.append(float(acc))

        print(f"    LSTM (dep.) – User {u} → mean acc = "
              f"{np.mean(fold_accs[-n_folds:]):.3f}")

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs
