import os, numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_DIR = "data_vec"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

X, y = [], []
counts = {lab: 0 for lab in LABELS}

for i, lab in enumerate(LABELS):
    d = os.path.join(DATA_DIR, lab)
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        if fn.endswith(".npy"):
            vec = np.load(os.path.join(d, fn)).astype(np.float32)
            if vec.shape[0] == 48:
                X.append(vec)
                y.append(i)
                counts[lab] += 1

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("Samples:", X.shape, "Labels:", y.shape)
print("Per-class counts (must NOT be too low):")
for lab in LABELS:
    print(lab, counts[lab])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(LABELS), activation="softmax"),  # 36
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cb = [
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80, batch_size=128,
    callbacks=cb
)

os.makedirs("models", exist_ok=True)
model.save("models/mlp48.keras")
print("Saved: models/mlp48.keras")
