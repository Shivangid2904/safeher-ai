import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# LOAD DATA (ROBUST PATH)
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

data_path = os.path.abspath(
    os.path.join(BASE_DIR, "../../data/synthetic/risk_dataset.csv")
)

print(f"Loading dataset from: {data_path}")

df = pd.read_csv(data_path)

# -----------------------------
# FEATURES & LABELS
# -----------------------------
X = df[['lat', 'long', 'hour', 'crime_score', 'crowd_density']]
y = df['risk']

# Encode labels (Safe, Moderate, High → 0,1,2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# BUILD MODEL
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy:.2f}")

model_path = os.path.join(BASE_DIR, "model.h5")
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

model.save(model_path)
joblib.dump(label_encoder, encoder_path)

print("\nModel saved at:", model_path)
print("Encoder saved at:", encoder_path)

print("\n✅ Training complete!")