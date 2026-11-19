
import os
import re
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------
# SETTINGS
# -----------------------
tf.keras.utils.set_random_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "listings", "data", "dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(
    MODEL_DIR, "construction_dl_model.keras"
)
PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR, "construction_preprocessor.joblib"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
print("Loading construction dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

def normalize_column(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())

column_map = {
    normalize_column(col): col for col in df.columns
}

print("Available columns:", df.columns.tolist())

def find_column(possible_names):
    for name in possible_names:
        key = normalize_column(name)
        if key in column_map:
            return column_map[key]
    return None

# -----------------------
# FIND TARGET
# -----------------------
TARGET = find_column([
    "price_per_sqft",
    "Price_in_SqFt",
    "Price Per Sqft",
    "Price Per Square Feet",
    "price_sqft"
])

if TARGET is None:
    raise ValueError(
        "Price per square foot target column not found."
    )

# -----------------------
# FIND FEATURES
# -----------------------
state_col = find_column(["state"])
city_col = find_column(["city"])

bhk_col = find_column([
    "bhk", "bedrooms"
])

bath_col = find_column([
    "bathrooms", "bathroom", "bath"
])

age_col = find_column([
    "age", "Age_of_Property", "property_age"
])

size_col = find_column([
    "Size_in_SqFt", "size_sqft", "area_sqft",
    "size", "area"
])

categorical = []
numeric = []

for col in [state_col, city_col]:
    if col and col not in categorical:
        categorical.append(col)

for col in [bhk_col, bath_col, age_col, size_col]:
    if col and col not in numeric:
        numeric.append(col)

FEATURES = categorical + numeric

if not FEATURES:
    raise ValueError("No usable features detected.")

print("\nSelected features:", FEATURES)
print("Target:", TARGET)

# -----------------------
# CLEAN DATA
# -----------------------
df = df[FEATURES + [TARGET]].copy()

for col in numeric + [TARGET]:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

for col in categorical:
    df[col] = (
        df[col].astype("string").str.strip()
    )
    df[col] = df[col].replace(
        ["", "nan", "None"], np.nan
    )

df = df.dropna(subset=[TARGET])

for col in numeric:
    df[col] = df[col].fillna(df[col].median())

for col in categorical:
    df[col] = df[col].fillna("Unknown")

df = df[df[TARGET] > 0]

print("\nDataset shape:", df.shape)

# -----------------------
# FEATURES AND TARGET
# -----------------------
X = df[FEATURES]
y = df[TARGET]

# -----------------------
# TRAIN TEST SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# PREPROCESSING
# -----------------------
transformers = []

if categorical:
    transformers.append(
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical
        )
    )

if numeric:
    transformers.append(
        (
            "num",
            StandardScaler(),
            numeric
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

print("\nPreprocessing data...")

X_train = preprocessor.fit_transform(
    X_train
).astype(np.float32)

X_test = preprocessor.transform(
    X_test
).astype(np.float32)

joblib.dump(preprocessor, PREPROCESSOR_PATH)

print("Training data shape:", X_train.shape)

# -----------------------
# BUILD FASTER ANN
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train.shape[1],)
    ),

    tf.keras.layers.Dense(
        32, activation="relu"
    ),

    tf.keras.layers.Dense(
        16, activation="relu"
    ),

    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

# -----------------------
# EARLY STOPPING
# -----------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# -----------------------
# TRAIN MODEL
# -----------------------
print("\nTraining Construction ANN...")

history = model.fit(
    X_train,
    y_train.to_numpy(dtype=np.float32),

    epochs=30,
    batch_size=256,

    validation_split=0.1,

    callbacks=[early_stopping],

    verbose=1
)

# -----------------------
# PREDICTION
# -----------------------
print("\nMaking predictions...")

preds = model.predict(
    X_test,
    batch_size=256,
    verbose=0
).flatten()

# -----------------------
# EVALUATION
# -----------------------
mae = mean_absolute_error(y_test, preds)

mse = mean_squared_error(
    y_test, preds
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test, preds
)

n = len(y_test)
p = X_test.shape[1]

if n > p + 1:
    adjusted_r2 = 1 - (
        (1 - r2) * (n - 1) / (n - p - 1)
    )
else:
    adjusted_r2 = np.nan

print("\n========== CONSTRUCTION ANN RESULTS ==========")

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")

print("\nR2 Percentage:", round(r2 * 100, 2), "%")

# -----------------------
# SAVE MODEL
# -----------------------
model.save(MODEL_PATH)

print("\nConstruction ANN trained successfully!")

print("Model saved:", MODEL_PATH)
print("Preprocessor saved:", PREPROCESSOR_PATH)
