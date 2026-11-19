
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "listings", "data", "up.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "land_dl_model.keras")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "land_preprocessor.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(DATASET_PATH)

# clean numbers
def clean(x):
    x = str(x).lower().replace(",", "").strip()
    if "cr" in x:
        return float(x.replace("cr", "")) * 1e7
    if "l" in x:
        return float(x.replace("l", "")) * 1e5
    return float(x)

# df["Area"] = df["Area"].apply(clean)
# df["price"] = df["price"].apply(clean)

df = df.dropna()

# -----------------------
# FEATURES
# -----------------------
X = df[["State", "District", "Plot_Type"]]
y = df["Price_per_sqft"]

# -----------------------
# PREPROCESSOR
# -----------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), ["State", "District", "Plot_Type"])
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

joblib.dump(preprocessor, PREPROCESSOR_PATH)

# -----------------------
# ANN MODEL
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# -----------------------
# MODEL EVALUATION
# -----------------------
y_pred = model.predict(X_test, verbose=0).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Adjusted R2
n = len(y_test)
p = X_test.shape[1]  # number of encoded features

if n > p + 1:
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
else:
    adjusted_r2 = np.nan

print("\n========== LAND ANN EVALUATION ==========")
print(f"MAE        : {mae:.4f}")
print(f"MSE        : {mse:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R2 Score   : {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")
print("======================================")

# -----------------------
# SAVE MODEL
# -----------------------
model.save(MODEL_PATH)

print("✅ LAND ANN MODEL TRAINED, EVALUATED & SAVED")
