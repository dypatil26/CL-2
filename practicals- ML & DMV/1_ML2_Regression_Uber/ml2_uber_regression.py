import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------
# Upload helper: the program asks for its dataset when this runs.
#   * Google Colab            -> shows Colab's "Choose files" button
#   * Jupyter / Python on PC  -> opens a file-picker window
#   * anything else           -> asks you to type the file's path
# The dataset can be kept anywhere on the computer.
# ------------------------------------------------------------------
def upload_file(name, multiple=False):
    print(f"\n>>> Please upload / select: {name}")
    try:
        from google.colab import files                  # only exists on Google Colab
        paths = list(files.upload().keys())
        return paths if multiple else paths[0]
    except ImportError:
        pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)               # keep the picker in front
        ext = name[name.rfind("."):] if ("." in name and " " not in name) else ".*"
        types = [("Data files", "*" + ext), ("All files", "*.*")]
        if multiple:
            paths = list(filedialog.askopenfilenames(title="Select " + name, filetypes=types, parent=root))
        else:
            paths = filedialog.askopenfilename(title="Select " + name, filetypes=types, parent=root)
        root.destroy()
        if paths:
            print("Selected:", paths)
            return paths
    except Exception:
        pass
    typed = input(f"Type the full path of {name}: ").strip()
    if multiple:
        return [p.strip().strip('"') for p in typed.split(",")]
    return typed.strip('"')

print("=== Step 1: Load Dataset ===")
df = pd.read_csv(upload_file("uber.csv"))
print(df.head())
print("\nShape:", df.shape)
print(df.info())

print("\n=== Step 2: Pre-process the Dataset ===")
df = df.drop(columns=["Unnamed: 0", "key"])
print("Missing values:\n", df.isnull().sum())
df = df.dropna()
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df = df.dropna(subset=["pickup_datetime"])

# feature engineering from date-time
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["dayofweek"] = df["pickup_datetime"].dt.dayofweek


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance between two GPS points in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


df["distance_km"] = haversine(df["pickup_longitude"], df["pickup_latitude"],
                              df["dropoff_longitude"], df["dropoff_latitude"])
print("\nAfter feature engineering:\n", df.head())

print("\n=== Step 3: Identify Outliers ===")
num_cols = ["fare_amount", "distance_km", "passenger_count"]
plt.figure(figsize=(12, 4))
for i, c in enumerate(num_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[c])
    plt.title(f"Before: {c}")
plt.tight_layout()
plt.show()

before = len(df)
# rule-based cleaning: impossible values
df = df[(df["fare_amount"] > 0) & (df["passenger_count"].between(1, 6))]
df = df[df["pickup_latitude"].between(40, 42) & df["pickup_longitude"].between(-75, -72)]
df = df[df["dropoff_latitude"].between(40, 42) & df["dropoff_longitude"].between(-75, -72)]
df = df[df["distance_km"] > 0]
# IQR method for fare and distance
for c in ["fare_amount", "distance_km"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"{c}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} -> keep [{low:.2f}, {high:.2f}]")
    df = df[df[c].between(low, high)]
print(f"Rows removed as outliers/invalid: {before - len(df)}  (remaining {len(df)})")

print("\n=== Step 4: Check the Correlation ===")
corr = df[["fare_amount", "distance_km", "passenger_count", "hour", "day", "month", "year", "dayofweek"]].corr()
print(corr["fare_amount"].sort_values(ascending=False).round(3))
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\n=== Step 5: Train/Test Split and Scaling ===")
features = ["distance_km", "passenger_count", "hour", "day", "month", "year", "dayofweek"]
X = df[features]
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Train:", X_train.shape, " Test:", X_test.shape)

print("\n=== Step 6: Implement Linear, Ridge and Lasso Regression ===")
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression (alpha=1)": Ridge(alpha=1.0),
    "Lasso Regression (alpha=0.1)": Lasso(alpha=0.1),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    results.append([name, r2, rmse, mae])
    print(f"\n{name}")
    print("  Coefficients:", {f: round(float(c), 3) for f, c in zip(features, model.coef_)})
    print("  Intercept   :", round(float(model.intercept_), 3))

print("\n=== Step 7: Evaluate and Compare the Models ===")
res = pd.DataFrame(results, columns=["Model", "R2", "RMSE", "MAE"])
print(res.round(4).to_string(index=False))

plt.figure(figsize=(6, 4))
plt.bar(res["Model"].str.split(" ").str[0], res["R2"], color=["tab:blue", "tab:orange", "tab:green"])
plt.ylabel("R2 score")
plt.title("Model Comparison (R2)")
plt.tight_layout()
plt.show()

best = models["Linear Regression"].predict(X_test)
plt.figure(figsize=(6, 5))
plt.scatter(y_test, best, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted (Linear Regression)")
plt.tight_layout()
plt.show()
