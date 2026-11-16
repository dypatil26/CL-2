import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
df = pd.read_csv(upload_file("Wine.csv"))
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nClass distribution (Customer_Segment):\n", df["Customer_Segment"].value_counts().sort_index())

print("\n=== Step 2: Separate Features (X) and Target (y) ===")
X = df.drop("Customer_Segment", axis=1)
y = df["Customer_Segment"]
print("Features:", list(X.columns))

print("\n=== Step 3: Standardize the Features ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Mean after scaling (approx 0):", np.round(X_scaled.mean(axis=0)[:5], 3))
print("Std  after scaling (approx 1):", np.round(X_scaled.std(axis=0)[:5], 3))

print("\n=== Step 4: Covariance Matrix, Eigenvalues (manual check) ===")
cov = np.cov(X_scaled.T)
eig_vals, eig_vecs = np.linalg.eig(cov)
eig_vals = np.sort(eig_vals)[::-1]
print("Covariance matrix shape:", cov.shape)
print("Top 5 eigenvalues:", np.round(eig_vals[:5], 3))

print("\n=== Step 5: Apply PCA with all components ===")
pca_full = PCA()
pca_full.fit(X_scaled)
ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)
for i, (r, c) in enumerate(zip(ratio, cum), start=1):
    print(f"PC{i:<2}  variance = {r:.4f}   cumulative = {c:.4f}")
k95 = int(np.argmax(cum >= 0.95) + 1)
print(f"\nComponents needed to keep 95% variance: {k95}")

plt.figure(figsize=(8, 4))
plt.bar(range(1, len(ratio) + 1), ratio, label="Individual")
plt.step(range(1, len(cum) + 1), cum, where="mid", color="red", label="Cumulative")
plt.axhline(0.95, ls="--", color="grey")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot - Wine Dataset")
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== Step 6: Reduce to 2 Principal Components ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Customer_Segment"] = y.values
print(pca_df.head())
print("Variance captured by 2 PCs:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

print("\n=== Step 7: Feature Loadings (contribution of each feature) ===")
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=X.columns)
print(loadings.round(3))

print("\n=== Step 8: Visualize Data after PCA ===")
plt.figure(figsize=(7, 6))
colors = {1: "red", 2: "green", 3: "blue"}
for seg, c in colors.items():
    part = pca_df[pca_df["Customer_Segment"] == seg]
    plt.scatter(part["PC1"], part["PC2"], c=c, label=f"Class {seg}", alpha=0.7, edgecolors="k")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine Dataset after PCA (13 features -> 2)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nConclusion: 13 correlated features were reduced to 2 principal components "
      "while keeping most of the variance, and the wine classes are clearly separable in the PCA plot.")
