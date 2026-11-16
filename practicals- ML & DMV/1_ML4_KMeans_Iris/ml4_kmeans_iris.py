import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
data = pd.read_csv(upload_file("Iris.csv"))
print(data.head())
print("\nShape:", data.shape)
print("\nSpecies count:\n", data["Species"].value_counts())

print("\n=== Step 2: Select Features (drop Id and Species - clustering is unsupervised) ===")
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
X_scaled = StandardScaler().fit_transform(X)
print(X.describe().round(2))

print("\n=== Step 3: Elbow Method (WCSS for k = 1..10) ===")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    print(f"k = {i:<2}  WCSS = {kmeans.inertia_:.2f}")

plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker="o")
plt.axvline(3, ls="--", color="red", label="Elbow at k = 3")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (inertia)")
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== Step 4: Silhouette Score (extra check) ===")
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    print(f"k = {k}  silhouette = {silhouette_score(X_scaled, labels):.4f}")

print("\n=== Step 5: Apply K-Means with k = 3 ===")
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
data["Cluster"] = y_kmeans
print("Cluster sizes:\n", data["Cluster"].value_counts().sort_index())
centers = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
print("\nCluster centres (original units):\n",
      pd.DataFrame(centers, columns=X.columns).round(2))

print("\n=== Step 6: Compare Clusters with Actual Species ===")
print(pd.crosstab(data["Species"], data["Cluster"]))

print("\n=== Step 7: Visualize Clusters ===")
Xv = X.values
plt.figure(figsize=(7, 5))
for c, col in zip(range(3), ["purple", "orange", "green"]):
    plt.scatter(Xv[y_kmeans == c, 2], Xv[y_kmeans == c, 3], s=50, c=col, label=f"Cluster {c + 1}")
plt.scatter(centers[:, 2], centers[:, 3], s=250, c="red", marker="X", label="Centroids")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-Means Clusters on Iris (k = 3)")
plt.legend()
plt.tight_layout()
plt.show()
