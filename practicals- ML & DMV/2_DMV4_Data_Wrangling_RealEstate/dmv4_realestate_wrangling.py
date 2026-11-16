import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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

print("=== Step 1: Import Dataset and Clean Column Names ===")
df = pd.read_csv(upload_file("RealEstate_Prices.csv"))
print("Original columns:", list(df.columns))


def clean_name(c):
    c = c.strip().lower()
    c = re.sub(r"\(.*?\)", "", c)          # remove text in brackets: (Rs), (sqft)
    c = re.sub(r"[^a-z0-9]+", "_", c)      # spaces / special chars -> _
    return c.strip("_")


df.columns = [clean_name(c) for c in df.columns]
df = df.rename(columns={"sq_ft": "area_sqft", "bedrooms": "bedrooms", "lot_size": "lot_size_sqft", "sale_price": "price"})
print("Cleaned columns :", list(df.columns))
print(df.head())

print("\n=== Step 2: Handle Missing Values ===")
print(df.isnull().sum())
df = df.dropna(subset=["price"])                                          # target missing -> drop row
df["bathrooms"] = df["bathrooms"].fillna(df.groupby("bedrooms")["bathrooms"].transform("median"))
df["lot_size_sqft"] = df["lot_size_sqft"].fillna(0)                       # apartments have no lot
print("\nAfter handling:\n", df.isnull().sum().sum(), "missing values left")

print("\n=== Step 3: Merge with Neighborhood Information ===")
info = pd.read_csv(upload_file("Neighborhood_Info.csv"))
info.columns = [clean_name(c) for c in info.columns]
df = df.merge(info, on="neighborhood", how="left")
print(df[["neighborhood", "avg_income_lpa", "schools_nearby", "metro_access"]].drop_duplicates().head(8))

print("\n=== Step 4: Filter and Subset ===")
df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_year"] = df["sale_date"].dt.year
recent_apts = df[(df["sale_year"] >= 2023) & (df["property_type"] == "Apartment")]
print("Apartments sold in 2023-24:", len(recent_apts))
baner = df.loc[df["neighborhood"] == "Baner", ["property_id", "area_sqft", "price"]]
print("Properties in Baner:", len(baner))
print(baner.head())

print("\n=== Step 5: Encode Categorical Variables ===")
le = LabelEncoder()
df["metro_access_enc"] = le.fit_transform(df["metro_access"])            # label encoding (binary)
df = pd.get_dummies(df, columns=["property_type"], prefix="type", dtype=int)   # one-hot encoding
print([c for c in df.columns if c.startswith("type_")] + ["metro_access_enc"])

print("\n=== Step 6: Aggregate - Summary Statistics ===")
df["price_per_sqft"] = (df["price"] / df["area_sqft"]).round(0)
by_hood = df.groupby("neighborhood").agg(avg_price=("price", "mean"), median_price=("price", "median"),
                                         avg_price_sqft=("price_per_sqft", "mean"), count=("price", "size"))
print((by_hood.sort_values("avg_price_sqft", ascending=False)).round(0))
types = [c for c in df.columns if c.startswith("type_")]
by_type = {t.replace("type_", ""): df.loc[df[t] == 1, "price"].mean() for t in types}
print("\nAverage price by property type:\n", pd.Series(by_type).round(0))

print("\n=== Step 7: Identify and Handle Outliers ===")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=df["price_per_sqft"])
plt.title("Price per sqft (before)")
q1, q3 = df["price_per_sqft"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = df[(df["price_per_sqft"] < low) | (df["price_per_sqft"] > high)]
print(f"IQR limits: [{low:.0f}, {high:.0f}]  -> outliers found: {len(out)}")
print(out[["property_id", "neighborhood", "area_sqft", "price", "price_per_sqft"]].head(10))
df = df[(df["price_per_sqft"] >= low) & (df["price_per_sqft"] <= high)]
plt.subplot(1, 2, 2)
sns.boxplot(y=df["price_per_sqft"])
plt.title("Price per sqft (after)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
df.groupby("neighborhood")["price_per_sqft"].mean().sort_values().plot(kind="barh", color="darkcyan")
plt.title("Average Price per Sq.Ft by Neighborhood")
plt.xlabel("Rs / sqft")
plt.tight_layout()
plt.show()

print("\nCorrelation of features with price:")
print(df[["price", "area_sqft", "bedrooms", "bathrooms", "year_built", "avg_income_lpa", "schools_nearby"]]
      .corr()["price"].sort_values(ascending=False).round(3))

df.to_csv("RealEstate_Prices_wrangled.csv", index=False)
print("\nFinal shape:", df.shape, "-> saved RealEstate_Prices_wrangled.csv")
