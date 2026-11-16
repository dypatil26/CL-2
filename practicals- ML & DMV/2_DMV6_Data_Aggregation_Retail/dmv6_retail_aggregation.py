import os
import pandas as pd
import matplotlib.pyplot as plt

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

print("=== Step 1: Import Dataset ===")
df = pd.read_csv(upload_file("Retail_Sales_Data.csv"))
print(df.head())

print("\n=== Step 2: Explore Structure and Content ===")
print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing values:", int(df.isnull().sum().sum()))
print("\nSummary:\n", df.describe().round(2))
df["Date"] = pd.to_datetime(df["Date"])

print("\n=== Step 3: Relevant Variables ===")
print("Group by -> Region, Product Category | Measure -> Total Amount, Quantity")
print("Regions   :", df["Region"].unique())
print("Categories:", df["Product Category"].unique())

print("\n=== Step 4: Total Sales by Region (groupby + sum) ===")
region_sales = df.groupby("Region")["Total Amount"].sum().sort_values(ascending=False)
print(region_sales)
print("\nMultiple aggregations per region:")
print(df.groupby("Region").agg(total_sales=("Total Amount", "sum"), avg_sale=("Total Amount", "mean"),
                               transactions=("Transaction ID", "count"), units=("Quantity", "sum")).round(2))

print("\n=== Step 5: Visualize Sales Distribution by Region ===")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
region_sales.plot(kind="bar", ax=ax[0], color="cornflowerblue", edgecolor="k")
ax[0].set_title("Total Sales by Region")
ax[0].set_ylabel("Sales Amount")
for i, v in enumerate(region_sales.values):
    ax[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom")
ax[1].pie(region_sales, labels=region_sales.index, autopct="%1.1f%%", startangle=90)
ax[1].set_title("Sales Share by Region")
plt.tight_layout()
plt.show()

print("\n=== Step 6: Top-performing Regions ===")
top = region_sales.head(3)
print("Top 3 regions:\n", top)
print(f"\nBest region: {region_sales.idxmax()} with sales {region_sales.max():,.0f} "
      f"({region_sales.max() / region_sales.sum() * 100:.1f}% of total)")

print("\n=== Step 7: Sales by Region AND Product Category ===")
rc = df.groupby(["Region", "Product Category"])["Total Amount"].sum().unstack()
print(rc)
print("\nSame using pivot_table:")
pivot = pd.pivot_table(df, values="Total Amount", index="Region", columns="Product Category", aggfunc="sum", margins=True)
print(pivot)
print("\nBest category in each region:\n", rc.idxmax(axis=1))

print("\n=== Step 8: Stacked and Grouped Bar Plots ===")
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
rc.plot(kind="bar", stacked=True, ax=ax[0], colormap="tab20")
ax[0].set_title("Stacked: Sales by Region and Category")
ax[0].set_ylabel("Sales Amount")
rc.plot(kind="bar", ax=ax[1], colormap="tab20")
ax[1].set_title("Grouped: Sales by Region and Category")
plt.tight_layout()
plt.show()

print("\n=== Extra: Monthly Sales Trend by Region ===")
monthly = df.groupby([df["Date"].dt.to_period("M"), "Region"])["Total Amount"].sum().unstack()
print(monthly.head())
monthly.plot(figsize=(12, 4), marker="o", title="Monthly Sales by Region")
plt.ylabel("Sales Amount")
plt.tight_layout()
plt.show()
