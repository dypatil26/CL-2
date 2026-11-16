import os
import json
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

print("=== Step 1: Load Sales Data from CSV, Excel and JSON ===")
csv_df = pd.read_csv(upload_file("sales_data.csv"))
xl_df = pd.read_excel(upload_file("sales_data.xlsx"))           # needs openpyxl
with open(upload_file("sales_data.json")) as f:
    raw = json.load(f)
json_df = pd.json_normalize(raw)                                        # flattens nested JSON
print("CSV  :", csv_df.shape, list(csv_df.columns))
print("Excel:", xl_df.shape, list(xl_df.columns))
print("JSON :", json_df.shape, list(json_df.columns))

print("\n=== Step 2: Explore Structure, Missing Values and Quality Issues ===")
for name, d in [("CSV", csv_df), ("Excel", xl_df), ("JSON", json_df)]:
    print(f"\n--- {name} ---")
    print(d.head(3))
    print("Missing values:\n", d.isnull().sum()[d.isnull().sum() > 0])
    print("Duplicate rows:", d.duplicated().sum())
print("\nCategory spellings in CSV:", csv_df["product_category"].unique())

print("\n=== Step 3: Convert to a Unified Format (same column names) ===")
xl_df = xl_df.rename(columns={"Order ID": "order_id", "Order Date": "order_date", "Customer Name": "customer_name",
                              "Region": "region", "Category": "product_category", "Qty": "quantity",
                              "Unit Price": "unit_price"})
json_df = json_df.rename(columns={"customer.name": "customer_name", "customer.region": "region",
                                  "item.category": "product_category", "item.quantity": "quantity",
                                  "item.unit_price": "unit_price"})
csv_df["order_date"] = pd.to_datetime(csv_df["order_date"], format="%d-%m-%Y")
xl_df["order_date"] = pd.to_datetime(xl_df["order_date"])
json_df["order_date"] = pd.to_datetime(json_df["order_date"], format="%Y-%m-%d")
cols = ["order_id", "order_date", "customer_name", "region", "product_category", "quantity", "unit_price"]
sales = pd.concat([csv_df[cols], xl_df[cols], json_df[cols]], ignore_index=True)   # merging datasets
print("Combined shape:", sales.shape)

print("\n=== Step 4: Data Cleaning ===")
before = len(sales)
sales = sales.drop_duplicates()
print("Duplicates removed:", before - len(sales))
sales["product_category"] = sales["product_category"].str.strip().str.title()      # 'electronics ' -> 'Electronics'
sales["quantity"] = sales["quantity"].fillna(sales["quantity"].median()).astype(int)
sales["unit_price"] = sales.groupby("product_category")["unit_price"].transform(lambda s: s.fillna(s.median()))
print("Missing values after cleaning:\n", sales.isnull().sum())

print("\n=== Step 5: Data Transformation (derived variables, splitting columns) ===")
sales["total_sales"] = sales["quantity"] * sales["unit_price"]
sales["month"] = sales["order_date"].dt.month_name().str[:3]
sales["quarter"] = "Q" + sales["order_date"].dt.quarter.astype(str)
sales[["first_name", "last_name"]] = sales["customer_name"].str.split(" ", n=1, expand=True)
print(sales.head())

print("\n=== Step 6: Analysis ===")
print("Descriptive statistics:\n", sales[["quantity", "unit_price", "total_sales"]].describe().round(2))
print("\nTotal sales            : Rs", round(sales["total_sales"].sum(), 2))
print("Average order value    : Rs", round(sales["total_sales"].mean(), 2))
print("Number of orders       :", sales["order_id"].nunique())
print("\nSales by region:\n", sales.groupby("region")["total_sales"].sum().round(2))
print("\nSales by category:\n", sales.groupby("product_category")["total_sales"].sum().sort_values(ascending=False).round(2))
print("\nCategory distribution (orders):\n", sales["product_category"].value_counts(normalize=True).round(3))

print("\n=== Step 7: Visualizations ===")
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sales.groupby("product_category")["total_sales"].sum().sort_values().plot(kind="barh", ax=ax[0, 0], color="steelblue")
ax[0, 0].set_title("Total Sales by Category")
ax[0, 0].set_xlabel("Sales (Rs)")
sales.groupby("region")["total_sales"].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax[0, 1])
ax[0, 1].set_title("Sales Share by Region")
ax[0, 1].set_ylabel("")
sales.boxplot(column="total_sales", by="product_category", ax=ax[1, 0], rot=30)
ax[1, 0].set_title("Order Value Distribution by Category")
ax[1, 0].set_xlabel("")
monthly = sales.set_index("order_date").resample("ME")["total_sales"].sum()
ax[1, 1].plot(monthly.index, monthly.values, marker="o")
ax[1, 1].set_title("Monthly Sales Trend")
ax[1, 1].tick_params(axis="x", rotation=45)
plt.suptitle("Sales Analysis from CSV + Excel + JSON")
plt.tight_layout()
plt.show()

sales.to_csv("cleaned_combined_sales.csv", index=False)
print("Cleaned data saved to cleaned_combined_sales.csv")
