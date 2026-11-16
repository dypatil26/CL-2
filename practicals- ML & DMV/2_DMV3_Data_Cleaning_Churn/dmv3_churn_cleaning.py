import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

print("=== Step 1: Import the Dataset ===")
df = pd.read_csv(upload_file("Telecom_Customer_Churn.csv"))
print(df.head())

print("\n=== Step 2: Explore Structure and Content ===")
print("Shape:", df.shape)
print(df.dtypes)
print("\nChurn distribution:\n", df["Churn"].value_counts())

print("\n=== Step 3: Handle Missing Values ===")
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)          # blank strings are hidden NaNs
print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())

print("\n=== Step 4: Remove Duplicate Records ===")
print("Duplicates found:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

print("\n=== Step 5: Standardize Inconsistent Data ===")
for c in ["gender", "Partner", "Contract"]:
    print(f"Before {c}: {df[c].unique()}")
df["gender"] = df["gender"].str.strip().str.capitalize()
df["Partner"] = df["Partner"].str.strip().str.lower().map({"yes": "Yes", "y": "Yes", "no": "No"})
df["Contract"] = df["Contract"].str.strip().str.lower().map(
    {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})
for c in ["gender", "Partner", "Contract"]:
    print(f"After  {c}: {df[c].unique()}")

print("\n=== Step 6: Convert Columns to Correct Data Types ===")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"].clip(lower=0))
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].dtypes)

print("\n=== Step 7: Identify and Handle Outliers ===")
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, c in zip(ax, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.boxplot(y=df[c], ax=a)
    a.set_title(c)
plt.suptitle("Outlier check (before)")
plt.tight_layout()
plt.show()
q1, q3 = df["tenure"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
print(f"tenure IQR limits: [{low}, {high}]")
print("Outlier rows:\n", df[(df["tenure"] < low) | (df["tenure"] > high)][["customerID", "tenure"]])
df["tenure"] = df["tenure"].clip(lower=low, upper=72)           # tenure cannot be < 0 or > 72 months
print("tenure range after capping:", df["tenure"].min(), "-", df["tenure"].max())

print("\n=== Step 8: Feature Engineering ===")
df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-1 yr", "1-2 yr", "2-4 yr", "4-6 yr"])
services = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["num_services"] = (df[services] == "Yes").sum(axis=1)
df["avg_charge_per_month"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).round(2)
print(df[["tenure", "tenure_group", "num_services", "avg_charge_per_month"]].head())
print("\nChurn rate by contract:\n", df.groupby("Contract")["Churn"].mean().round(3))
print("\nChurn rate by tenure group:\n", df.groupby("tenure_group", observed=True)["Churn"].mean().round(3))

sns.barplot(data=df, x="Contract", y="Churn")
plt.title("Churn Rate by Contract Type")
plt.tight_layout()
plt.show()

print("\n=== Step 9: Encode and Normalize ===")
model_df = df.drop(columns=["customerID"])
model_df = pd.get_dummies(model_df, drop_first=True, dtype=int)
num = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_charge_per_month"]
model_df[num] = MinMaxScaler().fit_transform(model_df[num])
print("Encoded + scaled shape:", model_df.shape)
print(model_df[num].describe().loc[["min", "max"]].round(2))

print("\n=== Step 10: Train/Test Split ===")
X = model_df.drop("Churn", axis=1)
y = model_df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("Churn ratio train/test:", round(y_train.mean(), 3), "/", round(y_test.mean(), 3))

print("\n=== Step 11: Export Cleaned Dataset ===")
df.to_csv("Telecom_Customer_Churn_cleaned.csv", index=False)
model_df.to_csv("Telecom_Customer_Churn_model_ready.csv", index=False)
print("Saved: Telecom_Customer_Churn_cleaned.csv and Telecom_Customer_Churn_model_ready.csv")
