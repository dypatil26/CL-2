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
df = pd.read_csv(upload_file("City_Air_Quality.csv"))
print(df.head())

print("\n=== Step 2: Explore Structure and Content ===")
print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe().round(2))

print("\n=== Step 3: Identify Relevant Variables ===")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")
df["PM10"] = df["PM10"].interpolate()
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
print("Time variable: Date | Pollutants:", pollutants, "| Target: AQI, AQI_Bucket")

print("\n=== Step 4: Overall AQI Trend (line / time-series plot) ===")
plt.figure(figsize=(12, 4))
plt.plot(df.index, df["AQI"], color="grey", alpha=0.5, label="Daily AQI")
plt.plot(df.index, df["AQI"].rolling(30).mean(), color="red", lw=2, label="30-day moving average")
plt.axhline(100, ls="--", color="green", label="Satisfactory limit (100)")
plt.title("AQI Trend over Time - Pune")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== Step 5: Individual Pollutant Trends (separate line plots) ===")
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
colors = ["tab:red", "tab:orange", "tab:blue", "tab:purple", "tab:brown", "tab:green"]
for ax, p, c in zip(axes.ravel(), pollutants, colors):
    ax.plot(df.index, df[p], color=c, lw=0.8)
    ax.set_title(p)
    ax.grid(alpha=0.3)
fig.suptitle("Pollutant Levels over Time")
plt.tight_layout()
plt.show()

print("\n=== Step 6: Compare AQI across Time Periods (bar / stacked bar) ===")
monthly = df["AQI"].resample("ME").mean()
print("Monthly average AQI:\n", monthly.round(1).head(12))
plt.figure(figsize=(12, 4))
plt.bar(monthly.index.strftime("%b-%y"), monthly.values, color=["red" if v > 100 else "green" for v in monthly.values])
plt.xticks(rotation=60)
plt.title("Monthly Average AQI (red = above 100)")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

bucket_month = pd.crosstab(df.index.to_period("M"), df["AQI_Bucket"])
order = [b for b in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"] if b in bucket_month.columns]
bucket_month[order].plot(kind="bar", stacked=True, figsize=(12, 4),
                         color=["green", "yellowgreen", "gold", "orange", "red", "darkred"][:len(order)])
plt.title("Number of Days in Each AQI Category per Month (stacked)")
plt.ylabel("Days")
plt.tight_layout()
plt.show()

print("\n=== Step 7: Distribution of AQI (box / violin plots) ===")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
data = [df.loc[df["AQI_Bucket"] == b, "AQI"] for b in order]
ax[0].boxplot(data, labels=order)
ax[0].set_title("AQI Distribution by Category (box plot)")
ax[0].tick_params(axis="x", rotation=30)
seasons = df.index.month.map(lambda m: "Winter" if m in (11, 12, 1, 2) else "Summer" if m in (3, 4, 5) else "Monsoon")
s_order = ["Winter", "Summer", "Monsoon"]
ax[1].violinplot([df.loc[seasons == s, "AQI"] for s in s_order], showmedians=True)
ax[1].set_xticks([1, 2, 3])
ax[1].set_xticklabels(s_order)
ax[1].set_title("AQI Distribution by Season (violin plot)")
plt.tight_layout()
plt.show()

print("\n=== Step 8: AQI vs Pollutants (scatter / bubble chart) ===")
print("Correlation with AQI:\n", df[pollutants + ["AQI"]].corr()["AQI"].round(3))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(df["PM2.5"], df["AQI"], alpha=0.5, c="tab:red", label="PM2.5")
ax[0].scatter(df["PM10"], df["AQI"], alpha=0.3, c="tab:blue", label="PM10")
ax[0].set_xlabel("Concentration (ug/m3)")
ax[0].set_ylabel("AQI")
ax[0].set_title("AQI vs Particulate Matter")
ax[0].legend()
b = ax[1].scatter(df["NO2"], df["AQI"], s=df["CO"] * 120, c=df["PM2.5"], cmap="plasma", alpha=0.6, edgecolors="k")
ax[1].set_xlabel("NO2")
ax[1].set_ylabel("AQI")
ax[1].set_title("Bubble chart: size = CO, colour = PM2.5")
plt.colorbar(b, ax=ax[1], label="PM2.5")
plt.tight_layout()
plt.show()

print("\n=== Step 9: Summary ===")
print("Worst 5 days:\n", df["AQI"].nlargest(5))
print("\nDays per category:\n", df["AQI_Bucket"].value_counts())
