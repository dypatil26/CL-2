import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
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
df = pd.read_csv(upload_file("Stock_Prices.csv"))
print(df.head())

print("\n=== Step 2: Explore Structure and Content ===")
print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing values:", int(df.isnull().sum().sum()))
print(df.describe().round(2))

print("\n=== Step 3: Convert Date Column to datetime and set as index ===")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")
print("Date range:", df.index.min().date(), "to", df.index.max().date())
print("Index type:", type(df.index).__name__)

print("\n=== Step 4: Historical Price Trend ===")
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(df.index, df["Close"], label="Close")
ax[0].set_title("Closing Price over Time")
ax[0].set_ylabel("Price")
ax[1].bar(df.index, df["Volume"], color="grey")
ax[1].set_title("Trading Volume")
plt.tight_layout()
plt.show()

print("\n=== Step 5: Moving / Rolling Averages ===")
df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_50"] = df["Close"].rolling(50).mean()
df["MA_200"] = df["Close"].rolling(200).mean()
print(df[["Close", "MA_20", "MA_50", "MA_200"]].dropna().tail())
plt.figure(figsize=(12, 5))
plt.plot(df["Close"], alpha=0.5, label="Close")
plt.plot(df["MA_20"], label="20-day MA")
plt.plot(df["MA_50"], label="50-day MA")
plt.plot(df["MA_200"], label="200-day MA")
plt.title("Moving Averages smooth out noise and show the trend")
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== Step 6: Seasonality Analysis ===")
df["Return"] = df["Close"].pct_change() * 100
df["Weekday"] = df.index.day_name()
df["Month"] = df.index.month
print("Average daily return by weekday (%):\n",
      df.groupby("Weekday")["Return"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]).round(3))
print("\nAverage daily return by month (%):\n", df.groupby("Month")["Return"].mean().round(3))
monthly = df["Close"].resample("ME").mean()
decomp = seasonal_decompose(monthly, model="multiplicative", period=12)
decomp.plot().set_size_inches(11, 8)
plt.suptitle("Seasonal Decomposition of Monthly Close Price", y=1.0)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(x=df["Month"], y=df["Return"], ax=ax[0])
ax[0].set_title("Daily Return by Month")
sns.boxplot(x=df["Weekday"], y=df["Return"], order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], ax=ax[1])
ax[1].set_title("Daily Return by Weekday")
plt.tight_layout()
plt.show()

print("\n=== Step 7: Correlation with Volume and Market Index ===")
df["Index_Return"] = df["Market_Index"].pct_change() * 100
corr = df[["Close", "Volume", "Market_Index", "Return", "Index_Return"]].corr()
print(corr.round(3))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 5))
plt.scatter(df["Index_Return"], df["Return"], alpha=0.4)
plt.xlabel("Market Index daily return (%)")
plt.ylabel("Stock daily return (%)")
plt.title("Stock vs Market Returns")
plt.tight_layout()
plt.show()

print("\n=== Step 8: Forecasting with ARIMA and Exponential Smoothing ===")
series = df["Close"].asfreq("B").ffill()
print("ADF test p-value (raw prices)  :", round(adfuller(series)[1], 4), "-> > 0.05 means NOT stationary")
print("ADF test p-value (1st diff)    :", round(adfuller(series.diff().dropna())[1], 4), "-> stationary, so d = 1")
train, test = series[:-60], series[-60:]

arima = ARIMA(train, order=(5, 1, 0)).fit()
arima_fc = arima.forecast(steps=len(test))
es = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
es_fc = es.forecast(len(test))

for name, fc in [("ARIMA(5,1,0)", arima_fc), ("Exponential Smoothing", es_fc)]:
    rmse = np.sqrt(mean_squared_error(test, fc))
    mape = np.mean(np.abs((test - fc) / test)) * 100
    print(f"{name:<22} RMSE = {rmse:8.2f}   MAE = {mean_absolute_error(test, fc):8.2f}   MAPE = {mape:.2f}%")

future = ARIMA(series, order=(5, 1, 0)).fit().get_forecast(steps=30)
ci = future.conf_int()
plt.figure(figsize=(12, 5))
plt.plot(series[-250:], label="History")
plt.plot(test.index, arima_fc, label="ARIMA forecast (test)", color="red")
plt.plot(test.index, es_fc, label="Exp. Smoothing (test)", color="green")
plt.plot(future.predicted_mean.index, future.predicted_mean, label="Next 30 days", color="purple")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="purple", alpha=0.15, label="95% confidence")
plt.title("Stock Price Forecast")
plt.legend()
plt.tight_layout()
plt.show()
print("\nNext 5 forecasted closing prices:\n", future.predicted_mean.head().round(2))
