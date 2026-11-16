import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# ------------------------------------------------------------------
# Step 1: API key  (register free at https://openweathermap.org/api)
# Put your key here OR set environment variable OWM_API_KEY.
# If no key / no internet, the program uses weather_sample.json,
# which has exactly the same JSON format as the real API response.
# ------------------------------------------------------------------
API_KEY = os.environ.get("OWM_API_KEY", "")          # e.g. "a1b2c3d4e5..."
CITIES = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai"]
URL = "https://api.openweathermap.org/data/2.5/forecast"   # 5 day / 3 hour forecast


def fetch_city(city):
    """Step 2: call the Web API with the key and return JSON."""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    r = requests.get(URL, params=params, timeout=10)
    r.raise_for_status()                 # raises error for 401 (bad key), 404 (bad city) ...
    return r.json()


print("=== Step 1-2: Retrieve Weather Data from OpenWeatherMap API ===")
responses = {}
if API_KEY:
    try:
        for c in CITIES:
            responses[c] = fetch_city(c)
            print(f"Fetched {c}: HTTP 200, {responses[c]['cnt']} records")
    except Exception as e:
        print("API call failed:", e)
        responses = {}
if not responses:
    print("Using offline sample (weather_sample.json) - same format as API response")
    with open(upload_file("weather_sample.json")) as f:
        responses = json.load(f)

print("\nSample raw JSON record:\n", json.dumps(responses[CITIES[0]]["list"][0], indent=2)[:600])

print("\n=== Step 3: Extract Relevant Attributes ===")
rows = []
for city, data in responses.items():
    lat, lon = data["city"]["coord"]["lat"], data["city"]["coord"]["lon"]
    for item in data["list"]:
        rows.append({
            "city": city, "lat": lat, "lon": lon,
            "datetime": item.get("dt_txt"),
            "temperature": item["main"].get("temp"),
            "humidity": item["main"].get("humidity"),
            "pressure": item["main"].get("pressure"),
            "wind_speed": item.get("wind", {}).get("speed"),
            "precipitation": item.get("rain", {}).get("3h", 0.0),   # missing key = no rain
            "condition": item["weather"][0]["main"],
        })
df = pd.DataFrame(rows)
print(df.head())
print("Shape:", df.shape)

print("\n=== Step 4: Clean and Pre-process ===")
print("Missing values before:\n", df.isnull().sum()[df.isnull().sum() > 0])
df["datetime"] = pd.to_datetime(df["datetime"])
df["humidity"] = df.groupby("city")["humidity"].transform(lambda s: s.interpolate().bfill().ffill())
df = df.drop_duplicates()
df["date"] = df["datetime"].dt.date
print("Missing values after:", int(df.isnull().sum().sum()))

print("\n=== Step 5: Data Modeling - Weather Statistics ===")
stats = df.groupby("city").agg(avg_temp=("temperature", "mean"), max_temp=("temperature", "max"),
                               min_temp=("temperature", "min"), avg_humidity=("humidity", "mean"),
                               avg_wind=("wind_speed", "mean"), total_rain=("precipitation", "sum")).round(2)
print(stats)
pune = df[df["city"] == CITIES[0]].set_index("datetime")
trend = pune["temperature"].rolling(8).mean()
print(f"\n{CITIES[0]} 24-hour rolling mean temperature (last 3):\n", trend.dropna().tail(3).round(2))

print("\n=== Step 6: Visualizations ===")
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for c in df["city"].unique():
    s = df[df["city"] == c]
    ax[0, 0].plot(s["datetime"], s["temperature"], label=c)
ax[0, 0].set_title("Temperature over Time")
ax[0, 0].set_ylabel("Temp (deg C)")
ax[0, 0].legend()
ax[0, 0].tick_params(axis="x", rotation=30)
stats["total_rain"].plot(kind="bar", ax=ax[0, 1], color="royalblue", title="Total Precipitation (mm)")
ax[1, 0].scatter(df["temperature"], df["humidity"], c=df["wind_speed"], cmap="viridis")
ax[1, 0].set_xlabel("Temperature")
ax[1, 0].set_ylabel("Humidity")
ax[1, 0].set_title("Temperature vs Humidity (colour = wind speed)")
sns.boxplot(data=df, x="city", y="wind_speed", ax=ax[1, 1])
ax[1, 1].set_title("Wind Speed Variation")
plt.tight_layout()
plt.show()

print("\n=== Step 7: Aggregation by Time Period (daily) ===")
daily = df.groupby(["city", "date"]).agg(avg_temp=("temperature", "mean"), max_temp=("temperature", "max"),
                                        rain=("precipitation", "sum")).round(2)
print(daily.head(10))
daily["avg_temp"].unstack(0).plot(kind="bar", figsize=(10, 4), title="Daily Average Temperature by City")
plt.ylabel("deg C")
plt.tight_layout()
plt.show()

print("\n=== Step 8: Geographical Visualization ===")
geo = df.groupby("city").agg(lat=("lat", "first"), lon=("lon", "first"), temp=("temperature", "mean")).reset_index()
plt.figure(figsize=(6, 7))
sc = plt.scatter(geo["lon"], geo["lat"], c=geo["temp"], s=geo["temp"] * 25, cmap="coolwarm", edgecolors="k")
for _, r in geo.iterrows():
    plt.annotate(f"{r['city']}\n{r['temp']:.1f}C", (r["lon"], r["lat"]), xytext=(6, 6), textcoords="offset points")
plt.colorbar(sc, label="Avg Temperature")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Average Temperature across Indian Cities")
plt.tight_layout()
plt.show()

print("\n=== Step 9: Relationships between Attributes (Correlation Heatmap) ===")
corr = df[["temperature", "humidity", "pressure", "wind_speed", "precipitation"]].corr()
print(corr.round(2))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Weather Attributes")
plt.tight_layout()
plt.show()
