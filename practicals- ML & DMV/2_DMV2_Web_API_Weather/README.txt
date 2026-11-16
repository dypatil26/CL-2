============================================================
DMV2 - INTERACTING WITH WEB APIs:
ANALYZING WEATHER DATA FROM OPENWEATHERMAP
============================================================

## PROBLEM STATEMENT

Analyzing Weather Data from OpenWeatherMap API.
Interact with the OpenWeatherMap API to retrieve weather
data for specific locations and perform data modeling and
visualization to analyze weather patterns over time.

Tasks:
1. Register and obtain API key from OpenWeatherMap.
2. Use the API key to retrieve weather data.
3. Extract temperature, humidity, wind speed,
   precipitation.
4. Clean and pre-process data.
5. Data modeling: average, max/min, trends.
6. Visualize: line, bar, scatter plots.
7. Aggregate by time period (daily).
8. Geographical visualization across locations.
9. Correlation plots / heatmaps between attributes.

============================================================
DATASET
============================================================

Live data: OpenWeatherMap 5 day / 3 hour forecast API
URL: https://api.openweathermap.org/data/2.5/forecast

Cities: Pune, Mumbai, Delhi, Bengaluru, Chennai
(40 records per city = 5 days x 8 per day)

Offline backup: weather_sample.json (in this folder)
* Same JSON structure as the real API response.
* Used automatically when no API key is given or there
  is no internet - so the practical always runs in lab.

============================================================
HOW TO GET THE API KEY
============================================================

1. Go to https://openweathermap.org/ and Sign Up (free).
2. Open your profile -> "My API keys".
3. Copy the default key (activation may take 10 min - 2 h).
4. Either edit the program:
       API_KEY = "your_key_here"
   or set environment variable:
       Windows : set OWM_API_KEY=your_key_here
       Linux   : export OWM_API_KEY=your_key_here

============================================================
OBJECTIVE
============================================================

* To understand Web APIs, HTTP requests and JSON.
* To retrieve live data using an API key.
* To clean, model, aggregate and visualize weather data.

============================================================
LIBRARIES USED
============================================================

1. requests    - send HTTP GET request to API
2. json        - read JSON
3. pandas      - DataFrame, cleaning, aggregation
4. matplotlib  - plots, geographic scatter
5. seaborn     - box plot, heatmap
6. os          - environment variable, paths

============================================================
HOW TO RUN
============================================================

pip install requests pandas matplotlib seaborn

python dmv2_weather_api.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  weather_sample.json (asked only when no API key
  is set or the API call fails)
(The copy is included in this folder, but you can keep it
anywhere.)

* Jupyter Notebook / Python on your computer
    -> a file-picker window opens; select the file.
       (If you don't see it, check behind the browser.)
* Google Colab
    -> click "Choose files" under the cell and pick it.
* If no window can open
    -> type the file's full path when asked.

This works the same whether you paste the whole program
into one cell or paste it block by block (each block of
code separated by an empty line), as long as the blocks
are run in order. The upload_file() function near the top
is the code that does this.

============================================================
THEORY
============================================================

## API (Application Programming Interface)

An API is a set of rules that lets two programs talk to
each other. A Web API is accessed over the internet using
HTTP.

------------------------------------------------------------

## REST API

* Uses HTTP methods: GET (read), POST (create),
  PUT (update), DELETE (delete)
* Resources identified by URLs (endpoints)
* Data usually returned in JSON
* Stateless - every request has all needed info

------------------------------------------------------------

## Request and Response

Request:
GET https://api.openweathermap.org/data/2.5/forecast
    ?q=Pune&appid=KEY&units=metric

* Endpoint: /data/2.5/forecast
* Query parameters: q (city), appid (key), units

Response: JSON + HTTP status code
* 200 OK
* 401 Unauthorized (invalid/inactive key)
* 404 Not Found (wrong city)
* 429 Too Many Requests (rate limit)

------------------------------------------------------------

## API Key

A unique code that identifies and authenticates the user,
tracks usage and enforces rate limits. Keep it secret.

------------------------------------------------------------

## OpenWeatherMap API

Online service giving current, forecast and historical
weather data for any location. Free plan: 60 calls/min.

Important JSON fields:
list[i].dt_txt          - date-time
list[i].main.temp       - temperature
list[i].main.humidity   - humidity (%)
list[i].main.pressure   - pressure (hPa)
list[i].wind.speed      - wind speed (m/s)
list[i].rain["3h"]      - rain in last 3 h (only if rain)
city.coord.lat / lon    - location

------------------------------------------------------------

## How to interact with Web APIs in Python

1. Get API key
2. Build URL and parameters
3. requests.get(url, params=...)
4. Check status (r.raise_for_status())
5. Parse JSON (r.json())
6. Extract fields into a DataFrame

------------------------------------------------------------

## Data Modeling and Aggregation

* Summary statistics: mean, max, min per city
* Rolling mean for trend
* Aggregation by period using groupby(date) or resample

------------------------------------------------------------

## Correlation Heatmap

Shows Pearson correlation between all pairs of variables
as coloured cells. Temperature and humidity are usually
negatively correlated.

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1 - Key and settings

API_KEY = os.environ.get("OWM_API_KEY", "")
CITIES = ["Pune", "Mumbai", ...]
URL = "https://api.openweathermap.org/data/2.5/forecast"

------------------------------------------------------------

## CODE BLOCK 2 - API call

def fetch_city(city):
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    r = requests.get(URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

* Sends GET request; units=metric gives deg C.
* raise_for_status() raises error for 401/404 etc.
* If call fails or no key, weather_sample.json is used.

------------------------------------------------------------

## CODE BLOCK 3 - Extraction

for item in data["list"]:
    rows.append({"temperature": item["main"].get("temp"),
                 "humidity": ...,
                 "wind_speed": item.get("wind", {}).get("speed"),
                 "precipitation": item.get("rain", {}).get("3h", 0.0),
                 ...})

* .get() avoids KeyError when a field is missing
  (rain key exists only when it rains -> default 0).

------------------------------------------------------------

## CODE BLOCK 4 - Cleaning

pd.to_datetime(df["datetime"])
groupby("city")["humidity"].transform(
    lambda s: s.interpolate().bfill().ffill())
drop_duplicates()

* Missing humidity filled by interpolation within the
  same city (time-series aware).

------------------------------------------------------------

## CODE BLOCK 5 - Modeling

groupby("city").agg(avg_temp=("temperature","mean"),
                    max_temp=..., min_temp=..., total_rain=...)
pune["temperature"].rolling(8).mean()   # 24-hour trend

------------------------------------------------------------

## CODE BLOCK 6 - Visualization

Line: temperature over time per city
Bar : total precipitation per city
Scatter: temperature vs humidity (colour = wind)
Box : wind speed variation per city

------------------------------------------------------------

## CODE BLOCK 7 - Daily aggregation

groupby(["city", "date"]).agg(...)
unstack(0).plot(kind="bar")

------------------------------------------------------------

## CODE BLOCK 8 - Geographic plot

plt.scatter(lon, lat, c=temp, s=temp*25, cmap="coolwarm")

* Cities placed at real longitude/latitude, colour and
  size = average temperature.

------------------------------------------------------------

## CODE BLOCK 9 - Correlation

df[[...]].corr(); sns.heatmap(corr, annot=True)

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. requests.get(url, params, timeout)
2. response.raise_for_status(), response.json()
3. json.load(), json.dumps()
4. dict.get()
5. pd.DataFrame(), pd.to_datetime()
6. interpolate(), bfill(), ffill()
7. groupby().agg(), rolling().mean()
8. unstack()
9. plt.plot(), plt.scatter(), sns.boxplot(), sns.heatmap()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Get API key
     ↓
requests.get() for each city ──(fail)──> weather_sample.json
     ↓
JSON response
     ↓
Extract fields → DataFrame
     ↓
Clean (datetime, interpolate missing, duplicates)
     ↓
Statistics + rolling trend
     ↓
Line / bar / scatter / box plots
     ↓
Daily aggregation
     ↓
Geographic plot + correlation heatmap

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed (offline sample):

city       avg_temp max_temp min_temp avg_humidity total_rain
Bengaluru   24.14    30.65    18.81     70.08        8.75
Chennai     30.25    35.88    23.44     70.00        7.72
Delhi       21.92    27.38    16.72     70.91       18.59
Mumbai      29.01    35.16    23.39     68.70        2.65
Pune        26.92    32.23    21.38     68.35       18.11

Correlation temperature vs humidity = -0.67
(as temperature rises in the afternoon, humidity drops)

With a real key, the numbers are the live forecast.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-3 are from the lab manual)

1. What is OpenWeatherMap?

Ans:
An online service that provides current, forecast and
historical weather data for any location through Web APIs.

------------------------------------------------------------

2. What is Web API?

Ans:
An interface that allows programs to communicate over the
internet using HTTP, usually returning JSON or XML.

------------------------------------------------------------

3. How can we interact with Web APIs?

Ans:
Get an API key, send an HTTP request (requests.get) to the
endpoint with parameters, check the status code, parse the
JSON response and convert it to a DataFrame.

------------------------------------------------------------

4. What is an API key?

Ans:
A unique identifier used to authenticate requests and
track usage.

------------------------------------------------------------

5. What is JSON?

Ans:
JavaScript Object Notation - lightweight key-value text
format for data exchange.

------------------------------------------------------------

6. What is REST?

Ans:
Representational State Transfer - architectural style for
Web APIs using HTTP methods on URL resources.

------------------------------------------------------------

7. Name HTTP methods.

Ans:
GET, POST, PUT, PATCH, DELETE.

------------------------------------------------------------

8. What do status codes 200, 401, 404 mean?

Ans:
200 success, 401 unauthorized (bad key), 404 not found.

------------------------------------------------------------

9. What is rate limiting?

Ans:
Limit on number of API calls in a time period.

------------------------------------------------------------

10. Why use dict.get() for rain?

Ans:
"rain" key is present only when it rains; get() returns a
default (0) instead of raising KeyError.

------------------------------------------------------------

11. What does units=metric do?

Ans:
Returns temperature in Celsius and wind in m/s (default is
Kelvin).

------------------------------------------------------------

12. What does the correlation heatmap show?

Ans:
Strength and direction of linear relationship between each
pair of weather attributes.

============================================================
CONCLUSION
============================================================

Weather data for five Indian cities was retrieved from the
OpenWeatherMap Web API (or an identical offline sample),
extracted from JSON, cleaned, modelled, aggregated daily
and visualized with line, bar, scatter, geographic and
correlation plots. Hence Web APIs can be used to obtain
live data for analysis and visualization.

============================================================
END OF FILE
============================================================
