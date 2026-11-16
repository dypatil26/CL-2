============================================================
DMV7 - TIME SERIES DATA ANALYSIS:
ANALYSIS AND VISUALIZATION OF STOCK MARKET DATA
============================================================

## PROBLEM STATEMENT

Analysis and Visualization of Stock Market Data.
The dataset contains historical stock prices (date,
closing price, volume ...). Perform time series analysis
to identify trends, patterns and predictors, and build
models to forecast future stock prices.

Tasks:
1. Import "Stock_Prices.csv".
2. Explore structure and content.
3. Ensure the date column is datetime.
4. Line / time series plots of historical prices.
5. Moving / rolling averages.
6. Seasonality analysis (weekly, monthly, yearly).
7. Correlation with trading volume / market index.
8. ARIMA or exponential smoothing forecast.

============================================================
DATASET
============================================================

Stock_Prices.csv  (included, 1044 trading days, 2021-2024)

Columns:
Date, Open, High, Low, Close, Adj Close, Volume,
Market_Index

* Same format as Yahoo Finance / Kaggle stock CSVs.
* Market_Index = a market index (like NIFTY) to study
  correlation.
* Any real stock CSV downloaded from Yahoo Finance
  (Date, Open, High, Low, Close, Adj Close, Volume) will
  work - only the Market_Index step needs that column.

============================================================
OBJECTIVE
============================================================

* To handle date-time indexed data.
* To visualize trends with moving averages.
* To analyze seasonality and correlation.
* To forecast using ARIMA and Exponential Smoothing.

============================================================
LIBRARIES USED
============================================================

1. pandas, numpy           - time-series handling
2. matplotlib, seaborn     - plots
3. statsmodels
   * seasonal_decompose    - trend/seasonal/residual
   * adfuller              - stationarity test
   * ARIMA                 - ARIMA model
   * ExponentialSmoothing  - Holt's method
4. sklearn.metrics         - RMSE, MAE

============================================================
HOW TO RUN
============================================================

pip install pandas numpy matplotlib seaborn statsmodels scikit-learn

python dmv7_stock_time_series.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  Stock_Prices.csv
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

## Time Series

A sequence of data points recorded at successive, equally
spaced points in time (daily stock price, monthly sales).

## Time Series Analysis

Studying time series data to find patterns (trend,
seasonality, cycles) and to forecast future values.

------------------------------------------------------------

## Components of a Time Series

1. Trend       - long-term increase/decrease
2. Seasonality - regular repeating pattern
                 (weekly, monthly, yearly)
3. Cyclic      - long irregular up/down waves
4. Residual    - random noise

Additive:        Y = T + S + R
Multiplicative:  Y = T x S x R

------------------------------------------------------------

## Moving Average

Average of last n values; smooths noise.
MA_20 (short-term), MA_50, MA_200 (long-term).
When MA_50 crosses above MA_200 -> "golden cross" (bullish).

------------------------------------------------------------

## Stationarity

A series is stationary if its mean and variance do not
change over time. ARIMA needs stationarity.
Stock prices are NOT stationary; daily differences are.

ADF (Augmented Dickey-Fuller) test:
p-value < 0.05 -> stationary.

------------------------------------------------------------

## ARIMA (p, d, q)

AutoRegressive Integrated Moving Average.
* AR(p): uses p past values
* I(d) : differencing d times to make stationary
* MA(q): uses q past forecast errors

ARIMA(5,1,0) = 5 lags, 1 difference, no MA term.

------------------------------------------------------------

## Exponential Smoothing

Forecast = weighted average of past values where weights
decrease exponentially (recent values matter more).
* Simple: level only
* Holt  : level + trend (used here)
* Holt-Winters: level + trend + seasonality

------------------------------------------------------------

## Forecast Evaluation

RMSE = sqrt(mean((actual - forecast)^2))
MAE  = mean(|actual - forecast|)
MAPE = mean(|actual - forecast| / actual) x 100 %

------------------------------------------------------------

## Why Data Visualization is Important

* Reveals trends, patterns and outliers quickly
* Makes complex data easy to understand
* Helps decision making and communication
* Detects data quality problems

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1-2 - Import and explore

df = pd.read_csv(...); df.describe()

------------------------------------------------------------

## CODE BLOCK 3 - Datetime index

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

* DatetimeIndex enables resample, rolling, date parts.

------------------------------------------------------------

## CODE BLOCK 4 - Price trend

subplots: Close price (top) + Volume bars (bottom)

------------------------------------------------------------

## CODE BLOCK 5 - Moving averages

df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_50"], df["MA_200"]

------------------------------------------------------------

## CODE BLOCK 6 - Seasonality

df["Return"] = df["Close"].pct_change() * 100
groupby("Weekday")["Return"].mean()
groupby("Month")["Return"].mean()
seasonal_decompose(monthly, model="multiplicative",
                   period=12)
box plots of returns by month and weekday

------------------------------------------------------------

## CODE BLOCK 7 - Correlation

df["Index_Return"] = df["Market_Index"].pct_change()*100
df[["Close","Volume","Market_Index","Return",
    "Index_Return"]].corr()
scatter of stock return vs market return

------------------------------------------------------------

## CODE BLOCK 8 - Forecasting

series = df["Close"].asfreq("B").ffill()
adfuller(series)          -> p = 0.999 (not stationary)
adfuller(series.diff())   -> p = 0.0   (stationary, d=1)
train, test = series[:-60], series[-60:]
ARIMA(train, order=(5,1,0)).fit().forecast(60)
ExponentialSmoothing(train, trend="add").fit().forecast(60)
RMSE, MAE, MAPE for both
ARIMA on full data -> next 30 days with 95% confidence
interval (get_forecast().conf_int())

* asfreq("B") sets business-day frequency.
* Last 60 days kept as test to check accuracy.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. pd.to_datetime(), set_index(), asfreq(), ffill()
2. rolling().mean()
3. pct_change()
4. index.day_name(), index.month
5. resample("ME").mean()
6. seasonal_decompose()
7. corr(), sns.heatmap()
8. adfuller()
9. ARIMA().fit().forecast(), get_forecast().conf_int()
10. ExponentialSmoothing().fit().forecast()
11. mean_squared_error(), mean_absolute_error()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Import CSV → Explore
     ↓
Datetime index
     ↓
Price + volume plot
     ↓
Moving averages (20, 50, 200)
     ↓
Seasonality (weekday/month returns, decomposition)
     ↓
Correlation with volume / market index
     ↓
ADF test → train/test split
     ↓
ARIMA vs Exponential Smoothing → error metrics
     ↓
30-day forecast with confidence band

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

ADF p-value raw prices = 0.999 -> not stationary
ADF p-value 1st diff   = 0.000 -> stationary (d = 1)

Model                  RMSE     MAE     MAPE
ARIMA(5,1,0)          220.22  179.38   6.77%
Exponential Smoothing  91.19   81.93   3.23%

Stock return is strongly correlated with market index
return (the stock moves with the market).
Forecast plot shows next 30 days with a widening 95%
confidence band (uncertainty grows with time).

NOTE: Stock prices are close to a random walk - forecasts
show the expected trend, not exact future prices.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What is data visualization?

Ans:
Graphical representation of data to reveal patterns,
trends and outliers.

------------------------------------------------------------

2. What is time series analysis?

Ans:
Analyzing data points collected over time to identify
trend, seasonality and patterns and to forecast future
values.

------------------------------------------------------------

3. How to perform analysis and visualization of stock
   market data?

Ans:
Load data, convert date to datetime index, plot closing
price and volume, compute moving averages, analyze
returns and seasonality, check correlation with volume
and index, test stationarity and forecast using ARIMA /
exponential smoothing.

------------------------------------------------------------

4. Why is data visualization important?

Ans:
It makes complex data easy to understand, reveals trends
and anomalies quickly, and supports decision making.

------------------------------------------------------------

5. What are components of a time series?

Ans:
Trend, seasonality, cyclic variation and residual (noise).

------------------------------------------------------------

6. What is a moving average?

Ans:
Average of the last n observations, used to smooth data.

------------------------------------------------------------

7. What is stationarity?

Ans:
Constant mean and variance over time.

------------------------------------------------------------

8. What is the ADF test?

Ans:
Augmented Dickey-Fuller test for stationarity; p < 0.05
means stationary.

------------------------------------------------------------

9. Explain p, d, q in ARIMA.

Ans:
p = autoregressive lags, d = differencing order,
q = moving-average lags of errors.

------------------------------------------------------------

10. What is exponential smoothing?

Ans:
Forecasting by weighted average of past values with
exponentially decreasing weights.

------------------------------------------------------------

11. What is seasonal decomposition?

Ans:
Splitting a series into trend, seasonal and residual parts.

------------------------------------------------------------

12. What is MAPE?

Ans:
Mean Absolute Percentage Error - average % error.

------------------------------------------------------------

13. Why is the confidence interval wider for later days?

Ans:
Uncertainty accumulates as we forecast further ahead.

============================================================
CONCLUSION
============================================================

Stock price data was converted to a time series,
visualized with price, volume and moving-average plots,
analyzed for seasonality and correlation with the market
index, and forecast using ARIMA and Exponential
Smoothing. Exponential smoothing gave lower error (MAPE
3.2%). Hence time series analysis helps identify trends
and forecast future values.

============================================================
END OF FILE
============================================================
