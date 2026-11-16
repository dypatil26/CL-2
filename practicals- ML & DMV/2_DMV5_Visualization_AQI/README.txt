============================================================
DMV5 - DATA VISUALIZATION USING MATPLOTLIB:
AIR QUALITY INDEX (AQI) TRENDS IN A CITY
============================================================

## PROBLEM STATEMENT

Analyzing Air Quality Index (AQI) Trends in a City.
The dataset has air quality measurements over time: date,
pollutant levels (PM2.5, PM10, CO ...) and AQI values.
Use matplotlib to create visualizations that represent AQI
trends and patterns for different pollutants.

Tasks:
1. Import "City_Air_Quality.csv".
2. Explore structure and content.
3. Identify relevant variables (date, pollutants, AQI).
4. Line / time-series plot of overall AQI trend.
5. Separate line plots for each pollutant.
6. Bar / stacked bar plots to compare AQI across periods.
7. Box / violin plots for AQI distribution by category.
8. Scatter / bubble charts: AQI vs pollutant levels.
9. Customize with labels, titles, legends, colours.

============================================================
DATASET
============================================================

City_Air_Quality.csv  (included, 731 days: 2023-2024, Pune)

Columns:
Date, City, PM2.5, PM10, NO2, SO2, CO, O3, AQI, AQI_Bucket

* AQI is computed using the Indian CPCB sub-index method
  from PM2.5 and PM10.
* AQI_Bucket: Good, Satisfactory, Moderate, Poor,
  Very Poor, Severe
* Pollution is higher in winter (Nov-Feb) - realistic.
* 15 missing PM10 values (filled by interpolation).

Same column names as the popular Kaggle dataset
"Air Quality Data in India" (city_day.csv), so that file
can also be used after filtering one city.

============================================================
OBJECTIVE
============================================================

* To learn matplotlib plot types and customization.
* To visualize time-series trends.
* To compare and analyze distributions and relationships.

============================================================
LIBRARIES USED
============================================================

1. pandas              - load, datetime index, resample
2. matplotlib.pyplot   - ALL plots in this practical

============================================================
HOW TO RUN
============================================================

pip install pandas matplotlib

python dmv5_aqi_visualization.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  City_Air_Quality.csv
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

## Data Visualization

Graphical representation of data to understand trends,
patterns, outliers and relationships quickly.

------------------------------------------------------------

## Python Visualization Libraries

* Matplotlib - base library, full control
* Seaborn    - statistical plots on top of matplotlib
* Plotly     - interactive plots
* Bokeh, Altair, Folium (maps)

------------------------------------------------------------

## Matplotlib

The most widely used Python 2-D plotting library.
pyplot module gives MATLAB-like functions.

Figure = whole window; Axes = one plot inside it.
fig, ax = plt.subplots(rows, cols)

------------------------------------------------------------

## Plot Types Used

1. Line plot      - plt.plot()      trend over time
2. Bar plot       - plt.bar()       compare categories
3. Stacked bar    - plot(kind="bar", stacked=True)
4. Box plot       - plt.boxplot()   median, quartiles,
                                    outliers
5. Violin plot    - plt.violinplot() box plot + density
6. Scatter plot   - plt.scatter()   relationship between
                                    2 variables
7. Bubble chart   - scatter with size (s=) = 3rd variable
8. Histogram      - plt.hist()      frequency

------------------------------------------------------------

## Customization

plt.title(), plt.xlabel(), plt.ylabel(), plt.legend(),
plt.grid(), color=, cmap= (colormap), alpha=
(transparency), plt.xticks(rotation=), plt.colorbar(),
plt.axhline() (reference line), plt.tight_layout()

------------------------------------------------------------

## Colormaps ("maps") in Matplotlib

Sequential (viridis, plasma), diverging (coolwarm),
qualitative (tab10), cyclic (twilight).

------------------------------------------------------------

## Air Quality Index (AQI)

A single number (0-500) that tells how polluted the air
is. India (CPCB):
0-50 Good, 51-100 Satisfactory, 101-200 Moderate,
201-300 Poor, 301-400 Very Poor, 401-500 Severe.

AQI = maximum of sub-indices of pollutants.

Moving average: rolling(30).mean() smooths daily noise to
show the long-term trend.

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1-2 - Import and explore

df = pd.read_csv(...); df.dtypes; df.isnull().sum();
df.describe()

------------------------------------------------------------

## CODE BLOCK 3 - Variables

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")
df["PM10"] = df["PM10"].interpolate()

* Date as index makes time-series plotting and
  resampling easy.

------------------------------------------------------------

## CODE BLOCK 4 - Overall AQI trend

plt.plot(df.index, df["AQI"], alpha=0.5)
plt.plot(df.index, df["AQI"].rolling(30).mean())
plt.axhline(100, ls="--")

* Grey = daily, red = 30-day moving average,
  green dashed line = satisfactory limit.

------------------------------------------------------------

## CODE BLOCK 5 - Pollutant trends

fig, axes = plt.subplots(3, 2, sharex=True)
for ax, p, c in zip(axes.ravel(), pollutants, colors):
    ax.plot(df.index, df[p], color=c)

* 6 separate line plots in one figure.

------------------------------------------------------------

## CODE BLOCK 6 - Bar / stacked bar

monthly = df["AQI"].resample("ME").mean()
plt.bar(..., color=["red" if v > 100 else "green" ...])
pd.crosstab(month, AQI_Bucket).plot(kind="bar", stacked=True)

* Monthly average AQI bar chart (red = unhealthy months).
* Stacked bar = number of days in each AQI category per
  month.

------------------------------------------------------------

## CODE BLOCK 7 - Box and violin plots

ax[0].boxplot(data, labels=order)
ax[1].violinplot([... per season ...], showmedians=True)

* Distribution of AQI for each category and each season.

------------------------------------------------------------

## CODE BLOCK 8 - Scatter and bubble

ax[0].scatter(df["PM2.5"], df["AQI"])
ax[1].scatter(df["NO2"], df["AQI"], s=df["CO"]*120,
              c=df["PM2.5"], cmap="plasma")
plt.colorbar(...)

* Bubble size = CO, colour = PM2.5 (4 variables in one
  plot).

------------------------------------------------------------

## CODE BLOCK 9 - Summary

nlargest(5), value_counts() of buckets.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. pd.to_datetime(), set_index(), interpolate()
2. rolling().mean(), resample("ME").mean()
3. pd.crosstab()
4. plt.plot(), plt.bar(), plt.boxplot(),
   plt.violinplot(), plt.scatter()
5. plt.subplots()
6. title(), xlabel(), ylabel(), legend(), grid(),
   colorbar(), axhline(), xticks(rotation=)
7. plt.tight_layout(), plt.show()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Import CSV → Explore
     ↓
Datetime index, fill missing
     ↓
Line plot: AQI trend + moving average
     ↓
Subplots: each pollutant trend
     ↓
Bar + stacked bar: monthly comparison
     ↓
Box + violin: distributions
     ↓
Scatter + bubble: AQI vs pollutants
     ↓
Summary

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Monthly average AQI: Jan-2023 = 201, Feb-2023 = 169 ...
(winter months are red / unhealthy, monsoon months green)

Days per category (731 days):
Satisfactory 263, Moderate 242, Good 125, Poor 70,
Very Poor 31

Scatter plot: AQI increases almost linearly with PM2.5 and
PM10 -> particulate matter drives AQI in this city.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What do you mean by data visualization?

Ans:
Representing data graphically (charts, plots, maps) to
understand patterns, trends and outliers easily.

------------------------------------------------------------

2. What are the different libraries in Python (for
   visualization)?

Ans:
Matplotlib, Seaborn, Plotly, Bokeh, Altair, Folium,
pandas plotting.

------------------------------------------------------------

3. What is Matplotlib?

Ans:
A Python 2-D plotting library (pyplot module) for creating
static, animated and interactive plots like line, bar,
scatter, histogram, box and pie charts.

------------------------------------------------------------

4. What are the different maps in Matplotlib?

Ans:
Colormaps: sequential (viridis, plasma, Blues),
diverging (coolwarm, RdBu), qualitative (tab10, Set1),
cyclic (twilight). (Geographical maps are drawn with
Basemap/Cartopy.)

------------------------------------------------------------

5. What is AQI?

Ans:
Air Quality Index - number from 0-500 showing pollution
level; based on the worst pollutant sub-index.

------------------------------------------------------------

6. What is a time series plot?

Ans:
A line plot with time on the x-axis.

------------------------------------------------------------

7. Difference between box plot and violin plot?

Ans:
Box plot shows median/quartiles/outliers; violin also
shows the full density shape.

------------------------------------------------------------

8. What is a bubble chart?

Ans:
A scatter plot where point size represents a third
variable.

------------------------------------------------------------

9. What is a stacked bar chart?

Ans:
Bars split into segments showing parts of a total.

------------------------------------------------------------

10. Why use a moving average?

Ans:
To smooth short-term fluctuations and show the trend.

------------------------------------------------------------

11. Difference between figure and axes?

Ans:
Figure is the whole canvas; axes is an individual plot
area inside it.

------------------------------------------------------------

12. What does plt.tight_layout() do?

Ans:
Adjusts spacing so labels and titles do not overlap.

============================================================
CONCLUSION
============================================================

The city air quality data was explored and visualized
using matplotlib line, bar, stacked bar, box, violin,
scatter and bubble plots with proper labels, legends and
colours. The plots show that AQI peaks in winter and is
driven mainly by PM2.5 and PM10. Hence matplotlib
effectively represents trends and patterns in data.

============================================================
END OF FILE
============================================================
