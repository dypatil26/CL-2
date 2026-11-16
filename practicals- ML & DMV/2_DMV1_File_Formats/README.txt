============================================================
DMV1 - DATA LOADING, STORAGE AND FILE FORMATS:
ANALYZING SALES DATA FROM MULTIPLE FILE FORMATS
============================================================

## PROBLEM STATEMENT

Analyzing Sales Data from Multiple File Formats.
Load and analyze sales data from different file formats
(CSV, Excel, JSON), and perform data cleaning,
transformation and analysis on the dataset.

Tasks:
1. Obtain and load sales data from CSV, Excel and JSON.
2. Explore structure, find inconsistencies, missing
   values and data quality issues.
3. Perform data cleaning (missing values, duplicates,
   inconsistent formatting).
4. Convert data into a unified format (one DataFrame).
5. Data transformation: merging, splitting columns,
   deriving new variables.
6. Analysis: descriptive statistics, aggregation, total
   sales, average order value, category distribution.
7. Visualizations: bar plot, pie chart, box plot.

============================================================
DATASET
============================================================

Three files in this folder (one per region):

1. sales_data.csv   - West region (308 rows)
   columns: order_id, order_date (dd-mm-yyyy),
   customer_name, region, product_category, quantity,
   unit_price
   Issues: 12 missing unit_price, 8 duplicate rows,
           "electronics " (wrong case + extra space)

2. sales_data.xlsx  - North region (250 rows)
   columns: Order ID, Order Date, Customer Name, Region,
   Category, Qty, Unit Price   (DIFFERENT column names)
   Issues: 10 missing Qty

3. sales_data.json  - South region (250 records)
   NESTED structure:
   {"order_id":..., "order_date": "yyyy-mm-dd",
    "customer": {"name":..., "region":...},
    "item": {"category":..., "quantity":...,
             "unit_price":...}}
   Issues: 8 null unit_price

============================================================
OBJECTIVE
============================================================

* To load data from CSV, Excel and JSON formats.
* To identify and fix data quality issues.
* To combine heterogeneous data into one structure.
* To analyze and visualize sales data.

============================================================
LIBRARIES USED
============================================================

1. pandas      - read_csv, read_excel, json_normalize,
                 cleaning, aggregation
2. json        - load JSON file
3. openpyxl    - engine used by pandas to read .xlsx
4. matplotlib  - charts
5. os          - file paths

============================================================
HOW TO RUN
============================================================

pip install pandas matplotlib openpyxl

python dmv1_sales_file_formats.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  sales_data.csv, then sales_data.xlsx,
  then sales_data.json (three prompts, one after another)
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

Output file created: cleaned_combined_sales.csv

============================================================
THEORY
============================================================

## Data Loading

Data loading = reading data from storage (files,
databases, web) into memory structures like DataFrames.

------------------------------------------------------------

## Common File Formats

1. CSV (Comma Separated Values)
   * Plain text, one row per line, values separated by
     commas. Small, universal, no data types/formatting.
   * pd.read_csv(), df.to_csv()

2. Excel (.xlsx)
   * Binary/XML spreadsheet, multiple sheets, formatting.
   * pd.read_excel() (needs openpyxl), df.to_excel()

3. JSON (JavaScript Object Notation)
   * Key-value text format, supports nesting; used by
     web APIs.
   * json.load(), pd.read_json(), pd.json_normalize()
     (flattens nested JSON: customer.name -> column)

Others: Parquet, HDF5, SQL databases, XML, pickle.

------------------------------------------------------------

## Data Quality Issues

* Missing values (NaN / null)
* Duplicate records
* Inconsistent formatting (case, spaces, spellings)
* Different column names for same data
* Different date formats
* Wrong data types

------------------------------------------------------------

## Data Cleaning

Process of detecting and correcting errors:
* drop_duplicates()
* fillna() with mean/median/mode or group median
* str.strip(), str.title() for text
* pd.to_datetime() for dates

------------------------------------------------------------

## Data Transformation

Converting data into a useful form:
* Merging / concatenating datasets (pd.concat, merge)
* Splitting columns (customer_name -> first, last)
* Deriving new variables (total_sales = qty * price,
  month, quarter)

------------------------------------------------------------

## Data Analysis

Inspecting and summarizing data to get insight:
descriptive statistics, groupby aggregations, KPIs such as
Total Sales and Average Order Value (AOV = total sales /
number of orders).

------------------------------------------------------------

## Data Visualization

Graphical representation of data:
* Bar plot  - compare categories
* Pie chart - share of a whole
* Box plot  - distribution and outliers
* Line plot - trend over time

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1 - Load three formats

csv_df  = pd.read_csv(upload_file("sales_data.csv"))
xl_df   = pd.read_excel(upload_file("sales_data.xlsx"))
raw     = json.load(open(upload_file("sales_data.json")))
json_df = pd.json_normalize(raw)

* json_normalize flattens nested keys into columns like
  customer.name, item.unit_price.

------------------------------------------------------------

## CODE BLOCK 2 - Explore

d.head(), d.isnull().sum(), d.duplicated().sum()
csv_df["product_category"].unique()

* Shows missing values, duplicates and the inconsistent
  category spelling.

------------------------------------------------------------

## CODE BLOCK 3 - Unified format

xl_df.rename(columns={"Order ID": "order_id", ...})
json_df.rename(columns={"customer.name": "customer_name", ...})
pd.to_datetime(... format="%d-%m-%Y")   # CSV
pd.to_datetime(... format="%Y-%m-%d")   # JSON
sales = pd.concat([csv_df[cols], xl_df[cols], json_df[cols]])

* Same column names and date type in all three, then
  concatenated (merged) into one DataFrame (808 rows).

------------------------------------------------------------

## CODE BLOCK 4 - Cleaning

sales.drop_duplicates()
sales["product_category"].str.strip().str.title()
sales["quantity"].fillna(median)
groupby("product_category")["unit_price"]
     .transform(lambda s: s.fillna(s.median()))

* Removes 8 duplicates.
* Fixes "electronics " -> "Electronics".
* Missing quantity -> median quantity.
* Missing price -> median price OF THAT CATEGORY (more
  accurate than overall median).

------------------------------------------------------------

## CODE BLOCK 5 - Transformation

sales["total_sales"] = quantity * unit_price
sales["month"], sales["quarter"]
sales[["first_name","last_name"]] =
      sales["customer_name"].str.split(" ", n=1, expand=True)

* Derived variables and column splitting.

------------------------------------------------------------

## CODE BLOCK 6 - Analysis

describe(), sum(), mean(), nunique()
groupby("region") / groupby("product_category")
value_counts(normalize=True)

* Total sales, average order value, sales by region and
  category, category distribution.

------------------------------------------------------------

## CODE BLOCK 7 - Visualization

barh (sales by category), pie (region share),
boxplot (order value by category),
resample("ME") line plot (monthly trend)

------------------------------------------------------------

## CODE BLOCK 8 - Storage

sales.to_csv("cleaned_combined_sales.csv", index=False)

* Saves cleaned unified data.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. pd.read_csv(), pd.read_excel(), json.load()
2. pd.json_normalize()
3. rename(), pd.to_datetime()
4. pd.concat()
5. drop_duplicates(), fillna(), groupby().transform()
6. str.strip(), str.title(), str.split()
7. describe(), groupby().sum(), value_counts()
8. resample()
9. plot(kind="barh"/"pie"), boxplot()
10. to_csv()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

CSV        Excel        JSON (nested)
  ↓          ↓              ↓
read_csv  read_excel   json_normalize
  └──────────┼──────────────┘
             ↓
Explore (missing, duplicates, inconsistencies)
             ↓
Rename columns + convert dates → concat
             ↓
Clean (duplicates, text, missing values)
             ↓
Transform (total_sales, month, quarter, split name)
             ↓
Analyze (stats, groupby, KPIs)
             ↓
Visualize → Save CSV

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

CSV (308, 7)   Excel (250, 7)   JSON (250, 7)
Combined shape: (808, 7)
Duplicates removed: 8
Missing values after cleaning: 0

Total sales         : Rs 2,82,34,162
Average order value : Rs 35,293
Number of orders    : 800

Electronics and Furniture give the highest sales; the
pie chart shows the share of West, North and South.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What is Data Visualization?

Ans:
Graphical representation of data using charts, graphs and
maps so patterns, trends and outliers are easy to see.

------------------------------------------------------------

2. What are different methods used in Visualization?

Ans:
Bar chart, pie chart, line chart, histogram, box plot,
scatter plot, heatmap, area chart, maps.

------------------------------------------------------------

3. What is Data Analysis?

Ans:
Inspecting, cleaning, transforming and modelling data to
discover useful information and support decisions.

------------------------------------------------------------

4. What is Data Cleaning and Data Transformation?

Ans:
Cleaning = fixing or removing incorrect, duplicate,
missing or inconsistent data. Transformation = converting
data into a suitable format/structure (merging, splitting,
deriving, normalizing, encoding).

------------------------------------------------------------

5. Difference between CSV, Excel and JSON?

Ans:
CSV: plain text table. Excel: spreadsheet with sheets and
formatting. JSON: key-value text, supports nested data,
used by web APIs.

------------------------------------------------------------

6. What does json_normalize() do?

Ans:
Converts nested JSON into a flat table.

------------------------------------------------------------

7. Which library is needed to read .xlsx?

Ans:
openpyxl (engine used by pandas.read_excel).

------------------------------------------------------------

8. Why fill missing price with category median?

Ans:
Prices differ a lot between categories; category median is
a more realistic estimate than the overall median.

------------------------------------------------------------

9. What is the difference between concat and merge?

Ans:
concat stacks DataFrames (rows or columns); merge joins
them on common key columns (like SQL JOIN).

------------------------------------------------------------

10. What is Average Order Value?

Ans:
Total sales / number of orders.

------------------------------------------------------------

11. What is a box plot used for?

Ans:
Shows median, quartiles, spread and outliers.

------------------------------------------------------------

12. Why convert dates using pd.to_datetime()?

Ans:
So dates can be compared, sorted, resampled and parts like
month/quarter can be extracted.

============================================================
CONCLUSION
============================================================

Sales data was loaded from CSV, Excel and nested JSON
files, explored for quality issues, unified into one
DataFrame, cleaned, transformed and analyzed. Total sales,
average order value and category/region distributions were
computed and visualized. Hence data from multiple file
formats can be combined for meaningful analysis.

============================================================
END OF FILE
============================================================
