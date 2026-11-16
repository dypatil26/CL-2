============================================================
DMV4 - DATA WRANGLING:
REAL ESTATE MARKET
============================================================

## PROBLEM STATEMENT

Data Wrangling on Real Estate Market.
The dataset contains housing prices with property
characteristics, location, sale prices and other features.
Perform data wrangling to find factors influencing prices
and prepare the data for further analysis or modelling.

Tasks:
1. Import "RealEstate_Prices.csv" and clean column names
   (remove spaces, special characters, rename).
2. Handle missing values (imputation or removal).
3. Merge additional data (neighborhood demographics).
4. Filter and subset data (time period, type, location).
5. Encode categorical variables (one-hot / label).
6. Aggregate - average prices by neighborhood / type.
7. Identify and handle outliers.

============================================================
DATASET
============================================================

1. RealEstate_Prices.csv  (1200 properties, Pune areas)

   Messy column names on purpose:
   "Property ID", "Sale Date " (trailing space),
   "Neighborhood", "Property Type", "Sq. Ft.",
   "Bedrooms#", "Bathrooms", "Year Built",
   "Lot Size (sqft)", "Sale Price (Rs)"

   Issues: 30 missing prices, 25 missing bathrooms,
   lot size missing for apartments, 6 extreme prices
   (data-entry errors).

2. Neighborhood_Info.csv  (for merging)
   Neighborhood, Avg_Income_LPA, Schools_Nearby,
   Metro_Access

============================================================
OBJECTIVE
============================================================

* To understand data wrangling.
* To clean column names and handle missing values.
* To merge, filter, encode and aggregate data.
* To detect and treat outliers.

============================================================
LIBRARIES USED
============================================================

1. pandas                  - wrangling
2. re                      - regular expressions for
                             cleaning column names
3. matplotlib, seaborn     - plots
4. sklearn.preprocessing.LabelEncoder

============================================================
HOW TO RUN
============================================================

pip install pandas matplotlib seaborn scikit-learn

python dmv4_realestate_wrangling.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  RealEstate_Prices.csv (Step 1), then
  Neighborhood_Info.csv (Step 3)
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

Output: RealEstate_Prices_wrangled.csv

============================================================
THEORY
============================================================

## Data Wrangling (Data Munging)

The process of transforming raw, messy data into a clean,
structured and enriched format suitable for analysis.

------------------------------------------------------------

## Steps in Data Wrangling

1. Discovering   - understand the data
2. Structuring   - organize format/columns
3. Cleaning      - fix errors, missing, outliers
4. Enriching     - add data from other sources (merge)
5. Validating    - check quality and consistency
6. Publishing    - save/share the prepared data

------------------------------------------------------------

## Cleaning Column Names

Good names: lowercase, no spaces, no special characters,
meaningful -> easy to use as df.column_name.
"Sale Price (Rs)" -> "sale_price" -> "price"

------------------------------------------------------------

## Merging (Joining)

pd.merge(left, right, on="key", how=...)
* inner : only matching keys
* left  : all rows of left table
* right : all rows of right table
* outer : all rows of both

------------------------------------------------------------

## Filtering and Subsetting

Selecting rows by conditions (boolean indexing):
df[(df.year >= 2023) & (df.type == "Apartment")]
Selecting columns: df.loc[condition, [cols]]

------------------------------------------------------------

## Encoding Categorical Variables

* Label encoding: each category -> integer
  (good for binary or ordered categories)
* One-hot encoding: one 0/1 column per category
  (good for nominal categories with no order)

------------------------------------------------------------

## Aggregation

Summarizing groups: groupby().agg(mean, median, count)

------------------------------------------------------------

## Outliers

Extreme values that can distort averages and models.
IQR method:
lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR
Using price per sq.ft. is better than raw price because
big houses are naturally expensive.

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1 - Clean column names

def clean_name(c):
    c = c.strip().lower()
    c = re.sub(r"\(.*?\)", "", c)      # remove (Rs)
    c = re.sub(r"[^a-z0-9]+", "_", c)  # special -> _
    return c.strip("_")
df.columns = [clean_name(c) for c in df.columns]
df.rename(columns={"sq_ft": "area_sqft", ...})

* "Sale Date " -> sale_date, "Sq. Ft." -> area_sqft,
  "Bedrooms#" -> bedrooms, "Sale Price (Rs)" -> price

------------------------------------------------------------

## CODE BLOCK 2 - Missing values

df.dropna(subset=["price"])
bathrooms -> median bathrooms of same bedroom count
lot_size_sqft -> 0 (apartments have no lot)

* Target (price) missing -> remove row (cannot impute
  the answer).

------------------------------------------------------------

## CODE BLOCK 3 - Merge

info = pd.read_csv("Neighborhood_Info.csv")
df = df.merge(info, on="neighborhood", how="left")

* Adds income, schools and metro access to each property.

------------------------------------------------------------

## CODE BLOCK 4 - Filter / subset

df[(df["sale_year"] >= 2023) &
   (df["property_type"] == "Apartment")]
df.loc[df["neighborhood"] == "Baner", [...]]

------------------------------------------------------------

## CODE BLOCK 5 - Encoding

LabelEncoder() on metro_access (Yes/No -> 1/0)
pd.get_dummies(df, columns=["property_type"], prefix="type")

------------------------------------------------------------

## CODE BLOCK 6 - Aggregation

df["price_per_sqft"] = price / area_sqft
groupby("neighborhood").agg(avg_price=..., median_price=...,
                            avg_price_sqft=..., count=...)
average price by property type

------------------------------------------------------------

## CODE BLOCK 7 - Outliers

IQR on price_per_sqft, box plots before/after,
remove rows outside limits.

------------------------------------------------------------

## CODE BLOCK 8

bar chart of avg price per sqft by neighborhood
correlation of features with price
to_csv("RealEstate_Prices_wrangled.csv")

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. re.sub()             - regex replace
2. rename()
3. dropna(subset=), fillna(), groupby().transform()
4. pd.merge()
5. boolean indexing, loc[]
6. LabelEncoder(), pd.get_dummies()
7. groupby().agg()
8. quantile()
9. corr()
10. to_csv()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Import CSV
     ↓
Clean column names (regex)
     ↓
Handle missing values
     ↓
Merge neighborhood info
     ↓
Filter / subset examples
     ↓
Encode categorical variables
     ↓
Aggregate (price per sqft by area/type)
     ↓
Outlier detection and removal
     ↓
Visualize + save

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Cleaned columns: property_id, sale_date, neighborhood,
property_type, area_sqft, bedrooms, bathrooms,
year_built, lot_size_sqft, price

After handling: 0 missing values
IQR limits on price/sqft: [3262, 11499] -> 7 outliers
Final shape: (1163, 19)

BEFORE outlier removal, Kharadi looks the most expensive
(avg 15,734 Rs/sqft) only because of a few data-entry
errors (prices of 50-90 crore). AFTER removing outliers:

Aundh 8837 > Kothrud 8628 > Baner 8253 > Viman Nagar 7978
> Kharadi 7186 > Hinjewadi 6564 > Wakad 6240
> Hadapsar 5893  (Rs per sq.ft.)

This shows why outlier handling is important. Area is the
strongest factor; neighborhood income and schools also
correlate positively with price.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-5 are from the lab manual)

1. What is Data Wrangling?

Ans:
Transforming raw, messy data into a clean, structured,
enriched form ready for analysis.

------------------------------------------------------------

2. What are the different steps in data wrangling?

Ans:
Discovering, structuring, cleaning, enriching (merging),
validating and publishing.

------------------------------------------------------------

3. What is data cleaning?

Ans:
Fixing or removing incorrect, missing, duplicate or
inconsistent data.

------------------------------------------------------------

4. How do you handle missing data?

Ans:
Remove rows/columns or impute with mean/median/mode/group
statistics or domain logic (e.g. lot size 0 for flats).

------------------------------------------------------------

5. What is data visualization?

Ans:
Graphical representation of data to reveal patterns.

------------------------------------------------------------

6. Why clean column names?

Ans:
Spaces/special characters cause errors and make code hard
to write; clean names are consistent and readable.

------------------------------------------------------------

7. Types of joins in pandas merge?

Ans:
inner, left, right, outer.

------------------------------------------------------------

8. Difference between label and one-hot encoding?

Ans:
Label gives integers (implies order); one-hot creates
binary columns (no order).

------------------------------------------------------------

9. Why use price per sq.ft. for outliers?

Ans:
It normalizes for size; raw price is naturally high for
big houses.

------------------------------------------------------------

10. What is boolean indexing?

Ans:
Selecting rows using True/False conditions.

------------------------------------------------------------

11. Why drop rows with missing price?

Ans:
Price is the target; imputing it would create fake answers.

============================================================
CONCLUSION
============================================================

The real estate dataset was wrangled by cleaning column
names, handling missing values, merging neighborhood data,
filtering, encoding categorical variables, aggregating
prices and removing outliers. The prepared dataset shows
that area and neighborhood strongly influence housing
prices. Hence data wrangling converts messy data into
analysis-ready data.

============================================================
END OF FILE
============================================================
