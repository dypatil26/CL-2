============================================================
DMV6 - DATA AGGREGATION:
SALES PERFORMANCE BY REGION IN A RETAIL COMPANY
============================================================

## PROBLEM STATEMENT

Analyzing Sales Performance by Region in a Retail Company.
The dataset contains sales transactions (date, product
category, quantity, sales amount, region). Perform data
aggregation to analyze sales performance by region and
identify the top-performing regions.

Tasks:
1. Import "Retail_Sales_Data.csv".
2. Explore structure and content.
3. Identify variables for aggregation (region, sales
   amount, product category).
4. Group by region and calculate total sales per region.
5. Bar plot / pie chart of sales by region.
6. Identify top-performing regions.
7. Group by region AND product category - total sales.
8. Stacked / grouped bar plots to compare them.

============================================================
DATASET
============================================================

Retail_Sales_Data.csv  (included, 2000 transactions, 2024)

Columns:
Transaction ID, Date, Customer ID, Gender, Age,
Region (North, South, East, West, Central),
Product Category (Electronics, Clothing, Beauty,
Home & Kitchen, Sports),
Quantity, Price per Unit, Total Amount

(Similar to the Kaggle "Retail Sales Dataset", with a
Region column added as the manual requires.)

============================================================
OBJECTIVE
============================================================

* To understand data aggregation with groupby and pivot.
* To compute totals, averages and counts per group.
* To visualize aggregated results.
* To identify top-performing regions and categories.

============================================================
LIBRARIES USED
============================================================

1. pandas      - groupby, agg, pivot_table, unstack
2. matplotlib  - bar, pie, stacked and grouped bar plots

============================================================
HOW TO RUN
============================================================

pip install pandas matplotlib

python dmv6_retail_aggregation.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  Retail_Sales_Data.csv
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

## Data Aggregation

Data aggregation is the process of gathering data and
expressing it in a summarized form - e.g. total, average,
count, max, min - usually for groups of records.

Example: 2000 transactions -> 5 numbers (sales per region)

------------------------------------------------------------

## Split - Apply - Combine

pandas groupby works in 3 steps:
1. Split   : divide data into groups (by Region)
2. Apply   : apply a function on each group (sum)
3. Combine : join results into a new table

------------------------------------------------------------

## Aggregate Functions

sum(), mean(), median(), count(), size(), min(), max(),
std(), var(), nunique(), first(), last()

Multiple at once:
df.groupby("Region").agg(total=("Total Amount","sum"),
                         avg=("Total Amount","mean"))

------------------------------------------------------------

## Multi-level Grouping

groupby(["Region", "Product Category"]) creates groups for
every combination. unstack() turns the inner index into
columns (a matrix/table).

------------------------------------------------------------

## Pivot Table

pd.pivot_table(df, values=, index=, columns=, aggfunc=,
margins=True)
Same as Excel pivot table; margins adds row/column totals.

------------------------------------------------------------

## Steps in Data Aggregation

1. Collect / import data
2. Choose grouping variables (dimensions)
3. Choose measures (numeric values)
4. Choose aggregate functions
5. Compute (groupby / pivot)
6. Present / visualize results

------------------------------------------------------------

## Visualization of Aggregated Data

* Bar chart     - compare totals of groups
* Pie chart     - share of each group
* Stacked bar   - total and composition
* Grouped bar   - side-by-side comparison
* Line chart    - aggregated values over time

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1-2 - Import and explore

df = pd.read_csv(...); df.dtypes; df.describe()
df["Date"] = pd.to_datetime(df["Date"])

------------------------------------------------------------

## CODE BLOCK 3 - Variables

df["Region"].unique(), df["Product Category"].unique()

* Region & Product Category = grouping variables.
* Total Amount & Quantity = measures.

------------------------------------------------------------

## CODE BLOCK 4 - Total sales per region

region_sales = df.groupby("Region")["Total Amount"]
                 .sum().sort_values(ascending=False)

df.groupby("Region").agg(total_sales=("Total Amount","sum"),
                         avg_sale=("Total Amount","mean"),
                         transactions=("Transaction ID","count"),
                         units=("Quantity","sum"))

------------------------------------------------------------

## CODE BLOCK 5 - Bar and pie

region_sales.plot(kind="bar"); ax.text(...) adds values
ax.pie(region_sales, autopct="%1.1f%%")

------------------------------------------------------------

## CODE BLOCK 6 - Top regions

region_sales.head(3); region_sales.idxmax()

* idxmax() returns the region with maximum sales.

------------------------------------------------------------

## CODE BLOCK 7 - Region x Category

rc = df.groupby(["Region","Product Category"])
       ["Total Amount"].sum().unstack()
pd.pivot_table(df, values="Total Amount", index="Region",
               columns="Product Category", aggfunc="sum",
               margins=True)
rc.idxmax(axis=1)  -> best category in each region

------------------------------------------------------------

## CODE BLOCK 8 - Stacked and grouped bars

rc.plot(kind="bar", stacked=True)
rc.plot(kind="bar")

------------------------------------------------------------

## CODE BLOCK 9 - Extra: monthly trend

groupby([df["Date"].dt.to_period("M"), "Region"]).sum()
.unstack().plot()

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. groupby()
2. sum(), mean(), count()
3. agg() with named aggregation
4. sort_values(), head()
5. idxmax()
6. unstack()
7. pd.pivot_table()
8. dt.to_period()
9. plot(kind="bar"/"pie", stacked=True)

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Import CSV → Explore
     ↓
Select dimensions (Region, Category) + measure (Amount)
     ↓
groupby Region → sum → bar + pie
     ↓
Top regions (head, idxmax)
     ↓
groupby Region + Category → unstack / pivot_table
     ↓
Stacked + grouped bar plots
     ↓
Monthly trend by region

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Best region: West with sales 4,20,240 (27.1% of total)
Top 3 regions: West, North, South
East and Central have lowest sales.

Region x Category table shows Electronics gives the
largest share in most regions.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-3 are from the lab manual)

1. What is Data Aggregation?

Ans:
Collecting data and summarizing it (sum, mean, count etc.)
for groups to get high-level insight.

------------------------------------------------------------

2. What is data visualization?

Ans:
Graphical representation of data.

------------------------------------------------------------

3. What are the different steps performed in data
   aggregation?

Ans:
Import data, choose grouping columns, choose measures,
choose aggregate functions, compute with groupby/pivot,
and present/visualize results.

------------------------------------------------------------

4. What is split-apply-combine?

Ans:
Split data into groups, apply a function to each, combine
results.

------------------------------------------------------------

5. Difference between groupby and pivot_table?

Ans:
Both aggregate; pivot_table directly gives a 2-D table
with index and columns and supports margins (totals).

------------------------------------------------------------

6. What does unstack() do?

Ans:
Moves an index level to columns.

------------------------------------------------------------

7. What does idxmax() return?

Ans:
Index label of the maximum value.

------------------------------------------------------------

8. Difference between count() and size()?

Ans:
count() excludes NaN; size() counts all rows.

------------------------------------------------------------

9. Stacked vs grouped bar chart?

Ans:
Stacked shows total and parts in one bar; grouped shows
parts side by side for easy comparison.

------------------------------------------------------------

10. What does margins=True do?

Ans:
Adds "All" row/column with totals.

============================================================
CONCLUSION
============================================================

Retail transactions were aggregated by region and by
region-category combinations using groupby and
pivot_table. Bar, pie, stacked and grouped bar charts
showed that West is the top-performing region. Hence data
aggregation summarizes large transaction data into useful
business insight.

============================================================
END OF FILE
============================================================
