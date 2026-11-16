============================================================
DMV3 - DATA CLEANING AND PREPARATION:
CUSTOMER CHURN IN A TELECOMMUNICATIONS COMPANY
============================================================

## PROBLEM STATEMENT

Analyzing Customer Churn in a Telecommunications Company.
The dataset contains information about customers and
whether they have churned (discontinued the service).
Perform data cleaning and preparation to gain insight into
factors that contribute to churn.

Tasks:
1. Import "Telecom_Customer_Churn.csv".
2. Explore structure and content.
3. Handle missing values.
4. Remove duplicate records.
5. Standardize inconsistent data (formatting/spelling).
6. Convert columns to correct data types.
7. Identify and handle outliers.
8. Feature engineering.
9. Normalize / scale data.
10. Split into training and testing sets.
11. Export the cleaned dataset.

============================================================
DATASET
============================================================

Telecom_Customer_Churn.csv  (included, 7068 rows, 21 cols)

Based on the IBM Telco Customer Churn dataset (public):
https://github.com/IBM/telco-customer-churn-on-icp4d

To practise every task, some realistic problems were added:
* 25 duplicate rows
* inconsistent text: "male", "FEMALE", " Male", "yes",
  "NO", "Y", "month-to-month", "two year"
* 40 missing MonthlyCharges
* impossible tenure values (250, 300, -3)
* TotalCharges stored as TEXT with blank " " values
  (this issue exists in the original IBM file too)

Main columns:
customerID, gender, SeniorCitizen, Partner, Dependents,
tenure (months), PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges,
Churn (Yes/No) - TARGET

============================================================
OBJECTIVE
============================================================

* To perform all data cleaning steps on a real dataset.
* To prepare data for machine learning.
* To find factors related to customer churn.

============================================================
LIBRARIES USED
============================================================

1. pandas, numpy           - cleaning and transformation
2. matplotlib, seaborn     - box plots, bar plot
3. sklearn.preprocessing.MinMaxScaler - normalization
4. sklearn.model_selection.train_test_split

============================================================
HOW TO RUN
============================================================

pip install pandas numpy matplotlib seaborn scikit-learn

python dmv3_churn_cleaning.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  Telecom_Customer_Churn.csv
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

Output files:
* Telecom_Customer_Churn_cleaned.csv     (readable)
* Telecom_Customer_Churn_model_ready.csv (encoded+scaled)

============================================================
THEORY
============================================================

## Customer Churn

Churn = a customer stops using a company's service.
Churn rate = churned customers / total customers.
Retaining customers is cheaper than finding new ones, so
companies analyze churn factors.

------------------------------------------------------------

## Data Preparation

All steps that turn raw data into clean, consistent,
model-ready data: cleaning, transforming, encoding,
scaling, splitting. It usually takes ~70-80% of a data
science project's time.

------------------------------------------------------------

## Data Cleaning Steps

1. Missing values
   * Delete rows/columns
   * Impute: mean (normal data), median (skewed data),
     mode (categorical), formula-based

2. Duplicates
   * df.duplicated(), df.drop_duplicates()

3. Inconsistent data
   * Same value written differently ("Male"/"male")
   * Fix with str.strip(), str.lower(), map()

4. Data types
   * Numbers stored as text -> pd.to_numeric(errors=
     "coerce") (invalid text becomes NaN)

5. Outliers
   * Detect: box plot, IQR, Z-score, domain rules
   * Handle: remove, cap (clip), transform

------------------------------------------------------------

## Feature Engineering

Creating new informative features from existing ones,
e.g. tenure_group, number of services, average charge.

------------------------------------------------------------

## Encoding

* Label encoding: category -> number
* One-hot encoding: one 0/1 column per category
  (pd.get_dummies, drop_first=True avoids redundancy)

------------------------------------------------------------

## Normalization / Scaling

Min-Max: x' = (x - min) / (max - min)  -> range [0, 1]
Standardization: z = (x - mean) / SD

------------------------------------------------------------

## Train/Test Split

Data divided (e.g. 80/20) to train a model and test it on
unseen data. stratify=y keeps the same churn ratio in
both parts.

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1 - Import and explore

df = pd.read_csv(...); df.shape; df.dtypes
df["Churn"].value_counts()

------------------------------------------------------------

## CODE BLOCK 2 - Missing values

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())

* Blank strings are hidden missing values - converted to
  NaN first so they can be counted.
* MonthlyCharges filled with median.

------------------------------------------------------------

## CODE BLOCK 3 - Duplicates

df.duplicated().sum(); df.drop_duplicates()

------------------------------------------------------------

## CODE BLOCK 4 - Inconsistent data

df["gender"].str.strip().str.capitalize()
df["Partner"].str.strip().str.lower()
         .map({"yes": "Yes", "y": "Yes", "no": "No"})
df["Contract"].str.lower().map({...})

* Before/after unique values printed.

------------------------------------------------------------

## CODE BLOCK 5 - Data types

pd.to_numeric(df["TotalCharges"], errors="coerce")
fillna(MonthlyCharges * tenure)
SeniorCitizen 0/1 -> "No"/"Yes", Churn Yes/No -> 1/0

* Missing TotalCharges calculated by formula (new
  customers with tenure 0 have total 0).

------------------------------------------------------------

## CODE BLOCK 6 - Outliers

sns.boxplot(); IQR limits on tenure
df["tenure"].clip(lower=0, upper=72)

* Tenure must be between 0 and 72 months in this data;
  invalid values are capped.

------------------------------------------------------------

## CODE BLOCK 7 - Feature engineering

tenure_group = pd.cut(tenure, [-1,12,24,48,72])
num_services = count of "Yes" in 7 service columns
avg_charge_per_month = TotalCharges / tenure
groupby("Contract")["Churn"].mean()

------------------------------------------------------------

## CODE BLOCK 8 - Encode and scale

pd.get_dummies(model_df, drop_first=True, dtype=int)
MinMaxScaler().fit_transform(model_df[num])

------------------------------------------------------------

## CODE BLOCK 9 - Split

train_test_split(X, y, test_size=0.2, stratify=y)

------------------------------------------------------------

## CODE BLOCK 10 - Export

df.to_csv("Telecom_Customer_Churn_cleaned.csv")
model_df.to_csv("Telecom_Customer_Churn_model_ready.csv")

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. replace(), isnull().sum(), fillna()
2. duplicated(), drop_duplicates()
3. str.strip(), str.lower(), str.capitalize(), map()
4. pd.to_numeric(errors="coerce")
5. quantile(), clip()
6. pd.cut()
7. groupby().mean()
8. pd.get_dummies()
9. MinMaxScaler()
10. train_test_split()
11. to_csv()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Import CSV → Explore
     ↓
Missing values → Duplicates → Inconsistent text
     ↓
Data types → Outliers
     ↓
Feature engineering → churn insights
     ↓
One-hot encode → Min-Max scale
     ↓
Train/Test split
     ↓
Export cleaned files

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Duplicates found: 24
gender before: Female, Male, " Male", FEMALE, male
gender after : Female, Male
tenure range after capping: 0 - 72

Churn rate by contract:
Month-to-month  0.426  (42.6%)
One year        0.114
Two year        0.031

X_train: (5635, 35)   X_test: (1409, 35)
Churn ratio train/test: 0.265 / 0.265

Insight: customers on month-to-month contracts and with
short tenure churn the most.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-2 are from the lab manual)

1. What is Data Preparation and cleaning?

Ans:
Data cleaning = detecting and fixing missing, duplicate,
inconsistent and incorrect data. Data preparation = the
whole process of cleaning, transforming, encoding,
scaling and splitting data so it is ready for analysis or
modelling.

------------------------------------------------------------

2. How to perform analysis on a dataset?

Ans:
Import data, explore it (shape, types, summary), clean it,
transform/engineer features, compute statistics and group
comparisons (e.g. churn rate by contract), visualize, and
draw conclusions.

------------------------------------------------------------

3. What is churn?

Ans:
When a customer stops using a company's service.

------------------------------------------------------------

4. How did you handle missing values?

Ans:
Blank TotalCharges converted to NaN and filled using
MonthlyCharges x tenure; MonthlyCharges filled by median.

------------------------------------------------------------

5. Why median instead of mean?

Ans:
Median is not affected by outliers/skew.

------------------------------------------------------------

6. What does errors="coerce" do?

Ans:
Converts values that cannot be parsed into NaN instead of
raising an error.

------------------------------------------------------------

7. What is feature engineering?

Ans:
Creating new useful features from existing data.

------------------------------------------------------------

8. Difference between normalization and standardization?

Ans:
Normalization scales to [0,1]; standardization gives mean
0 and SD 1.

------------------------------------------------------------

9. Why drop_first=True in get_dummies?

Ans:
Avoids the dummy variable trap (one column is redundant).

------------------------------------------------------------

10. Why stratify while splitting?

Ans:
To keep the same churn proportion in train and test sets.

------------------------------------------------------------

11. What is capping (clipping)?

Ans:
Replacing values beyond limits with the limit values.

------------------------------------------------------------

12. Which factor most affects churn here?

Ans:
Contract type - month-to-month customers churn ~43%.

============================================================
CONCLUSION
============================================================

The telecom churn dataset was imported, explored and
cleaned by handling missing values, duplicates,
inconsistent text, wrong data types and outliers. New
features were engineered, data was encoded, scaled, split
and exported. Contract type and tenure were found to be
key factors in churn. Hence proper data cleaning and
preparation makes data reliable for analysis and modelling.

============================================================
END OF FILE
============================================================
