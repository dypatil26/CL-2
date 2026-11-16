============================================================
ML2 - REGRESSION ANALYSIS:
UBER FARE PREDICTION (LINEAR, RIDGE, LASSO)
============================================================

## PROBLEM STATEMENT

Predict the price of the Uber ride from a given pickup
point to the agreed drop-off location.
Perform the following tasks:
1. Pre-process the dataset.
2. Identify outliers.
3. Check the correlation.
4. Implement linear regression and ridge, Lasso
   regression models.
5. Evaluate the models and compare their respective
   scores like R2, RMSE, etc.

============================================================
DATASET
============================================================

Uber Fares Dataset

Dataset File:
uber.csv   (included in this folder - 5000 rows)

Original source (2 lakh rows, needs Kaggle login):
https://www.kaggle.com/datasets/yasserh/uber-fares-dataset

NOTE: the included uber.csv has EXACTLY the same columns
as the Kaggle file, so you can replace it with the full
Kaggle file and the program works without any change.
Some invalid rows (zero GPS, negative fare, 208
passengers, huge fares, missing drop-off) are included on
purpose so that outlier handling can be shown.

Columns:
* Unnamed: 0, key        - IDs (dropped)
* fare_amount            - TARGET (USD)
* pickup_datetime        - date and time of ride
* pickup_longitude/latitude
* dropoff_longitude/latitude
* passenger_count

============================================================
OBJECTIVE
============================================================

* To pre-process a real-world dataset.
* To detect and remove outliers.
* To study correlation between variables.
* To implement Linear, Ridge and Lasso regression.
* To compare models using R2, RMSE and MAE.

============================================================
LIBRARIES USED
============================================================

1. pandas, numpy          - data handling and maths
2. matplotlib, seaborn    - box plots, heatmap, plots
3. sklearn.model_selection.train_test_split
4. sklearn.preprocessing.StandardScaler
5. sklearn.linear_model.LinearRegression, Ridge, Lasso
6. sklearn.metrics.r2_score, mean_squared_error,
   mean_absolute_error

============================================================
HOW TO RUN
============================================================

pip install pandas numpy matplotlib seaborn scikit-learn

python ml2_uber_regression.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  uber.csv
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

## Regression

Regression is a supervised learning technique used to
predict a CONTINUOUS value (here: fare amount).

------------------------------------------------------------

## Linear Regression

Models the target as a straight-line (linear) combination
of features:

y = b0 + b1*x1 + b2*x2 + ... + bn*xn

* b0 = intercept, b1..bn = coefficients
* Coefficients are found by minimizing the
  Sum of Squared Errors (Ordinary Least Squares).

Cost Function (MSE):
J = (1/n) * SUM (y_actual - y_predicted)^2

------------------------------------------------------------

## Gradient Descent

An optimization algorithm that finds the coefficients
by repeatedly moving them in the direction that reduces
the cost function:

b = b - learning_rate * dJ/db

------------------------------------------------------------

## Overfitting and Regularization

Large coefficients can make a model fit noise
(overfitting). Regularization adds a penalty term to the
cost function to keep coefficients small.

------------------------------------------------------------

## Ridge Regression (L2 regularization)

Cost = MSE + alpha * SUM (b_i^2)

* Shrinks coefficients towards zero, never exactly zero.
* Useful when features are correlated (multicollinearity).

------------------------------------------------------------

## Lasso Regression (L1 regularization)

Cost = MSE + alpha * SUM |b_i|

* Can make coefficients EXACTLY zero.
* So Lasso also performs feature selection.

alpha = strength of penalty (alpha = 0 -> normal
linear regression).

------------------------------------------------------------

## Haversine Distance

Pickup and drop-off are GPS points; straight subtraction
is not distance. The haversine formula gives the
great-circle distance on Earth:

a = sin^2(dlat/2) + cos(lat1)cos(lat2)sin^2(dlon/2)
d = 2 * R * asin(sqrt(a)),  R = 6371 km

Distance is the most important feature for fare.

------------------------------------------------------------

## Outliers - IQR Method

IQR = Q3 - Q1
Lower limit = Q1 - 1.5 * IQR
Upper limit = Q3 + 1.5 * IQR
Values outside limits are outliers.

------------------------------------------------------------

## Correlation

Pearson correlation r ranges from -1 to +1.
* +1 strong positive, -1 strong negative, 0 none.

------------------------------------------------------------

## Evaluation Metrics

R2 Score  = 1 - SS_res / SS_tot
            (1 = perfect, 0 = predicts mean only)

RMSE      = sqrt( mean( (y - y_pred)^2 ) )
            (same unit as fare, lower is better)

MAE       = mean( |y - y_pred| )

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1 - Load

df = pd.read_csv(upload_file("uber.csv"))
print(df.info())

* Loads data and shows column types.

------------------------------------------------------------

## CODE BLOCK 2 - Pre-processing

df = df.drop(columns=["Unnamed: 0", "key"])
df = df.dropna()
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

* Removes ID columns (no predictive value).
* Drops rows with missing values.
* Converts text date to datetime type.

------------------------------------------------------------

## CODE BLOCK 3 - Feature Engineering

df["hour"] = df["pickup_datetime"].dt.hour
df["day"], df["month"], df["year"], df["dayofweek"]
df["distance_km"] = haversine(...)

* Extracts time features (rush hour, year inflation).
* Computes trip distance using haversine().

------------------------------------------------------------

## CODE BLOCK 4 - Outliers

sns.boxplot(y=df[c])
df = df[(df["fare_amount"] > 0) &
        (df["passenger_count"].between(1, 6))]
df = df[df["pickup_latitude"].between(40, 42) ...]
for c in ["fare_amount", "distance_km"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    ...
    df = df[df[c].between(low, high)]

* Box plots show outliers visually.
* Rule-based filter removes impossible values
  (negative fare, 0 coordinates, 208 passengers).
* IQR method removes extreme fares and distances.

------------------------------------------------------------

## CODE BLOCK 5 - Correlation

corr = df[[...]].corr()
sns.heatmap(corr, annot=True)

* distance_km has correlation ~0.92 with fare_amount.
* Other features have small correlation.

------------------------------------------------------------

## CODE BLOCK 6 - Split and Scale

X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size=0.2, random_state=42)
scaler.fit_transform(X_train); scaler.transform(X_test)

* 80% training, 20% testing.
* Scaler is fit ONLY on training data (avoids data
  leakage), then applied to test data.
* Scaling is needed for Ridge/Lasso penalties to be fair.

------------------------------------------------------------

## CODE BLOCK 7 - Models

models = {"Linear Regression": LinearRegression(),
          "Ridge": Ridge(alpha=1.0),
          "Lasso": Lasso(alpha=0.1)}
model.fit(X_train, y_train)
pred = model.predict(X_test)

* Trains all three models and prints coefficients.
* Lasso sets weak features (passenger_count, day, ...)
  to exactly 0 -> feature selection.

------------------------------------------------------------

## CODE BLOCK 8 - Evaluation

r2_score, mean_squared_error (sqrt -> RMSE),
mean_absolute_error

* Prints a comparison table and bar chart of R2.
* Actual vs Predicted scatter plot - points near the
  red diagonal mean good prediction.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. pd.to_datetime()      - convert to datetime
2. .dt.hour / .dt.year   - extract date parts
3. quantile()            - Q1, Q3 for IQR
4. between()             - range filter
5. corr()                - correlation matrix
6. sns.heatmap()         - heatmap
7. train_test_split()    - split data
8. StandardScaler()      - scaling
9. LinearRegression(), Ridge(), Lasso()
10. fit(), predict()
11. r2_score(), mean_squared_error(), mean_absolute_error()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Load uber.csv
     ↓
Drop IDs, missing values, convert datetime
     ↓
Feature engineering (time parts + haversine distance)
     ↓
Outlier detection (box plots) and removal (rules + IQR)
     ↓
Correlation heatmap
     ↓
Train/Test split + Scaling
     ↓
Linear | Ridge | Lasso
     ↓
R2, RMSE, MAE comparison + plots

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Rows removed as outliers/invalid: 96
Correlation fare vs distance: 0.918

Model                         R2      RMSE    MAE
Linear Regression           0.8824   1.2327  0.9799
Ridge Regression (alpha=1)  0.8824   1.2327  0.9800
Lasso Regression (alpha=0.1)0.8811   1.2394  0.9881

Meaning:
* All three models explain ~88% of fare variation.
* Ridge = Linear here because there is little
  multicollinearity.
* Lasso is slightly worse but uses only 2 features
  (distance and year) - simpler model.
* Distance is the strongest predictor of fare.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What is gradient descent?

Ans:
An iterative optimization method that updates model
parameters in the direction of the negative gradient of
the cost function until the cost is minimum.

------------------------------------------------------------

2. What is cost function?

Ans:
A function measuring prediction error. For linear
regression it is Mean Squared Error.

------------------------------------------------------------

3. How does Linear regression work?

Ans:
It fits a line/hyperplane y = b0 + b1x1 + ... that
minimizes the sum of squared differences between actual
and predicted values.

------------------------------------------------------------

4. What is the difference between linear, lasso and ridge
   regression?

Ans:
Linear has no penalty. Ridge adds L2 penalty (squares of
coefficients), shrinks them. Lasso adds L1 penalty
(absolute values), can make coefficients zero and does
feature selection.

------------------------------------------------------------

5. What is regularization?

Ans:
Adding a penalty to the cost function to prevent
overfitting.

------------------------------------------------------------

6. What is alpha in Ridge/Lasso?

Ans:
The regularization strength. Higher alpha = more shrinkage.

------------------------------------------------------------

7. What is R2 score?

Ans:
Proportion of variance in target explained by the model.

------------------------------------------------------------

8. What is RMSE?

Ans:
Root Mean Squared Error - average prediction error in the
same unit as the target.

------------------------------------------------------------

9. Why use haversine instead of Euclidean distance?

Ans:
Coordinates are on a sphere (Earth), haversine gives the
correct distance in km.

------------------------------------------------------------

10. How did you identify outliers?

Ans:
Box plots, domain rules (fare > 0, 1-6 passengers, valid
NYC coordinates) and the IQR method.

------------------------------------------------------------

11. What is multicollinearity?

Ans:
When independent variables are highly correlated with each
other. Ridge regression handles it well.

------------------------------------------------------------

12. Why scale features before Ridge and Lasso?

Ans:
The penalty depends on coefficient size; without scaling,
features with large units get unfair penalty.

------------------------------------------------------------

13. Why fit scaler only on training data?

Ans:
To avoid data leakage - test data must stay unseen.

------------------------------------------------------------

14. Which feature affects fare most?

Ans:
distance_km (correlation 0.92).

------------------------------------------------------------

15. What are assumptions of linear regression?

Ans:
Linearity, independence of errors, constant variance
(homoscedasticity), normal errors, no multicollinearity.

------------------------------------------------------------

16. What is overfitting and underfitting?

Ans:
Overfitting = good on train, bad on test. Underfitting =
bad on both.

============================================================
CONCLUSION
============================================================

The Uber dataset was cleaned, outliers were removed, trip
distance was engineered with the haversine formula and the
correlation was checked. Linear, Ridge and Lasso
regression were implemented; all achieved R2 about 0.88
and RMSE about 1.23. Lasso produced the simplest model by
removing weak features. Hence regression effectively
predicts Uber fares.

============================================================
END OF FILE
============================================================
