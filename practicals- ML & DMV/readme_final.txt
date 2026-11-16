============================================================
COMPUTER LABORATORY I (417525)
MACHINE LEARNING + DATA MODELING & VISUALIZATION
THEORY, DEFINITIONS AND VIVA GUIDE
============================================================

Course   : BE AI & DS (SPPU 2020 pattern) - Computer Lab I
Part I   : Machine Learning (6 practicals)
Part II  : Data Modeling and Visualization (7 practicals,
           syllabus says perform any 6)

============================================================
LIST OF PRACTICALS (FOLDERS)
============================================================

PART I - MACHINE LEARNING
1_ML1_PCA_Wine          PCA dimensionality reduction (Wine)
1_ML2_Regression_Uber   Linear, Ridge, Lasso (Uber fares)
1_ML3_SVM_Digits        SVM handwritten digit recognition
1_ML4_KMeans_Iris       K-Means + elbow method (Iris)
1_ML5_RandomForest_Car  Random Forest (car safety)
1_ML6_RL_TicTacToe      Q-learning Tic-Tac-Toe

PART II - DATA MODELING AND VISUALIZATION
2_DMV1_File_Formats            Sales data CSV/Excel/JSON
2_DMV2_Web_API_Weather         OpenWeatherMap API
2_DMV3_Data_Cleaning_Churn     Telecom customer churn
2_DMV4_Data_Wrangling_RealEstate Real estate prices
2_DMV5_Visualization_AQI       Air quality (matplotlib)
2_DMV6_Data_Aggregation_Retail Sales by region
2_DMV7_Time_Series_Stock       Stock prices + ARIMA

Every folder has:
* the .py program
* README.txt (problem statement, dataset, theory, code
  explanation, workflow, sample output, viva Q&A,
  conclusion)
* the dataset file(s) needed (except ML3 and ML6 which do
  not need files)

DATASETS ARE UPLOADED WHEN THE PROGRAM RUNS:
The programs do not look for the dataset in their own
folder. When a program reaches the step that loads data,
it asks you to upload it - a file-picker window on your
computer (Jupyter or plain Python), the "Choose files"
button on Google Colab, or typing the path if no window
can open. So you can paste the code into Jupyter (whole
program in one cell, or block by block in order) and keep
the dataset anywhere. Each practical's README.txt says
exactly which file(s) to pick.

============================================================
INSTALL EVERYTHING (ONE COMMAND)
============================================================

pip install pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl requests

Then run any program:
python <program_name>.py

All 13 programs were tested and run without errors on
Python 3.10 (pandas 2.2, scikit-learn 1.6, statsmodels 0.15).
Programs find their dataset in their own folder, so they
can be run from any directory or inside Jupyter Notebook
(keep the CSV in the notebook's folder).

============================================================
INDEX
============================================================

1. Machine Learning Basics
2. Types of Machine Learning
3. ML Pipeline
4. Data Pre-processing
5. Dimensionality Reduction (PCA, LDA)
6. Regression and Regularization
7. Classification (SVM, k-NN, Logistic)
8. Evaluation Metrics
9. Clustering (K-Means, Hierarchical, K-Medoids)
10. Ensemble Learning
11. Reinforcement Learning
12. Data Modeling and Visualization
13. File Formats and Web APIs
14. Data Cleaning, Wrangling, Aggregation
15. Time Series
16. Python Libraries
17. Important Definitions
18. Theoretical Viva Questions

============================================================
1. MACHINE LEARNING BASICS
============================================================

Machine Learning is a branch of AI in which computers
learn patterns from data and improve with experience
without being explicitly programmed.

Tom Mitchell: A program learns from experience E with
respect to task T and performance measure P if its
performance on T, measured by P, improves with E.

============================================================
2. TYPES OF MACHINE LEARNING
============================================================

1. Supervised     - labelled data
   * Regression (continuous output) - ML2
   * Classification (class output)   - ML3, ML5

2. Unsupervised   - no labels
   * Clustering                      - ML4
   * Dimensionality reduction        - ML1

3. Reinforcement  - learn from rewards  - ML6

4. Semi-supervised - few labels + many unlabelled

============================================================
3. ML PIPELINE
============================================================

Problem definition
     ↓
Data collection
     ↓
Data cleaning / pre-processing
     ↓
Feature engineering / selection
     ↓
Train-test split
     ↓
Model training
     ↓
Evaluation
     ↓
Hyper-parameter tuning
     ↓
Deployment / prediction

============================================================
4. DATA PRE-PROCESSING
============================================================

* Handling missing values (drop / mean / median / mode)
* Encoding categorical data
  - Label / Ordinal encoding
  - One-hot encoding
* Feature scaling
  - Normalization (Min-Max): (x - min)/(max - min)
  - Standardization (Z-score): (x - mean)/SD
* Outlier handling (IQR, Z-score, box plot)
* Train-test split (e.g. 80/20, 75/25)

Data leakage: fit scaler/encoder on training data only.

============================================================
5. DIMENSIONALITY REDUCTION
============================================================

## PCA (unsupervised)
Standardize -> covariance matrix -> eigenvalues and
eigenvectors -> sort -> choose top k -> project.
Explained variance ratio = eigenvalue / sum.

## LDA (supervised)
Finds axes that maximize between-class distance and
minimize within-class scatter. Max components = classes-1.

Curse of dimensionality: more features -> sparse data,
more computation, overfitting.

============================================================
6. REGRESSION AND REGULARIZATION
============================================================

Linear: y = b0 + b1x1 + ... + bnxn
Cost (MSE) = (1/n) SUM (y - y_hat)^2
Gradient descent: b = b - alpha * dJ/db

Ridge (L2): MSE + alpha SUM b^2   (shrinks)
Lasso (L1): MSE + alpha SUM |b|   (can make 0 -> feature
                                   selection)
ElasticNet: L1 + L2

Logistic regression: sigmoid(z) = 1/(1+e^-z) ->
probability -> classification.

============================================================
7. CLASSIFICATION
============================================================

## SVM
Maximum-margin hyperplane; support vectors; C controls
margin vs errors; kernels (linear, poly, rbf, sigmoid)
handle non-linear data via the kernel trick.

## k-NN
Lazy learner; majority class of K nearest neighbours;
distance: Euclidean, Manhattan, cosine.

## Decision Tree
If-else splits chosen by Gini or entropy/information gain.

## Naive Bayes
Bayes theorem with independence assumption.

============================================================
8. EVALUATION METRICS
============================================================

## Classification - Confusion Matrix

              Predicted 0   Predicted 1
Actual 0         TN            FP
Actual 1         FN            TP

Accuracy  = (TP+TN)/(TP+TN+FP+FN)
Error rate= 1 - accuracy
Precision = TP/(TP+FP)
Recall    = TP/(TP+FN)  (sensitivity)
Specificity = TN/(TN+FP)
F1 = 2PR/(P+R)

## Regression
MAE = mean|y - y_hat|
MSE = mean(y - y_hat)^2
RMSE = sqrt(MSE)
R2 = 1 - SS_res/SS_tot

## Clustering
WCSS (inertia), silhouette score (-1 to 1)

## Cross-validation
k-fold: split into k parts, train on k-1, test on 1,
repeat k times, average.

============================================================
9. CLUSTERING
============================================================

K-Means: choose K -> init centroids -> assign to nearest
-> update centroids -> repeat until stable.
Elbow method: plot WCSS vs K, pick the bend.
K-Medoids: centre is an actual point (robust).
K-Medians: median + Manhattan distance (robust).
Hierarchical: agglomerative (merge) / divisive (split),
dendrogram.
DBSCAN: density based, finds arbitrary shapes and noise.

============================================================
10. ENSEMBLE LEARNING
============================================================

Bagging: parallel models on bootstrap samples, vote/average
         (Random Forest) -> reduces variance
Boosting: sequential models fixing previous errors
         (AdaBoost, Gradient Boosting, XGBoost) -> reduces
         bias
Voting: hard (majority) / soft (average probabilities)
Stacking: meta-model on top of base models

Random Forest = bagging + random feature subset at each
split + majority vote.

============================================================
11. REINFORCEMENT LEARNING
============================================================

Agent, environment, state, action, reward, policy, episode.
MDP (S, A, P, R, gamma).
Q-learning update:
Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]
epsilon-greedy: explore with probability epsilon.
Applications: games, robotics, self-driving, recommender
systems.

============================================================
12. DATA MODELING AND VISUALIZATION
============================================================

Data modeling: organizing data and defining relationships,
structure and constraints (conceptual, logical, physical
models; ER diagrams).

Data visualization: graphical representation of data.

Chart selection:
* Compare categories  -> bar chart
* Part of whole       -> pie / stacked bar
* Trend over time     -> line chart
* Distribution        -> histogram, box, violin
* Relationship        -> scatter, bubble, heatmap
* Geography           -> maps

============================================================
13. FILE FORMATS AND WEB APIs
============================================================

CSV  - plain text, read_csv()
Excel- read_excel() (openpyxl)
JSON - nested key-value, json.load(), json_normalize()
Others: Parquet, HDF5, SQL, XML, pickle

Web API: interface over HTTP. REST uses GET, POST, PUT,
DELETE; data in JSON; API key for authentication.
requests.get(url, params) -> response.json().
Status codes: 200 OK, 401 Unauthorized, 404 Not Found,
429 Too Many Requests, 500 Server Error.

============================================================
14. DATA CLEANING, WRANGLING, AGGREGATION
============================================================

Cleaning: missing values, duplicates, inconsistent text,
wrong types, outliers.

Wrangling steps: discover, structure, clean, enrich
(merge), validate, publish.

Merge types: inner, left, right, outer.

Aggregation: split-apply-combine using groupby(),
agg(), pivot_table(); functions sum, mean, count, min,
max, median, std.

============================================================
15. TIME SERIES
============================================================

Components: trend, seasonality, cyclic, residual.
Moving average smooths noise.
Stationarity: constant mean/variance; ADF test p < 0.05.
ARIMA(p,d,q): AR lags, differencing, MA error lags.
Exponential smoothing: simple, Holt (trend),
Holt-Winters (trend + seasonality).

============================================================
16. PYTHON LIBRARIES
============================================================

numpy       - arrays, linear algebra
pandas      - DataFrames, cleaning, groupby
matplotlib  - basic plotting
seaborn     - statistical plots
scikit-learn- ML algorithms, preprocessing, metrics
statsmodels - statistics, time series (ARIMA)
requests    - HTTP / Web APIs
openpyxl    - Excel files

============================================================
17. IMPORTANT DEFINITIONS
============================================================

Feature        - input variable
Target / Label - output variable
Training set   - data used to train
Test set       - unseen data used to evaluate
Hyper-parameter- set before training (K, C, alpha)
Parameter      - learned during training (weights)
Overfitting    - good on train, poor on test
Underfitting   - poor on both
Bias           - error from wrong assumptions
Variance       - error from sensitivity to training data
Bias-variance trade-off - balance between the two
Outlier        - value far from others
Correlation    - strength of linear relationship (-1 to 1)
Centroid       - mean point of a cluster
Epoch          - one full pass over training data

============================================================
18. THEORETICAL VIVA QUESTIONS
============================================================

1. What is machine learning?
Ans: Learning patterns from data to make predictions
without explicit programming.

------------------------------------------------------------

2. Difference between supervised and unsupervised
   learning?
Ans: Supervised uses labelled data; unsupervised finds
patterns in unlabelled data.

------------------------------------------------------------

3. Difference between regression and classification?
Ans: Regression predicts continuous values;
classification predicts categories.

------------------------------------------------------------

4. What is PCA?
Ans: Unsupervised technique that projects data on
directions of maximum variance.

------------------------------------------------------------

5. What is overfitting and how to avoid it?
Ans: Model memorizes training data. Avoid with more data,
regularization, cross-validation, simpler models, pruning,
ensembles.

------------------------------------------------------------

6. Ridge vs Lasso?
Ans: Ridge L2 shrinks coefficients; Lasso L1 can make them
zero.

------------------------------------------------------------

7. What is a kernel in SVM?
Ans: A function computing similarity in a higher-
dimensional space (kernel trick).

------------------------------------------------------------

8. What is the elbow method?
Ans: Choosing K where WCSS curve bends.

------------------------------------------------------------

9. Bagging vs boosting?
Ans: Bagging parallel, reduces variance; boosting
sequential, reduces bias.

------------------------------------------------------------

10. What is Q-learning?
Ans: Model-free RL algorithm that learns action values
Q(s,a) using the Bellman update.

------------------------------------------------------------

11. What is a confusion matrix?
Ans: Table of actual vs predicted classes (TP, TN, FP, FN).

------------------------------------------------------------

12. Precision vs recall?
Ans: Precision = correctness of positive predictions;
recall = coverage of actual positives.

------------------------------------------------------------

13. What is cross-validation?
Ans: Repeated train/validate on different folds to
estimate performance reliably.

------------------------------------------------------------

14. What is data visualization and why is it important?
Ans: Graphical representation of data; reveals patterns
quickly and supports decisions.

------------------------------------------------------------

15. What is data wrangling?
Ans: Converting raw messy data into clean, structured,
enriched data.

------------------------------------------------------------

16. What is data aggregation?
Ans: Summarizing data by groups (sum, mean, count).

------------------------------------------------------------

17. What is a Web API?
Ans: Interface allowing programs to exchange data over
HTTP, usually JSON.

------------------------------------------------------------

18. CSV vs JSON?
Ans: CSV flat table; JSON nested key-value.

------------------------------------------------------------

19. What is a time series?
Ans: Data points recorded in time order at regular
intervals.

------------------------------------------------------------

20. What is ARIMA?
Ans: AutoRegressive Integrated Moving Average forecasting
model.

------------------------------------------------------------

21. What is normalization vs standardization?
Ans: Normalization -> [0,1]; standardization -> mean 0,
SD 1.

------------------------------------------------------------

22. What is one-hot encoding?
Ans: Creating one binary column per category.

------------------------------------------------------------

23. How do you detect outliers?
Ans: Box plot, IQR rule, Z-score (|z| > 3), domain rules.

------------------------------------------------------------

24. What is groupby in pandas?
Ans: Split-apply-combine operation for grouping and
aggregating.

------------------------------------------------------------

25. What is matplotlib?
Ans: Python 2-D plotting library.

------------------------------------------------------------

26. What is the bias-variance trade-off?
Ans: Simple models have high bias, complex models high
variance; best model balances both.

------------------------------------------------------------

27. What is gradient descent?
Ans: Iterative method to minimize cost by moving opposite
to the gradient.

------------------------------------------------------------

28. What is a hyper-parameter?
Ans: A setting chosen before training (K, C, gamma, alpha,
n_estimators).

------------------------------------------------------------

29. What is stationarity?
Ans: Constant mean and variance over time.

------------------------------------------------------------

30. What is feature engineering?
Ans: Creating new informative features from raw data.

============================================================
END OF FILE
============================================================
