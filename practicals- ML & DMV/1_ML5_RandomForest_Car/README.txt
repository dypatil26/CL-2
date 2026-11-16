============================================================
ML5 - ENSEMBLE LEARNING:
RANDOM FOREST CLASSIFIER FOR CAR SAFETY
============================================================

## PROBLEM STATEMENT

Implement Random Forest Classifier model to predict the
safety of the car.

============================================================
DATASET
============================================================

Car Evaluation Dataset (UCI)

Dataset File:
car_evaluation.csv   (included in this folder)

Source:
https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set
(original: UCI Machine Learning Repository)

Rows: 1728 cars, no missing values. All features are
categorical.

Features:
* buying   : vhigh, high, med, low   (buying price)
* maint    : vhigh, high, med, low   (maintenance cost)
* doors    : 2, 3, 4, 5more
* persons  : 2, 4, more              (capacity)
* lug_boot : small, med, big         (luggage boot size)
* safety   : low, med, high          (estimated safety)

Target (class):
* unacc (1210), acc (384), good (69), vgood (65)

NOTE: The Kaggle file has no header row. The included file
has a header added. If you use the Kaggle file, read it
with:
pd.read_csv("car_evaluation.csv", header=None,
            names=["buying","maint","doors","persons",
                   "lug_boot","safety","class"])

============================================================
OBJECTIVE
============================================================

* To understand ensemble learning and bagging.
* To implement Random Forest.
* To compare it with a single Decision Tree.
* To evaluate and find important features.

============================================================
LIBRARIES USED
============================================================

1. pandas                  - data handling
2. matplotlib, seaborn     - plots
3. sklearn.preprocessing.OrdinalEncoder
4. sklearn.model_selection.train_test_split,
   cross_val_score, StratifiedKFold
5. sklearn.tree.DecisionTreeClassifier
6. sklearn.ensemble.RandomForestClassifier
7. sklearn.metrics

============================================================
HOW TO RUN
============================================================

pip install pandas matplotlib seaborn scikit-learn

python ml5_random_forest_car.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  car_evaluation.csv
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

## Ensemble Learning

Ensemble learning combines many models (weak learners) to
build one stronger model. "Wisdom of the crowd".

Types:
* Bagging  (Bootstrap Aggregating) - e.g. Random Forest
* Boosting - e.g. AdaBoost, Gradient Boosting, XGBoost
* Stacking / Voting

------------------------------------------------------------

## Decision Tree

A tree of if-else questions on features. Each internal
node splits data on a feature, leaves give the class.
Split quality measured by:

Gini impurity = 1 - SUM (p_i^2)
Entropy       = - SUM p_i log2(p_i)

Problem: a single deep tree overfits (high variance).

------------------------------------------------------------

## Random Forest

Random Forest = many decision trees + two kinds of
randomness + majority voting.

1. Bootstrap sampling (bagging):
   Each tree is trained on a random sample of the data
   drawn WITH replacement (same size as dataset). About
   63% unique rows appear in each sample; the remaining
   ~37% are "out-of-bag" rows.

2. Feature randomness:
   At every split, only a random subset of features is
   considered (default sqrt(total features)). This makes
   trees different from each other (de-correlated).

3. Aggregation:
   Classification -> majority vote of all trees.
   Regression -> average.

Algorithm:
For b = 1 to B (number of trees):
   * Draw bootstrap sample D_b from dataset D of size N
   * Grow tree T_b; at each node pick m random features
     and choose best split among them
Prediction = majority vote of T_1 ... T_B

------------------------------------------------------------

## Why Random Forest over a single tree?

* Lower variance -> less overfitting
* More accurate and stable
* Gives feature importance
* Handles many features, robust to noise

------------------------------------------------------------

## Effect of number of trees (n_estimators)

More trees -> better and more stable accuracy up to a
point, then accuracy stays flat; only training time
increases. Random Forest does not overfit by adding trees.

------------------------------------------------------------

## Encoding Categorical Data

ML models need numbers. Since these categories have a
natural order (low < med < high < vhigh), ORDINAL encoding
is used, keeping the order.

------------------------------------------------------------

## Feature Importance

Measures how much each feature reduces Gini impurity
across all trees.

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1

df = pd.read_csv(upload_file("car_evaluation.csv"))
for col in df.columns: print(sorted(df[col].unique()))

* Loads data and prints the categories of every column.

------------------------------------------------------------

## CODE BLOCK 2

order = [["low","med","high","vhigh"], ...]
enc = OrdinalEncoder(categories=order)
X_enc = enc.fit_transform(X)

* Converts categories to numbers keeping the order,
  e.g. safety: low=0, med=1, high=2.

------------------------------------------------------------

## CODE BLOCK 3

train_test_split(X_enc, y, test_size=0.25, stratify=y)

* 75% train, 25% test, class ratios preserved.

------------------------------------------------------------

## CODE BLOCK 4

dt = DecisionTreeClassifier(random_state=42)

* Baseline single tree for comparison.

------------------------------------------------------------

## CODE BLOCK 5

rf = RandomForestClassifier(n_estimators=100,
                            criterion="gini",
                            random_state=42)
rf.fit(X_train, y_train)
cross_val_score(rf, X_enc, y,
     cv=StratifiedKFold(5, shuffle=True, random_state=42))

* 100 trees, Gini criterion.
* 5-fold cross validation gives a more reliable accuracy.

------------------------------------------------------------

## CODE BLOCK 6

for n in [1, 10, 50, 100, 200]:
    RandomForestClassifier(n_estimators=n)

* Shows accuracy improving with more trees then stabilizing.

------------------------------------------------------------

## CODE BLOCK 7

confusion_matrix, classification_report, heatmap

* Evaluation of predictions for all 4 classes.

------------------------------------------------------------

## CODE BLOCK 8

rf.feature_importances_

* safety and persons are the most important features.

------------------------------------------------------------

## CODE BLOCK 9

new_car = ["low","low","4","4","big","high"]
rf.predict(enc.transform(new_car))

* Predicts class of a new car -> vgood.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. OrdinalEncoder()
2. train_test_split()
3. DecisionTreeClassifier()
4. RandomForestClassifier(n_estimators, criterion)
5. fit(), predict()
6. cross_val_score(), StratifiedKFold()
7. accuracy_score(), confusion_matrix(),
   classification_report()
8. feature_importances_

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Load car_evaluation.csv
     ↓
Explore categories and class distribution
     ↓
Ordinal encoding
     ↓
Train/Test split
     ↓
Decision Tree (baseline) | Random Forest (100 trees)
     ↓
Cross validation + effect of number of trees
     ↓
Confusion matrix, report, feature importance
     ↓
Predict new car

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Decision Tree accuracy : 0.9884
Random Forest accuracy : 0.9884
5-fold CV accuracy     : 0.9826

n_estimators = 1   -> 0.9282
n_estimators = 10  -> 0.9745
n_estimators = 50  -> 0.9884
n_estimators = 100 -> 0.9884

Feature importance:
safety 0.305 > persons 0.239 > maint 0.150 > buying 0.143
> lug_boot 0.096 > doors 0.067

New car (low, low, 4, 4, big, high) -> vgood

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What is the primary reason to use Random Forest over a
   single decision tree?

Ans:
A single tree overfits (high variance). Random Forest
averages many de-correlated trees, reducing variance and
giving more accurate, stable predictions.

------------------------------------------------------------

2. How does increasing the number of trees in a Random
   Forest impact its performance?

Ans:
Accuracy improves and becomes stable up to a point, after
which it stays flat while training time and memory
increase. It does not cause overfitting.

------------------------------------------------------------

3. Explain the concept of bootstrap sampling.

Ans:
Creating a new training set of the same size by randomly
picking rows WITH replacement. Some rows repeat and ~37%
are left out (out-of-bag samples).

------------------------------------------------------------

4. Why is feature randomness introduced during the
   tree-building process in Random Forest?

Ans:
So that trees do not all split on the same strong feature;
this makes trees different (de-correlated), and averaging
de-correlated trees reduces variance more.

------------------------------------------------------------

5. What is ensemble learning?

Ans:
Combining multiple models to get better performance than
any single model.

------------------------------------------------------------

6. Difference between bagging and boosting?

Ans:
Bagging trains models in parallel on bootstrap samples and
averages (reduces variance). Boosting trains models
sequentially, each fixing previous errors (reduces bias).

------------------------------------------------------------

7. What is Gini impurity?

Ans:
1 - SUM p_i^2; measures how mixed classes are at a node.
0 = pure.

------------------------------------------------------------

8. What is out-of-bag (OOB) error?

Ans:
Error measured on rows not included in a tree's bootstrap
sample - a free validation estimate.

------------------------------------------------------------

9. Why ordinal encoding and not one-hot here?

Ans:
Categories have natural order (low < med < high); ordinal
encoding keeps that order with fewer columns.

------------------------------------------------------------

10. Which feature is most important for car evaluation?

Ans:
safety, followed by persons (capacity).

------------------------------------------------------------

11. What are hyper-parameters of Random Forest?

Ans:
n_estimators, max_depth, max_features, min_samples_split,
min_samples_leaf, criterion, bootstrap.

------------------------------------------------------------

12. Is the dataset balanced?

Ans:
No - unacc has 1210 rows, vgood only 65. stratify keeps
the ratio in train/test.

------------------------------------------------------------

13. Disadvantages of Random Forest?

Ans:
Slower and larger than single tree, less interpretable.

============================================================
CONCLUSION
============================================================

A Random Forest Classifier was built on the Car Evaluation
dataset after ordinal encoding. It achieved about 98.8%
test accuracy and 98.3% cross-validated accuracy. Safety
and passenger capacity were the most important features.
Hence ensemble learning with Random Forest accurately
predicts car acceptability/safety.

============================================================
END OF FILE
============================================================
