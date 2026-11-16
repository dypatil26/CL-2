============================================================
ML1 - FEATURE TRANSFORMATION:
PCA (PRINCIPAL COMPONENT ANALYSIS) ON WINE DATASET
============================================================

## PROBLEM STATEMENT

To use PCA Algorithm for dimensionality reduction on the
wine dataset and visualize data after applying PCA.

You have a dataset that includes measurements for different
variables on wine (alcohol, ash, magnesium, and so on).
Apply PCA and transform this data so that most variation in
the measurements is captured by a small number of principal
components, so that it is easier to distinguish between
the wine classes by inspecting these components.

============================================================
DATASET
============================================================

Wine Dataset

Dataset File:
Wine.csv   (included in this folder)

Source (same file as in lab manual):
https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv

Rows    : 178 wines
Columns : 14

13 Features:
* Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium,
  Total_Phenols, Flavanoids, Nonflavanoid_Phenols,
  Proanthocyanins, Color_Intensity, Hue, OD280, Proline

Target:
* Customer_Segment  (1, 2, 3 = three wine classes)

============================================================
OBJECTIVE
============================================================

* To understand dimensionality reduction.
* To apply PCA on a real dataset.
* To find how many components keep most of the variance.
* To visualize 13-dimensional data in 2 dimensions.

============================================================
LIBRARIES USED
============================================================

1. pandas
   * Load CSV and handle DataFrames.

2. numpy
   * Numerical operations, covariance, eigenvalues.

3. matplotlib.pyplot
   * Scree plot and scatter plot.

4. sklearn.preprocessing.StandardScaler
   * Standardizes features (mean 0, SD 1).

5. sklearn.decomposition.PCA
   * Performs Principal Component Analysis.

============================================================
HOW TO RUN
============================================================

pip install pandas numpy matplotlib scikit-learn

python ml1_pca_wine.py

OR paste the code into Jupyter Notebook and run it.

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  Wine.csv
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

## Dimensionality Reduction

Dimensionality reduction means reducing the number of
input features while keeping as much information as
possible.

Why it is needed:
* Too many features -> "curse of dimensionality"
* Slower training and more memory
* Correlated (redundant) features
* Data with more than 3 features cannot be plotted

Two types:
* Feature Selection  -> keep some original features
* Feature Extraction -> create new features (PCA, LDA)

------------------------------------------------------------

## Principal Component Analysis (PCA)

PCA is an UNSUPERVISED linear dimensionality reduction
technique. It finds new axes (principal components) along
which the data varies the most.

* PC1 = direction of maximum variance
* PC2 = next maximum variance, perpendicular (orthogonal)
        to PC1
* ... and so on

The principal components are uncorrelated with each other.

------------------------------------------------------------

## Steps of PCA

1. Standardize the data
   z = (x - mean) / standard deviation

2. Compute the covariance matrix (13 x 13)

3. Compute eigenvalues and eigenvectors of the
   covariance matrix
   * Eigenvector  = direction of principal component
   * Eigenvalue   = amount of variance on that direction

4. Sort eigenvectors by decreasing eigenvalue

5. Choose top k eigenvectors (k components)

6. Project data onto them:
   New Data = Standardized Data x Eigenvector Matrix

------------------------------------------------------------

## Explained Variance Ratio

Explained variance ratio of a component =
   its eigenvalue / sum of all eigenvalues

Cumulative explained variance tells how much total
information is kept by the first k components.

------------------------------------------------------------

## Scree Plot

A bar/line chart of explained variance for each component.
The "elbow" shows where adding components stops helping.

------------------------------------------------------------

## Why Standardize Before PCA?

PCA is based on variance. Proline has values ~1000 while
Hue is ~1. Without scaling, Proline alone would dominate
PC1. StandardScaler gives every feature equal weight.

------------------------------------------------------------

## PCA vs LDA

PCA                         | LDA
--------------------------- | ---------------------------
Unsupervised (no labels)    | Supervised (uses labels)
Maximizes variance          | Maximizes class separation
Max components = features   | Max components = classes-1

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1

Code:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Explanation:
* Imports the required libraries.

------------------------------------------------------------

## CODE BLOCK 2

Code:

def upload_file(name, multiple=False):
    # 1. Google Colab  -> files.upload() (Choose files button)
    # 2. PC (Jupyter)  -> tkinter file-picker window
    # 3. otherwise     -> input() asking for the file path
    ...
    return path

Explanation:
* Asks the user to upload / select the dataset when the
  program runs, so the dataset can be kept anywhere and
  does not need to be in the program's folder.
* Works in Jupyter (whole program in one cell or block by
  block), Google Colab and plain Python.

------------------------------------------------------------

## CODE BLOCK 3

Code:

df = pd.read_csv(upload_file("Wine.csv"))
print(df.head())
print(df.isnull().sum())
print(df["Customer_Segment"].value_counts())

Explanation:
* Loads the dataset.
* Shows first 5 rows, missing values (none) and
  class counts (59, 71, 48).

------------------------------------------------------------

## CODE BLOCK 4

Code:

X = df.drop("Customer_Segment", axis=1)
y = df["Customer_Segment"]

Explanation:
* X = 13 input features.
* y = class label (used ONLY for colouring the plot,
  PCA itself does not use labels).

------------------------------------------------------------

## CODE BLOCK 5

Code:

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Explanation:
* Standardizes each column to mean 0 and SD 1.

------------------------------------------------------------

## CODE BLOCK 6

Code:

cov = np.cov(X_scaled.T)
eig_vals, eig_vecs = np.linalg.eig(cov)

Explanation:
* Manually computes covariance matrix and its
  eigenvalues - this shows the mathematics behind PCA.
* Largest eigenvalue = variance of PC1.

------------------------------------------------------------

## CODE BLOCK 7

Code:

pca_full = PCA()
pca_full.fit(X_scaled)
ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)

Explanation:
* PCA with all 13 components.
* ratio = variance explained by each component.
* cum   = cumulative variance.
* k95   = components needed to keep 95% variance.

------------------------------------------------------------

## CODE BLOCK 8

Code:

plt.bar(...ratio...)
plt.step(...cum...)

Explanation:
* Draws the scree plot (individual + cumulative).

------------------------------------------------------------

## CODE BLOCK 9

Code:

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

Explanation:
* Reduces 13 features to 2 principal components.
* fit_transform() learns the components and projects
  the data onto them.

------------------------------------------------------------

## CODE BLOCK 10

Code:

loadings = pd.DataFrame(pca.components_.T,
           columns=["PC1", "PC2"], index=X.columns)

Explanation:
* Loadings = weight of every original feature in each PC.
* Large absolute value = feature strongly affects that PC.
* Flavanoids, Total_Phenols, OD280 dominate PC1.
* Color_Intensity, Proline, Ash dominate PC2.

------------------------------------------------------------

## CODE BLOCK 11

Code:

for seg, c in colors.items():
    plt.scatter(part["PC1"], part["PC2"], c=c, ...)

Explanation:
* Scatter plot of PC1 vs PC2 coloured by wine class.
* The three classes form three clear groups.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. read_csv()                 - loads CSV file
2. drop()                     - removes target column
3. StandardScaler().fit_transform() - scales data
4. np.cov()                   - covariance matrix
5. np.linalg.eig()            - eigenvalues/eigenvectors
6. PCA(n_components=k)        - creates PCA object
7. fit_transform()            - fits PCA and projects data
8. explained_variance_ratio_  - variance of each PC
9. components_                - eigenvectors (loadings)
10. np.cumsum()               - cumulative variance
11. plt.scatter()             - scatter plot

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Load Wine.csv
     ↓
Separate X (13 features) and y (class)
     ↓
Standardize X
     ↓
Covariance Matrix → Eigenvalues / Eigenvectors
     ↓
PCA (all components) → Scree Plot
     ↓
PCA (2 components)
     ↓
Loadings Table
     ↓
2-D Scatter Plot coloured by class

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

PC1 variance = 0.3620  (36.2%)
PC2 variance = 0.1921  (19.2%)
Variance captured by 2 PCs = 55.41 %
Components needed to keep 95% variance = 10

Plots:
1. Scree plot - variance falls quickly after first PCs.
2. PCA scatter plot - classes 1, 2, 3 are clearly
   separated using only PC1 and PC2.

Meaning:
Even though only 55% of total variance is kept, the
CLASS-related information is almost fully preserved,
so 2 components are enough to distinguish the wines.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are the oral questions from the lab manual)

1. What all libraries are used for performing PCA?

Ans:
pandas (load data), numpy (math), matplotlib (plots),
sklearn.preprocessing.StandardScaler (scaling) and
sklearn.decomposition.PCA (PCA).

------------------------------------------------------------

2. How is the data imported in Python?

Ans:
Using pandas: df = pd.read_csv("Wine.csv").
It returns a DataFrame.

------------------------------------------------------------

3. Explain working of PCA.

Ans:
Standardize data -> covariance matrix -> eigenvalues and
eigenvectors -> sort by eigenvalue -> select top k
eigenvectors -> project data on them.

------------------------------------------------------------

4. Explain how PCA is visualized.

Ans:
Data is reduced to 2 components and plotted as a scatter
plot of PC1 vs PC2, coloured by class. A scree plot shows
variance per component.

------------------------------------------------------------

5. What is dimensionality reduction?

Ans:
Reducing number of features while keeping most
information.

------------------------------------------------------------

6. What is a principal component?

Ans:
A new axis that is a linear combination of original
features, in the direction of maximum variance.

------------------------------------------------------------

7. Is PCA supervised or unsupervised?

Ans:
Unsupervised - it does not use the target labels.

------------------------------------------------------------

8. What is an eigenvalue and eigenvector in PCA?

Ans:
Eigenvector gives the direction of a principal component.
Eigenvalue gives the variance along that direction.

------------------------------------------------------------

9. Why do we standardize before PCA?

Ans:
Because PCA depends on variance; features with large
units would dominate otherwise.

------------------------------------------------------------

10. What is explained variance ratio?

Ans:
Fraction of total variance captured by a component.

------------------------------------------------------------

11. What is a scree plot?

Ans:
Plot of variance explained by each component, used to
choose number of components.

------------------------------------------------------------

12. Are principal components correlated?

Ans:
No, they are orthogonal (uncorrelated).

------------------------------------------------------------

13. How many components did you choose and why?

Ans:
2 for visualization (55% variance, clear class
separation). 10 would keep 95% variance.

------------------------------------------------------------

14. What is the difference between PCA and LDA?

Ans:
PCA is unsupervised and maximizes variance; LDA is
supervised and maximizes separation between classes.

------------------------------------------------------------

15. What are loadings?

Ans:
Coefficients showing how much each original feature
contributes to a principal component.

------------------------------------------------------------

16. What is the curse of dimensionality?

Ans:
As features increase, data becomes sparse and models need
much more data and time; performance can drop.

------------------------------------------------------------

17. What are disadvantages of PCA?

Ans:
New features are hard to interpret, only captures linear
relationships, sensitive to scaling and outliers.

------------------------------------------------------------

18. What does fit_transform() do?

Ans:
Learns parameters from data (fit) and applies the
transformation (transform) in one step.

============================================================
CONCLUSION
============================================================

PCA was applied on the 13-feature Wine dataset. After
standardization, 2 principal components captured 55.4% of
variance and clearly separated the three wine classes.
Hence PCA effectively reduces dimensionality and helps
visualize high-dimensional data.

============================================================
END OF FILE
============================================================
