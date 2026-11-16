============================================================
ML4 - CLUSTERING ANALYSIS:
K-MEANS CLUSTERING ON IRIS WITH ELBOW METHOD
============================================================

## PROBLEM STATEMENT

Implement K-Means clustering on Iris.csv dataset.
Determine the number of clusters using the elbow method.

============================================================
DATASET
============================================================

Iris Dataset

Dataset File:
Iris.csv   (included in this folder)

Source:
https://www.kaggle.com/datasets/uciml/iris
(same columns as the Kaggle file)

Rows: 150 flowers (50 of each species)

Columns:
* Id
* SepalLengthCm, SepalWidthCm
* PetalLengthCm, PetalWidthCm
* Species (Iris-setosa, Iris-versicolor, Iris-virginica)

Species is NOT used for clustering (unsupervised). It is
only used at the end to check the clusters.

============================================================
OBJECTIVE
============================================================

* To understand unsupervised learning and clustering.
* To implement the K-Means algorithm.
* To find the optimal K using the Elbow method.
* To visualize clusters and centroids.

============================================================
LIBRARIES USED
============================================================

1. pandas                 - load data
2. matplotlib             - elbow plot, cluster plot
3. sklearn.cluster.KMeans - K-Means algorithm
4. sklearn.preprocessing.StandardScaler
5. sklearn.metrics.silhouette_score

============================================================
HOW TO RUN
============================================================

pip install pandas matplotlib scikit-learn

python ml4_kmeans_iris.py

------------------------------------------------------------
UPLOADING THE DATASET
------------------------------------------------------------
The program does NOT need the dataset to be in the same
folder. When it reaches the step that loads data, it asks
you to upload it. Upload:
  Iris.csv
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

## Clustering

Clustering is an unsupervised learning technique that
groups similar data points together. There are no labels.

Points in the same cluster are similar; points in
different clusters are different.

------------------------------------------------------------

## K-Means Algorithm

K-Means is a centroid-based clustering algorithm that
divides data into K non-overlapping clusters.

Steps:
1. Choose K (number of clusters).
2. Initialize K centroids (randomly or k-means++).
3. Assignment: assign each data point to the nearest
   centroid (Euclidean distance).
4. Update: recompute each centroid as the mean of the
   points assigned to it.
5. Repeat steps 3-4 until centroids stop changing
   (convergence).

Euclidean distance:
d = sqrt( (x1 - c1)^2 + (x2 - c2)^2 + ... )

------------------------------------------------------------

## Objective: WCSS / Inertia

WCSS = Within-Cluster Sum of Squares
     = SUM over clusters SUM over points ||x - centroid||^2

K-Means tries to minimize WCSS.

------------------------------------------------------------

## Elbow Method

Run K-Means for K = 1 to 10 and plot WCSS vs K.
WCSS always decreases as K increases. The point where
the decrease slows sharply (looks like an elbow) is the
best K. For Iris, elbow is at K = 3.

------------------------------------------------------------

## Silhouette Score

s = (b - a) / max(a, b)
a = mean distance to points in own cluster
b = mean distance to points in nearest other cluster
Range -1 to +1, higher is better.

------------------------------------------------------------

## k-means++

Smart initialization: picks initial centroids far apart,
giving faster and better convergence than random.

------------------------------------------------------------

## Advantages / Disadvantages

+ Simple, fast, scalable
- Need to choose K
- Sensitive to initialization and outliers
- Assumes round (spherical), similar-size clusters

------------------------------------------------------------

## K-Means vs K-Medians vs K-Medoids

* K-Means  : centre = mean (sensitive to outliers)
* K-Medians: centre = median, Manhattan distance
             (robust to outliers)
* K-Medoids: centre = an actual data point

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1

data = pd.read_csv(upload_file("Iris.csv"))

* Loads the dataset.

------------------------------------------------------------

## CODE BLOCK 2

X = data[["SepalLengthCm", "SepalWidthCm",
          "PetalLengthCm", "PetalWidthCm"]]
X_scaled = StandardScaler().fit_transform(X)

* Only the 4 measurements are used.
* Scaling gives each feature equal importance in distance.

------------------------------------------------------------

## CODE BLOCK 3 - Elbow Method

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++",
                    n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

* Runs K-Means for K = 1..10.
* inertia_ = WCSS.
* n_init=10: algorithm is run 10 times with different
  starting centroids and best result is kept.

------------------------------------------------------------

## CODE BLOCK 4

plt.plot(range(1, 11), wcss, marker="o")

* Elbow plot - bend at K = 3.

------------------------------------------------------------

## CODE BLOCK 5

silhouette_score(X_scaled, labels)

* Extra check of cluster quality for K = 2..6.

------------------------------------------------------------

## CODE BLOCK 6

kmeans = KMeans(n_clusters=3, ...)
y_kmeans = kmeans.fit_predict(X_scaled)

* Final model with K = 3.
* fit_predict() returns cluster number of each flower.
* cluster_centers_ gives centroids (converted back to cm).

------------------------------------------------------------

## CODE BLOCK 7

pd.crosstab(data["Species"], data["Cluster"])

* Compares clusters with real species.

------------------------------------------------------------

## CODE BLOCK 8

plt.scatter(...petal length..., ...petal width...)
plt.scatter(centers[:, 2], centers[:, 3], marker="X")

* Plots the 3 clusters and centroids.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. read_csv()
2. StandardScaler().fit_transform()
3. KMeans(n_clusters, init, n_init, random_state)
4. fit(), fit_predict()
5. inertia_            - WCSS
6. cluster_centers_    - centroids
7. silhouette_score()
8. pd.crosstab()
9. plt.plot(), plt.scatter()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Load Iris.csv
     ↓
Select 4 features, scale
     ↓
K = 1..10 → WCSS → Elbow plot (K = 3)
     ↓
Silhouette check
     ↓
K-Means with K = 3
     ↓
Centroids + crosstab with species
     ↓
Cluster scatter plot

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Elbow at K = 3 (WCSS = 139.82)
Silhouette: K=2 0.58, K=3 0.46 (K=2 merges versicolor and
virginica, but elbow and domain knowledge say 3)

Crosstab:
                  Cluster 0  Cluster 1  Cluster 2
Iris-setosa            0        50         0
Iris-versicolor       39         0        11
Iris-virginica        14         0        36

* Setosa is perfectly separated.
* Versicolor and virginica overlap slightly.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What is clustering?

Ans:
Unsupervised grouping of similar data points into clusters
without labels.

------------------------------------------------------------

2. Where is clustering used in real life?

Ans:
Customer segmentation, document grouping, image
segmentation, anomaly/fraud detection, recommendation
systems, social network analysis.

------------------------------------------------------------

3. When to use k-means and k-medians?

Ans:
K-Means when data has few outliers and roughly spherical
clusters. K-Medians when data has outliers, because
median is robust (uses Manhattan distance).

------------------------------------------------------------

4. What is the difference between classification and
   clustering?

Ans:
Classification is supervised (labels known, predicts
predefined classes). Clustering is unsupervised (no
labels, discovers groups).

------------------------------------------------------------

5. What is a centroid?

Ans:
The mean point of all points in a cluster.

------------------------------------------------------------

6. What is WCSS?

Ans:
Within-Cluster Sum of Squares - sum of squared distances
of points from their centroid.

------------------------------------------------------------

7. Explain the elbow method.

Ans:
Plot WCSS vs K; the K where the curve bends (elbow) is
chosen.

------------------------------------------------------------

8. What is k-means++?

Ans:
An initialization method that spreads initial centroids
far apart.

------------------------------------------------------------

9. What distance does K-Means use?

Ans:
Euclidean distance.

------------------------------------------------------------

10. Why scale data before K-Means?

Ans:
Distance is affected by feature units; scaling makes all
features equally important.

------------------------------------------------------------

11. When does K-Means stop?

Ans:
When centroids do not change (or max iterations reached).

------------------------------------------------------------

12. What is silhouette score?

Ans:
Measure of how well a point fits its cluster compared to
other clusters (-1 to 1).

------------------------------------------------------------

13. Limitations of K-Means?

Ans:
Need K in advance, sensitive to outliers and initial
centroids, assumes spherical clusters.

------------------------------------------------------------

14. Why did versicolor and virginica overlap?

Ans:
Their petal and sepal measurements are very similar, so
distance-based clustering cannot separate them perfectly.

============================================================
CONCLUSION
============================================================

K-Means clustering was implemented on the Iris dataset.
The elbow method showed that the optimal number of
clusters is 3. Setosa formed a perfect cluster while
versicolor and virginica overlapped slightly. Hence
K-Means effectively groups data without labels.

============================================================
END OF FILE
============================================================
