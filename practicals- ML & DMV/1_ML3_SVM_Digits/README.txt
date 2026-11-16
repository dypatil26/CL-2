============================================================
ML3 - CLASSIFICATION ANALYSIS:
SVM FOR HANDWRITTEN DIGIT RECOGNITION
============================================================

## PROBLEM STATEMENT

Implementation of Support Vector Machines (SVM) for
classifying images of hand-written digits into their
respective numerical classes (0 to 9).

============================================================
DATASET
============================================================

Digits Dataset (built into scikit-learn)

Loaded using:
from sklearn import datasets
digits = datasets.load_digits()

No download needed.

* 1797 images
* Each image = 8 x 8 pixels (grey levels 0-16)
* Flattened to 64 features per image
* Target = digit 0 to 9 (10 classes, ~180 each)

============================================================
OBJECTIVE
============================================================

* To understand Support Vector Machines.
* To classify images using linear and RBF kernels.
* To tune hyper-parameters C and gamma.
* To evaluate using accuracy, confusion matrix and
  classification report.

============================================================
LIBRARIES USED
============================================================

1. numpy, pandas         - arrays and DataFrame
2. matplotlib, seaborn   - show images and heatmap
3. sklearn.datasets      - load_digits()
4. sklearn.model_selection.train_test_split, GridSearchCV
5. sklearn.preprocessing.StandardScaler
6. sklearn.svm.SVC       - Support Vector Classifier
7. sklearn.metrics       - accuracy, confusion matrix,
                           classification report

============================================================
HOW TO RUN
============================================================

pip install numpy pandas matplotlib seaborn scikit-learn

python ml3_svm_digits.py

============================================================
THEORY
============================================================

## Support Vector Machine (SVM)

SVM is a supervised learning algorithm used for
classification (and regression). It finds the best
boundary (hyperplane) that separates classes with the
MAXIMUM MARGIN.

* Hyperplane: decision boundary
  (line in 2-D, plane in 3-D, hyperplane in n-D)
  w.x + b = 0
* Support vectors: data points closest to the hyperplane.
  Only these points decide the boundary.
* Margin: distance between hyperplane and nearest support
  vectors. SVM maximizes it -> better generalization.

------------------------------------------------------------

## Hard Margin vs Soft Margin

* Hard margin: no point allowed inside margin (only for
  perfectly separable data).
* Soft margin: allows some mistakes, controlled by C.

C (regularization parameter):
* Large C -> small margin, fewer training errors
  (risk of overfitting)
* Small C -> wide margin, more errors allowed
  (smoother boundary)

------------------------------------------------------------

## Kernel Trick

When data is not linearly separable, SVM maps it into a
higher dimension where it becomes separable - without
actually computing the new coordinates. This is the
kernel trick.

Common kernels:
* linear : K(x, y) = x . y
* poly   : K(x, y) = (x . y + c)^d
* rbf    : K(x, y) = exp(-gamma * ||x - y||^2)
* sigmoid

gamma (RBF): how far the influence of one point reaches.
* Large gamma -> very local, wiggly boundary (overfit)
* Small gamma -> smooth boundary

------------------------------------------------------------

## Multi-class SVM

SVM is binary by nature. For 10 digits, sklearn SVC uses
One-vs-One: it trains a classifier for every pair of
classes (45 classifiers) and takes a vote.

------------------------------------------------------------

## Steps of SVM Algorithm

1. Load and prepare data
2. Split into train and test
3. Scale features
4. Choose kernel and parameters
5. Train (find support vectors and hyperplane)
6. Predict and evaluate

------------------------------------------------------------

## GridSearchCV

Tries every combination of given parameter values using
k-fold cross-validation and returns the best combination.

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1

digits = datasets.load_digits()
numbers = pd.DataFrame(digits.data)
numbers["label"] = digits.target

* Loads 1797 images. digits.images is 8x8 form,
  digits.data is flattened 64-column form.

------------------------------------------------------------

## CODE BLOCK 2

for ax, img, lab in zip(axes.ravel(), digits.images, ...):
    ax.imshow(img, cmap="gray_r")

* Displays first 10 images with labels.

------------------------------------------------------------

## CODE BLOCK 3

X = numbers.drop("label", axis=1)
y = numbers["label"]
train_test_split(X, y, test_size=0.2, stratify=y)
StandardScaler()

* 80/20 split. stratify keeps class proportions equal.
* Scaling improves SVM performance.

------------------------------------------------------------

## CODE BLOCK 4

model_linear = SVC(kernel="linear")
model_linear.fit(X_train, y_train)

* Linear SVM.

------------------------------------------------------------

## CODE BLOCK 5

model_rbf = SVC(kernel="rbf")

* Non-linear SVM using RBF kernel.

------------------------------------------------------------

## CODE BLOCK 6

params = {"C": [1, 10, 100], "gamma": [0.001, 0.01, 0.1]}
grid = GridSearchCV(SVC(kernel="rbf"), params, cv=5)
grid.fit(X_train, y_train)

* Tests 9 combinations with 5-fold cross-validation.
* best_params_ and best_estimator_ give the best model.

------------------------------------------------------------

## CODE BLOCK 7

confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
sns.heatmap(cm, annot=True)

* Shows which digits are confused (e.g. 8 with 1 or 9).

------------------------------------------------------------

## CODE BLOCK 8

scaler.inverse_transform(X_test).reshape(-1, 8, 8)

* Converts test rows back to images and shows predicted
  vs true label (green = correct, red = wrong).

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. datasets.load_digits()
2. imshow()                 - display image
3. train_test_split()
4. StandardScaler()
5. SVC(kernel=...)          - SVM classifier
6. fit(), predict()
7. GridSearchCV()           - hyper-parameter tuning
8. accuracy_score()
9. confusion_matrix()
10. classification_report()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Load digits (8x8 images)
     ↓
Visualize samples
     ↓
Flatten → 64 features, split, scale
     ↓
Linear SVM → accuracy
     ↓
RBF SVM → accuracy
     ↓
GridSearchCV (C, gamma)
     ↓
Confusion matrix + report + predicted images

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Linear kernel accuracy : 0.975
RBF kernel accuracy    : 0.975
Best parameters        : C = 100, gamma = 0.01
Test accuracy (best)   : 0.9833 (98.3%)

Confusion matrix is almost diagonal - only a few digits
(like 8 and 9) are misclassified.

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What is an SVM algorithm?

Ans:
A supervised algorithm that finds the hyperplane which
separates classes with maximum margin.

------------------------------------------------------------

2. Why is SVM the best algorithm (for this task)?

Ans:
It works well in high-dimensional spaces, is effective
when features > samples, uses only support vectors
(memory efficient), and kernels handle non-linear data.
It gives ~98% accuracy on digits.

------------------------------------------------------------

3. What are the steps of SVM algorithm?

Ans:
Load data -> split -> scale -> select kernel and C/gamma
-> train (find support vectors and maximum margin
hyperplane) -> predict -> evaluate.

------------------------------------------------------------

4. What does SVM do in machine learning?

Ans:
It classifies data (or does regression - SVR) by finding
the optimal decision boundary.

------------------------------------------------------------

5. What are support vectors?

Ans:
Data points nearest to the hyperplane that define the
margin.

------------------------------------------------------------

6. What is margin?

Ans:
Distance between the hyperplane and nearest support
vectors of each class.

------------------------------------------------------------

7. What is kernel trick?

Ans:
Computing similarity in a higher-dimensional space without
explicitly transforming the data.

------------------------------------------------------------

8. What is the role of C?

Ans:
Trade-off between wide margin and classification errors.

------------------------------------------------------------

9. What is gamma?

Ans:
RBF parameter controlling how far one training point's
influence reaches.

------------------------------------------------------------

10. How does SVM handle multi-class problems?

Ans:
One-vs-One (sklearn SVC default) or One-vs-Rest.

------------------------------------------------------------

11. What is GridSearchCV?

Ans:
Exhaustive search over parameter values using
cross-validation.

------------------------------------------------------------

12. What is cross-validation?

Ans:
Splitting training data into k folds, training on k-1 and
validating on 1, repeated k times.

------------------------------------------------------------

13. How many features does each digit image have?

Ans:
64 (8 x 8 pixels).

------------------------------------------------------------

14. What does the confusion matrix show?

Ans:
Counts of actual vs predicted classes; diagonal = correct.

------------------------------------------------------------

15. What are precision, recall, F1-score?

Ans:
Precision = TP/(TP+FP), Recall = TP/(TP+FN),
F1 = harmonic mean of precision and recall.

------------------------------------------------------------

16. Disadvantages of SVM?

Ans:
Slow on very large datasets, sensitive to C/gamma and
scaling, no direct probability output.

============================================================
CONCLUSION
============================================================

SVM was implemented on the handwritten digits dataset.
Linear and RBF kernels gave 97.5% accuracy and after
tuning (C=100, gamma=0.01) the RBF SVM reached 98.3%
accuracy. Hence SVM is highly effective for image
classification.

============================================================
END OF FILE
============================================================
