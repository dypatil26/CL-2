import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=== Step 1: Load the Digits Dataset (built into scikit-learn) ===")
digits = datasets.load_digits()
print("Images shape:", digits.images.shape, " (1797 images of 8x8 pixels)")
print("Data shape  :", digits.data.shape, " (each image flattened to 64 features)")
print("Classes     :", np.unique(digits.target))

numbers = pd.DataFrame(digits.data)
numbers["label"] = digits.target
print("\nClass counts:\n", numbers["label"].value_counts().sort_index())

print("\n=== Step 2: Visualize Sample Images ===")
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, img, lab in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"Label: {lab}")
    ax.axis("off")
plt.suptitle("Sample Handwritten Digits")
plt.tight_layout()
plt.show()

print("\n=== Step 3: Split and Scale ===")
X = numbers.drop("label", axis=1)
y = numbers["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Train:", X_train.shape, " Test:", X_test.shape)

print("\n=== Step 4: Linear SVM ===")
model_linear = SVC(kernel="linear")
model_linear.fit(X_train, y_train)
y_pred_lin = model_linear.predict(X_test)
print("Linear kernel accuracy:", round(accuracy_score(y_test, y_pred_lin), 4))

print("\n=== Step 5: Non-linear SVM (RBF kernel) ===")
model_rbf = SVC(kernel="rbf")
model_rbf.fit(X_train, y_train)
y_pred_rbf = model_rbf.predict(X_test)
print("RBF kernel accuracy:", round(accuracy_score(y_test, y_pred_rbf), 4))

print("\n=== Step 6: Hyper-parameter Tuning (GridSearchCV on C and gamma) ===")
params = {"C": [1, 10, 100], "gamma": [0.001, 0.01, 0.1]}
grid = GridSearchCV(SVC(kernel="rbf"), params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 4))
best = grid.best_estimator_
y_pred = best.predict(X_test)
print("Test accuracy with best model:", round(accuracy_score(y_test, y_pred), 4))

print("\n=== Step 7: Evaluation ===")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM (RBF)")
plt.tight_layout()
plt.show()

print("\n=== Step 8: Show Some Predictions ===")
X_test_imgs = scaler.inverse_transform(X_test).reshape(-1, 8, 8)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, img, true, pred in zip(axes.ravel(), X_test_imgs, y_test, y_pred):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"T:{true} P:{pred}", color="green" if true == pred else "red")
    ax.axis("off")
plt.suptitle("SVM Predictions (T = true, P = predicted)")
plt.tight_layout()
plt.show()
