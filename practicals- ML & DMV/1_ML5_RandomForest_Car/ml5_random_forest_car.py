import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------------------
# Upload helper: the program asks for its dataset when this runs.
#   * Google Colab            -> shows Colab's "Choose files" button
#   * Jupyter / Python on PC  -> opens a file-picker window
#   * anything else           -> asks you to type the file's path
# The dataset can be kept anywhere on the computer.
# ------------------------------------------------------------------
def upload_file(name, multiple=False):
    print(f"\n>>> Please upload / select: {name}")
    try:
        from google.colab import files                  # only exists on Google Colab
        paths = list(files.upload().keys())
        return paths if multiple else paths[0]
    except ImportError:
        pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)               # keep the picker in front
        ext = name[name.rfind("."):] if ("." in name and " " not in name) else ".*"
        types = [("Data files", "*" + ext), ("All files", "*.*")]
        if multiple:
            paths = list(filedialog.askopenfilenames(title="Select " + name, filetypes=types, parent=root))
        else:
            paths = filedialog.askopenfilename(title="Select " + name, filetypes=types, parent=root)
        root.destroy()
        if paths:
            print("Selected:", paths)
            return paths
    except Exception:
        pass
    typed = input(f"Type the full path of {name}: ").strip()
    if multiple:
        return [p.strip().strip('"') for p in typed.split(",")]
    return typed.strip('"')

print("=== Step 1: Load Car Evaluation Dataset ===")
df = pd.read_csv(upload_file("car_evaluation.csv"))
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
for col in df.columns:
    print(f"{col:<9}: {sorted(df[col].unique())}")
print("\nTarget distribution (class):\n", df["class"].value_counts())

print("\n=== Step 2: Encode Categorical Features (ordinal order kept) ===")
order = [
    ["low", "med", "high", "vhigh"],      # buying
    ["low", "med", "high", "vhigh"],      # maint
    ["2", "3", "4", "5more"],             # doors
    ["2", "4", "more"],                   # persons
    ["small", "med", "big"],              # lug_boot
    ["low", "med", "high"],               # safety
]
X = df.drop("class", axis=1).astype(str)
y = df["class"]
enc = OrdinalEncoder(categories=order)
X_enc = pd.DataFrame(enc.fit_transform(X), columns=X.columns)
print(X_enc.head())

print("\n=== Step 3: Train/Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.25, random_state=42, stratify=y)
print("Train:", X_train.shape, " Test:", X_test.shape)

print("\n=== Step 4: Baseline - Single Decision Tree ===")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Decision Tree accuracy:", round(accuracy_score(y_test, dt.predict(X_test)), 4))

print("\n=== Step 5: Random Forest Classifier ===")
rf = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("5-fold CV accuracy    :", round(cross_val_score(rf, X_enc, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean(), 4))

print("\n=== Step 6: Effect of Number of Trees ===")
for n in [1, 10, 50, 100, 200]:
    m = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train, y_train)
    print(f"n_estimators = {n:<3}  accuracy = {accuracy_score(y_test, m.predict(X_test)):.4f}")

print("\n=== Step 7: Evaluation ===")
labels = ["unacc", "acc", "good", "vgood"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n=== Step 8: Feature Importance ===")
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.round(4))
imp.plot(kind="bar", color="teal", figsize=(7, 4), title="Feature Importance (safety is most important)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

print("\n=== Step 9: Predict a New Car ===")
new_car = pd.DataFrame([["low", "low", "4", "4", "big", "high"]], columns=X.columns)
print(new_car)
print("Predicted class:", rf.predict(pd.DataFrame(enc.transform(new_car), columns=X.columns))[0])
