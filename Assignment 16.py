import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.svm import SVC

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# DATASET INFORMATION
# ==================================================

# Dataset Name:
# Iris Dataset / Any Kaggle Classification Dataset

# Kaggle Link Example:
# https://www.kaggle.com/datasets/uciml/iris

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==================================================
# PREPROCESSING
# ==================================================

target_column = df.columns[-1]

X = df.drop(columns=[target_column])

y = df[target_column]

# Convert categorical columns

X = pd.get_dummies(
    X,
    drop_first=True
)

# Convert target if categorical

if y.dtype == "object":
    y = pd.factorize(y)[0]

# ==================================================
# PART 1 : ADVANCED SUPERVISED ALGORITHMS
# ==================================================

# ==================================================
# Task 1 : Support Vector Machine (SVM)
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Linear Kernel

svm_linear = SVC(
    kernel='linear'
)

svm_linear.fit(
    X_train,
    y_train
)

linear_pred = svm_linear.predict(
    X_test
)

linear_accuracy = accuracy_score(
    y_test,
    linear_pred
)

# RBF Kernel

svm_rbf = SVC(
    kernel='rbf'
)

svm_rbf.fit(
    X_train,
    y_train
)

rbf_pred = svm_rbf.predict(
    X_test
)

rbf_accuracy = accuracy_score(
    y_test,
    rbf_pred
)

print("\nSVM Results")

print("Linear Kernel Accuracy:",
      linear_accuracy)

print("RBF Kernel Accuracy:",
      rbf_accuracy)

# ==================================================
# Task 2 : Decision Tree
# ==================================================

tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(
    X_test
)

tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)

print("\nDecision Tree Accuracy:",
      tree_accuracy)

# Tree Visualization

plt.figure(figsize=(12,8))

plot_tree(
    tree_model,
    filled=True
)

plt.title("Decision Tree")
plt.show()

# Underfitting Tree

underfit_tree = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

underfit_tree.fit(
    X_train,
    y_train
)

underfit_train_acc = underfit_tree.score(
    X_train,
    y_train
)

underfit_test_acc = underfit_tree.score(
    X_test,
    y_test
)

# Overfitting Tree

overfit_tree = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

overfit_tree.fit(
    X_train,
    y_train
)

overfit_train_acc = overfit_tree.score(
    X_train,
    y_train
)

overfit_test_acc = overfit_tree.score(
    X_test,
    y_test
)

print("\nUnderfitting Tree")

print("Train Accuracy:",
      underfit_train_acc)

print("Test Accuracy:",
      underfit_test_acc)

print("\nOverfitting Tree")

print("Train Accuracy:",
      overfit_train_acc)

print("Test Accuracy:",
      overfit_test_acc)

# ==================================================
# PART 2 : MODEL VALIDATION
# ==================================================

# ==================================================
# Task 3 : Train Validation Test Split
# ==================================================

X_train,
X_temp,
y_train,
y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

X_val,
X_test,
y_val,
y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

best_depth = 1
best_accuracy = 0

for depth in [2, 3, 4, 5]:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    val_pred = model.predict(
        X_val
    )

    acc = accuracy_score(
        y_val,
        val_pred
    )

    if acc > best_accuracy:

        best_accuracy = acc
        best_depth = depth

print("\nBest Depth From Validation:",
      best_depth)

final_model = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

final_model.fit(
    X_train,
    y_train
)

test_pred = final_model.predict(
    X_test
)

final_accuracy = accuracy_score(
    y_test,
    test_pred
)

print("Final Test Accuracy:",
      final_accuracy)

# ==================================================
# Task 4 : Cross Validation
# ==================================================

cv_scores = cross_val_score(
    final_model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nAverage CV Accuracy:")
print(cv_scores.mean())

print("\nSingle Test Accuracy:")
print(final_accuracy)

# ==================================================
# PART 3 : ENSEMBLE LEARNING
# ==================================================

# ==================================================
# Task 5 : Bagging vs Boosting
# ==================================================

print("""

Bagging:
Multiple models trained independently.
Final prediction uses voting/averaging.

Boosting:
Models trained sequentially.
Each new model corrects previous errors.

""")

# Bagging Classifier

bagging_model = BaggingClassifier(
    n_estimators=50,
    random_state=42
)

bagging_model.fit(
    X_train,
    y_train
)

bagging_pred = bagging_model.predict(
    X_test
)

bagging_accuracy = accuracy_score(
    y_test,
    bagging_pred
)

# AdaBoost

adaboost_model = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

adaboost_model.fit(
    X_train,
    y_train
)

adaboost_pred = adaboost_model.predict(
    X_test
)

adaboost_accuracy = accuracy_score(
    y_test,
    adaboost_pred
)

print("\nBagging Accuracy:",
      bagging_accuracy)

print("AdaBoost Accuracy:",
      adaboost_accuracy)

# ==================================================
# Task 6 : Random Forest
# ==================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("\nPerformance Comparison")

print("Decision Tree:",
      tree_accuracy)

print("Bagging:",
      bagging_accuracy)

print("Random Forest:",
      rf_accuracy)

# Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")

print(importance)

# Feature Importance Plot

plt.figure(figsize=(10,5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")

plt.show()

# ==================================================
# Classification Report
# ==================================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)

print("\nAssignment 16 Completed Successfully")
