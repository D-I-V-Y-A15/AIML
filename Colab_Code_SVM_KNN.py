# =====================================================================
# Lab Assignment 4 : Classification Model Training (SVM and KNN)
# Dataset : Breast Cancer Wisconsin (Diagnostic) - built into scikit-learn
# Works directly in Google Colab - no file upload needed.
# =====================================================================

# ---------- 1. Import libraries ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

import warnings; warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 7

# ---------- 2. Load and inspect the dataset ----------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# In sklearn: 0 = malignant, 1 = benign
class_names = list(data.target_names)          # ['malignant', 'benign']

print("Shape               :", df.shape)
print("Missing values      :", int(df.isnull().sum().sum()))
print("Duplicate rows      :", int(df.duplicated().sum()))
print("\nClass distribution:")
print(df["target"].map({0: "malignant", 1: "benign"}).value_counts())
display(df.describe().T.head(8))

# Class balance plot
plt.figure(figsize=(4.5, 3.5))
sns.countplot(x=df["target"].map({0: "malignant", 1: "benign"}), palette="Set2")
plt.title("Class distribution"); plt.xlabel(""); plt.tight_layout()
plt.savefig("fig_class_balance.png", dpi=200); plt.show()

# ---------- 3. Split data (80:20, stratified) ----------
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print("\nTrain size:", X_train.shape[0], " Test size:", X_test.shape[0])

# ---------- 4. Pipelines (scaling is inside the pipeline -> no data leakage) ----------
svm_pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(probability=True, random_state=RANDOM_STATE))])
knn_pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier())])

# ---------- 5. Hyper-parameter tuning with 5-fold CV on the training set ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

svm_grid = {"clf__kernel": ["linear", "rbf", "poly"],
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1]}
knn_grid = {"clf__n_neighbors": list(range(1, 26, 2)),
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"]}

svm_search = GridSearchCV(svm_pipe, svm_grid, cv=cv, scoring="f1", n_jobs=-1).fit(X_train, y_train)
knn_search = GridSearchCV(knn_pipe, knn_grid, cv=cv, scoring="f1", n_jobs=-1).fit(X_train, y_train)

print("\nBest SVM params :", svm_search.best_params_, "| CV F1 = %.4f" % svm_search.best_score_)
print("Best KNN params :", knn_search.best_params_, "| CV F1 = %.4f" % knn_search.best_score_)

models = {"SVM": svm_search.best_estimator_, "KNN": knn_search.best_estimator_}

# ---------- 6. Evaluation on the unseen test set ----------
rows, preds, probas = [], {}, {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    preds[name], probas[name] = y_pred, y_prob
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob[:, 1]),
        "Log Loss": log_loss(y_test, y_prob),
        "CV Acc (mean)": cv_acc.mean(),
        "CV Acc (std)": cv_acc.std(),
    })

results = pd.DataFrame(rows).set_index("Model").round(4)
print("\n=== Performance on test set ===")
display(results)
results.to_csv("results_table.csv")

for name in models:
    print(f"\n--- {name} : classification report ---")
    print(classification_report(y_test, preds[name], target_names=class_names, digits=4))

# ---------- 7. Confusion matrices ----------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, preds[name])
    print(f"{name} confusion matrix:\n", cm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=200); plt.show()

# ---------- 8. ROC curves ----------
plt.figure(figsize=(5, 4))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probas[name][:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results.loc[name, 'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("fig_roc.png", dpi=200); plt.show()

# ---------- 9. Effect of k on KNN (train vs CV accuracy) ----------
ks = range(1, 26, 2)
cv_scores, tr_scores = [], []
for k in ks:
    p = Pipeline([("scaler", StandardScaler()),
                  ("clf", KNeighborsClassifier(n_neighbors=k))])
    cv_scores.append(cross_val_score(p, X_train, y_train, cv=cv).mean())
    tr_scores.append(p.fit(X_train, y_train).score(X_train, y_train))
plt.figure(figsize=(5.5, 3.8))
plt.plot(ks, tr_scores, "o-", label="Training accuracy")
plt.plot(ks, cv_scores, "s-", label="5-fold CV accuracy")
plt.xlabel("k (number of neighbours)"); plt.ylabel("Accuracy")
plt.title("KNN: effect of k"); plt.legend(); plt.tight_layout()
plt.savefig("fig_knn_k.png", dpi=200); plt.show()

# ---------- 10. Effect of kernel on SVM (CV accuracy) ----------
kern_scores = {}
for kern in ["linear", "rbf", "poly", "sigmoid"]:
    p = Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(kernel=kern, C=1))])
    kern_scores[kern] = cross_val_score(p, X_train, y_train, cv=cv).mean()
plt.figure(figsize=(5, 3.6))
sns.barplot(x=list(kern_scores), y=list(kern_scores.values()), palette="viridis")
plt.ylim(0.8, 1.0); plt.ylabel("5-fold CV accuracy"); plt.title("SVM: effect of kernel (C=1)")
plt.tight_layout(); plt.savefig("fig_svm_kernel.png", dpi=200); plt.show()
print("\nSVM kernel comparison:", {k: round(v, 4) for k, v in kern_scores.items()})

# ---------- 11. Misclassified cases (important for a medical-type problem) ----------
for name in models:
    cm = confusion_matrix(y_test, preds[name])
    print(f"{name}: malignant predicted as benign (dangerous) = {cm[0, 1]} | "
          f"benign predicted as malignant (false alarm) = {cm[1, 0]}")
