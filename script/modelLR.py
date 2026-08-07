import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from starter_analisis_depresi import X_train, X_test, y_train, y_test, X, y, df, df_encoded

def train_logistic_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # Confusion Matrix - Logistic Regression
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


    # training-test split dengan beberapa random_state untuk melihat variasi akurasi
    for rs in [0, 1, 7, 42, 99]:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)
        scaler_test = StandardScaler()
        X_tr_scaled = scaler_test.fit_transform(X_tr)
        X_te_scaled = scaler_test.transform(X_te)
        
        lr_test = LogisticRegression(max_iter=1000)
        lr_test.fit(X_tr_scaled, y_tr)
        acc = accuracy_score(y_te, lr_test.predict(X_te_scaled))

    print("\n--- HASIL MODELLING LOGISTIC REGRESSION ---")
    print("Accuracy :", accuracy_score(y_test, y_pred_lr))
    print("Precision:", precision_score(y_test, y_pred_lr))
    print("Recall   :", recall_score(y_test, y_pred_lr))
    print("F1-Score :", f1_score(y_test, y_pred_lr))
    print(f"random_state={rs} -> Accuracy: {acc:.4f}")

    print("\nClassification Report - Logistic Regression:")
    print(classification_report(y_test, y_pred_lr))

    print("\n--- CROSS-VALIDATION LOGISTIC REGRESSION ---")
    lr_cv = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(lr_cv, StandardScaler().fit_transform(X), y, cv=5)

    print(df_encoded.corr()['Depression'].sort_values(ascending=False))
    print("Duplikat:", df.duplicated().sum())
    print("Cross-validation scores:", cv_scores)
    print("Rata-rata akurasi:", cv_scores.mean())
    print("Standar deviasi:", cv_scores.std())

    return lr_model, scaler