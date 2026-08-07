import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jbl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from starter_analisis_depresi import X_train, X_test, y_train, y_test

def evaluate_random_forest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)

    print("\n--- HASIL MODELLING RANDOM FOREST ---")
    print("Accuracy :", accuracy_score(y_test, y_pred_rf))
    print("Precision:", precision_score(y_test, y_pred_rf))
    print("Recall   :", recall_score(y_test, y_pred_rf))
    print("F1-Score :", f1_score(y_test, y_pred_rf))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)

    print("\n--- HASIL MODELLING RANDOM FOREST ---")
    print("Accuracy :", accuracy_score(y_test, y_pred_rf))
    print("Precision:", precision_score(y_test, y_pred_rf))
    print("Recall   :", recall_score(y_test, y_pred_rf))
    print("F1-Score :", f1_score(y_test, y_pred_rf))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return rf_model