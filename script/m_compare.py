import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.contingency_tables import mcnemar
import joblib

def compare_models(X_train, X_test, y_train, y_test):
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Train Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # Evaluate models
    print("\n--- EVALUASI MODEL ---")
    print("Random Forest:")
    print("Accuracy :", accuracy_score(y_test, y_pred_rf))
    print("\nLogistic Regression:")
    print("Accuracy :", accuracy_score(y_test, y_pred_lr))
# Load data
X_train, X_test, y_train, y_test, encoders = joblib.load('data_split.pkl')

# --- Train kedua model (sudah ada di file sebelumnya, tapi kita ulang) ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# --- Uji McNemar (perbandingan signifikansi) ---

y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# Buat tabel kontingensi: [RF benar/LR benar, RF benar/LR salah, RF salah/LR benar, RF salah/LR salah]
correct_rf = (y_pred_rf == y_test)
correct_lr = (y_pred_lr == y_test)

table = np.array([
    [np.sum(correct_rf & correct_lr), np.sum(correct_rf & ~correct_lr)],
    [np.sum(~correct_rf & correct_lr), np.sum(~correct_rf & ~correct_lr)]
])

print("\n--- UJI MCNEMAR (RF vs LR) ---")
print("Tabel kontingensi:")
print(table)

try:
    result = mcnemar(table, exact=True)
    print(f"P-value: {result.pvalue}")
    if result.pvalue < 0.05:
        print("Perbedaan signifikan secara statistik")
    else:
        print("Perbedaan tidak signifikan secara statistik")
except:
    print("Uji McNemar tidak bisa dijalankan (data terlalu kecil/terbatas)")

# --- Confidence Interval Akurasi (Bootstrap) ---
def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    accuracies = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        acc = accuracy_score(y_true[idx], y_pred[idx])
        accuracies.append(acc)
    lower = np.percentile(accuracies, (1-ci)/2 * 100)
    upper = np.percentile(accuracies, (1+ci)/2 * 100)
    return lower, upper

print("\n--- CONFIDENCE INTERVAL AKURASI ---")
ci_rf = bootstrap_ci(y_test.values, y_pred_rf)
ci_lr = bootstrap_ci(y_test.values, y_pred_lr)

print(f"Random Forest       : {ci_rf[0]:.3f} - {ci_rf[1]:.3f}")
print(f"Logistic Regression : {ci_lr[0]:.3f} - {ci_lr[1]:.3f}")