import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import shap

# Ganti nama file sesuai file yang kamu upload
df = pd.read_csv("Depression Student Dataset.csv")

# Cek struktur data
print("Shape data:", df.shape)
print("\nKolom yang tersedia:")
print(df.columns.tolist())
print("\nContoh data:")
print(df.head())

# 3. CEK MISSING VALUES & TIPE DATA
print("\nInfo dataset:")
print(df.info())

print("\nJumlah missing value per kolom:")
print(df.isnull().sum())
df = df.dropna()  # hapus baris dengan missing value

# 4. EXPLORATORY DATA ANALYSIS (EDA)

# Distribusi target (Depression: Yes/No)
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Depression')  # sesuaikan nama kolom target
plt.title("Distribusi Depresi pada Mahasiswa")
plt.show()

# Cek korelasi antar fitur numerik
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi Antar Fitur Numerik")
plt.show()

# Contoh: Academic Pressure vs Depression
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Depression', y='Academic Pressure')  # sesuaikan nama kolom
plt.title("Academic Pressure vs Depression")
plt.show()

# 5. PREPROCESSING (ENCODING DATA KATEGORIKAL)
df_encoded = df.copy()

# Encode semua kolom kategorikal (object type) jadi angka
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

print("\nData setelah encoding:")
print(df_encoded.head())

# 6. SPLIT FITUR (X) & TARGET (y)
# Ganti 'Depression' sesuai nama kolom target di dataset kamu
X = df_encoded.drop('Depression', axis=1)
y = df_encoded['Depression']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nJumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")

# 7. MODELING - RANDOM FOREST (mudah & powerful untuk pemula)
# Random Forest tetap pakai data asli (X_train, X_test) standart scaler
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)

print("\n=== HASIL RANDOM FOREST ===")
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

# 8. MODELING - LOGISTIC REGRESSION (pembanding, lebih simpel)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression pakai data yang sudah dinormalisasi
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\n=== HASIL LOGISTIC REGRESSION ===")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1-Score :", f1_score(y_test, y_pred_lr))

# 9. FEATURE IMPORTANCE (INI BAGIAN PALING PENTING UNTUK KTI!)
feature_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n=== FEATURE IMPORTANCE (Faktor Risiko Paling Berpengaruh) ===")
print(feature_importance)

plt.figure(figsize=(8,6))
sns.barplot(data=feature_importance, x='Importance', y='Fitur', palette='viridis')
plt.title("Faktor Risiko Depresi Paling Berpengaruh (Random Forest)")
plt.xlabel("Tingkat Kepentingan")
plt.ylabel("Fitur")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# 10. KESIMPULAN OTOMATIS (opsional, buat bantu nulis pembahasan)
# -----------------------------------------------------------
top_3_features = feature_importance.head(3)['Fitur'].tolist()
print(f"\n>> 3 Faktor risiko paling berpengaruh terhadap depresi: {top_3_features}")
print(">> Gunakan hasil ini sebagai dasar pembahasan di bagian 'Faktor Risiko Utama' KTI kamu.")

# =========================================================
# 11. SHAP ANALYSIS (Explainable AI)
# Ini level lebih dalam dari feature importance biasa.
# Feature importance cuma bilang "fitur mana yang penting",
# SHAP bisa jelasin "gimana caranya fitur itu mempengaruhi
# prediksi" - misal apakah nilai TINGGI di suatu fitur bikin
# risiko depresi naik atau turun. Ini yang bikin analisis
# kamu lebih 'berbobot' dibanding riset lain yang cuma
# stop di feature importance biasa.
# =========================================================

# Buat SHAP explainer khusus untuk model tree-based (Random Forest)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Untuk klasifikasi biner, shap_values bisa berupa list [class_0, class_1]
# atau array 3D tergantung versi SHAP. Kode ini handle keduanya.
if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]  # ambil kelas "Depression = Yes"
elif len(np.array(shap_values).shape) == 3:
    shap_values_plot = shap_values[:, :, 1]
else:
    shap_values_plot = shap_values

# --- SUMMARY PLOT: gambaran umum semua fitur ---
# Titik merah = nilai fitur tinggi, biru = nilai fitur rendah
# Posisi kanan = mendorong prediksi ke arah "Depression: Yes"
plt.figure()
shap.summary_plot(shap_values_plot, X_test, feature_names=X.columns, show=False)
plt.title("SHAP Summary Plot - Pengaruh Tiap Fitur terhadap Risiko Depresi")
plt.tight_layout()
plt.show()

# --- BAR PLOT: ranking fitur berdasarkan rata-rata dampak SHAP ---
plt.figure()
shap.summary_plot(shap_values_plot, X_test, feature_names=X.columns,
                   plot_type='bar', show=False)
plt.title("Ranking Faktor Risiko Berdasarkan SHAP Value")
plt.tight_layout()
plt.show()

# --- CONTOH INTERPRETASI SATU MAHASISWA (opsional, bagus buat ilustrasi di KTI) ---
# Menjelaskan kenapa satu individu diprediksi "berisiko depresi"
sample_idx = 0  # ganti index ini untuk lihat contoh lain
plt.figure()
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
    shap_values_plot[sample_idx],
    X_test.iloc[sample_idx],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.show() 