import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import shap

dataset = input("Masukkan path dataset (.csv): ")
df = pd.read_csv(dataset)

# Cek struktur data
print("Shape data:", df.shape)
print("\nKolom yang tersedia:")
print(df.columns.tolist())
print("\ndata:")
print(df.head())

# 3. CEK MISSING VALUES & TIPE DATA
print("\nInfo dataset:")
print(df.info())

print("\nJumlah missing value per kolom:")
print(df.isnull().sum())
df = df.dropna()  # hapus baris dengan missing value

# 4. EXPLORATORY DATA ANALYSIS (EDA)

# Distribusi target (Depression: Yes/No)
# FIX #1: tambah hue + legend=False biar palette per kategori kepakai (seaborn versi baru)
plt.figure(figsize=(5,4))
bars = sns.countplot(data=df, x='Depression', hue='Depression', palette=['skyblue', 'pink'], legend=False)
plt.title("Distribusi Depresi pada Mahasiswa")
for container in bars.containers:
    bars.bar_label(container)
plt.show()

# Cek korelasi antar fitur numerik
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi Antar Fitur Numerik")
plt.show()

# Academic Pressure vs Depression
plt.figure(figsize=(6,4))
bp = sns.boxplot(data=df, x='Depression', y='Academic Pressure', hue='Depression',
                  palette=['skyblue', 'pink'], legend=False)
colors = ['black', 'black']
categories = df['Depression'].unique()

for i, category in enumerate(categories):
    data = df[df['Depression'] == category]['Academic Pressure']

    q1 = data.quantile(0.25)
    median = data.median()
    q3 = data.quantile(0.75)
    minimum = data.min()
    maximum = data.max()

    stats = {
        minimum: 'Min',
        q1: 'Q1',
        median: 'Median',
        q3: 'Q3',
        maximum: 'Max'
    }

    for value, label in stats.items():
        bp.text(i + 0.35, value, f'{label}: {value:.2f}',
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=8,
                color=colors[i])

plt.xlim(-0.5, len(categories) - 0.2)
plt.title('Academic Pressure vs Depression')
plt.show()

# 5. PREPROCESSING (ENCODING DATA KATEGORIKAL)
# FIX #3 & #4: pisahkan kolom ordinal (Sleep Duration, Dietary Habits) yang di-mapping manual
# sesuai urutan levelnya, dan kolom biner yang aman pakai LabelEncoder (satu encoder per kolom,
# disimpan di dict biar bisa di-inverse_transform lagi nanti).
df_encoded = df.copy()

# --- Kolom ORDINAL: urutan levelnya penting, di-mapping manual ---
sleep_order = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3
}
df_encoded['Sleep Duration'] = df_encoded['Sleep Duration'].map(sleep_order)

diet_order = {
    'Unhealthy': 0,
    'Moderate': 1,
    'Healthy': 2
}
df_encoded['Dietary Habits'] = df_encoded['Dietary Habits'].map(diet_order)

# --- Kolom BINER: aman pakai LabelEncoder (satu encoder per kolom) ---
binary_cols = [
    'Gender',
    'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness',
    'Depression'
]

encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le

print("\nMapping label 'Depression':",
      dict(zip(encoders['Depression'].classes_,
                encoders['Depression'].transform(encoders['Depression'].classes_))))

print("\nData setelah encoding:")
print(df_encoded.head())

# 6. SPLIT FITUR (X) & TARGET (y)
X = df_encoded.drop('Depression', axis=1)
y = df_encoded['Depression']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nJumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")

# 7. MODELING - RANDOM FOREST
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

# 8. MODELING - LOGISTIC REGRESSION
# FIX #2: hapus refit ulang pakai data mentah yang bikin hasil scaling ketimpa sia-sia.
# Sekarang konsisten pakai X_train_scaled / X_test_scaled dari awal sampai akhir.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("\n--- HASIL MODELLING LOGISTIC REGRESSION ---")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1-Score :", f1_score(y_test, y_pred_lr))

# 9. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n FEATURE IMPORTANCE (Faktor Risiko Paling Berpengaruh)")
print(feature_importance)

plt.figure(figsize=(8,6))
sns.barplot(data=feature_importance, x='Importance', y='Fitur', palette='viridis', hue=True, legend=False)
plt.title("Faktor Risiko Depresi Paling Berpengaruh")
plt.xlabel("Tingkat Kepentingan")
plt.ylabel("Fitur")
plt.tight_layout()
plt.show()

# top 3 fitur paling berpengaruh
top_3_features = feature_importance.head(3)['Fitur'].tolist()
print(f"\n>> 3 Faktor risiko paling berpengaruh terhadap depresi: {top_3_features}")

# 11. SHAP ANALYSIS (Explainable AI)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]  # ambil kelas "Depression = Yes"
elif len(np.array(shap_values).shape) == 3:
    shap_values_plot = shap_values[:, :, 1]
else:
    shap_values_plot = shap_values

# --- SUMMARY PLOT ---
# Titik merah = nilai fitur tinggi, biru = nilai fitur rendah
# Posisi kanan = mendorong prediksi ke arah "Depression: Yes"
plt.figure()
shap.summary_plot(shap_values_plot, X_test, feature_names=X.columns, show=False)
plt.title("SHAP Summary Plot - Pengaruh Tiap Fitur terhadap Risiko Depresi")
plt.tight_layout()
plt.show()

# --- BAR PLOT ranking fitur berdasarkan rata-rata dampak SHAP ---
plt.figure()
shap.summary_plot(shap_values_plot, X_test, feature_names=X.columns,
                   plot_type='bar', show=False)
plt.title("Ranking Faktor Risiko Berdasarkan SHAP Value")
plt.tight_layout()
plt.show()

# INTERPRETASI SATU MAHASISWA (opsional) Menjelaskan kenapa satu individu diprediksi "berisiko depresi"
sample_idx = int(input("input index: "))
plt.figure()
shap.force_plot(
    explainer.expected_value[1] 
    if isinstance(explainer.expected_value, np.ndarray) 
    else explainer.expected_value,
    shap_values_plot[sample_idx],
    X_test.iloc[sample_idx],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.show()
