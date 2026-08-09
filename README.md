# Analisis & Prediksi Depresi Mahasiswa

Script Python untuk eksplorasi data, pemodelan machine learning, dan interpretasi faktor risiko depresi pada mahasiswa, menggunakan dataset `Depression_Student_Dataset.csv`.

## Fitur

- **EDA (Exploratory Data Analysis)**
  - Distribusi target Depression (Yes/No)
  - Heatmap korelasi antar fitur numerik
  - Boxplot Academic Pressure vs Depression, lengkap dengan statistik (Min, Q1, Median, Q3, Max)
- **Preprocessing**
  - Encoding ordinal manual untuk `Sleep Duration` dan `Dietary Habits` (urutan level dijaga)
  - Encoding biner untuk kolom Yes/No dan Gender
- **Modeling**
  - Random Forest Classifier
  - Logistic Regression (dengan standardisasi fitur)
  - Evaluasi: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Interpretasi Model**
  - Feature Importance dari Random Forest
  - SHAP Summary Plot & Bar Plot
  - SHAP Force Plot untuk interpretasi satu sampel individu

## Struktur Dataset

| Kolom | Tipe | Keterangan |
|---|---|---|
| Gender | Biner | Male / Female |
| Age | Numerik | Usia mahasiswa |
| Academic Pressure | Numerik | Skala tekanan akademik |
| Study Satisfaction | Numerik | Skala kepuasan belajar |
| Sleep Duration | Ordinal | Less than 5h / 5-6h / 7-8h / More than 8h |
| Dietary Habits | Ordinal | Unhealthy / Moderate / Healthy |
| Have you ever had suicidal thoughts? | Biner | Yes / No |
| Study Hours | Numerik | Jam belajar per hari |
| Financial Stress | Numerik | Skala tekanan finansial |
| Family History of Mental Illness | Biner | Yes / No |
| Depression (target) | Biner | Yes / No |

## Instalasi

Pastikan Python 3.9+ sudah terpasang, lalu install dependency:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

## Cara Menjalankan

```bash
cd script
python main.py
```

Saat dijalankan, script akan meminta input:

1. **Path dataset (.csv)** — masukkan path menuju `Depression_Student_Dataset.csv`
2. **Index sampel** (di akhir) — untuk menampilkan SHAP force plot dari satu mahasiswa tertentu (index sesuai baris di `X_test`)
Setiap tahap analisis akan menampilkan plot secara berurutan — tutup jendela plot untuk melanjutkan ke tahap berikutnya.

## Output yang Dihasilkan

- Grafik distribusi, korelasi, dan boxplot (EDA)
- Confusion matrix Random Forest dan logistic regression
- Ranking feature importance
- SHAP summary plot (dampak tiap fitur terhadap prediksi)
- SHAP force plot (penjelasan prediksi untuk 1 individu)
- Metrik evaluasi (accuracy, precision, recall, F1) untuk kedua model di terminal

## Catatan & Batasan

- Baris dengan missing value akan dihapus (`dropna()`)
- cek jumlah `isnull().sum()` di output untuk mengetahui seberapa banyak data yang terbuang.
- Encoding ordinal (`Sleep Duration`, `Dietary Habits`) mengasumsikan urutan level sesuai definisi di atas; sesuaikan mapping di kode jika kategori pada dataset berbeda.
- Model dievaluasi dengan satu kali `train_test_split` (belum menggunakan cross-validation), sehingga hasil akurasi dapat sedikit bervariasi tergantung `random_state`.
- SHAP `TreeExplainer` hanya kompatibel dengan model tree-based (Random Forest), bukan Logistic Regression.
