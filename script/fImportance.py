import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
from starter_analisis_depresi import X_train, X_test, y_train, y_test

def feature_importance(X_train, X_test, y_train, y_test):
    # Load data
    X_train, X_test, y_train, y_test, encoders = joblib.load('data_split.pkl')

    # --- Random Forest model (pakai model yang udah dilatih) ---
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # --- Feature Importance dari Random Forest ---
    feature_importance = pd.DataFrame({
        'Fitur': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\n--- FEATURE IMPORTANCE (Random Forest) ---")
    print(feature_importance)

    plt.figure(figsize=(8,6))
    sns.barplot(data=feature_importance, x='Importance', y='Fitur', palette='viridis', hue=False, legend=False)
    plt.title("Faktor Risiko Depresi Paling Berpengaruh")
    plt.xlabel("Tingkat Kepentingan")
    plt.ylabel("Fitur")
    plt.tight_layout()
    plt.show()

    # return rf_model, feature_importance

def run_shap_analysis(X_train, X_test,rf_model):
    try:
        # 1. Buat explainer
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)

        # 2. Ambil SHAP values untuk kelas positif (Depression=Yes)
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]  # ambil kelas "1"
        elif len(np.array(shap_values).shape) == 3:
            shap_values_plot = shap_values[:, :, 1]
        else:
            shap_values_plot = shap_values

        # 3. Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_plot,
            X_test,
            feature_names=X_test.columns,  # ← pakai X_test.columns, bukan X_train
            show=False
        )
        plt.title("SHAP Summary Plot - Pengaruh Tiap Fitur terhadap Risiko Depresi")
        plt.tight_layout()
        plt.savefig('output/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ SHAP Summary Plot disimpan di output/shap_summary.png")

        # 4. Bar Plot (Ranking Fitur)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_plot,
            X_test,
            feature_names=X_test.columns,
            plot_type='bar',
            show=False
        )
        plt.title("Ranking Faktor Risiko Berdasarkan SHAP Value")
        plt.tight_layout()
        plt.savefig('output/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(" SHAP Bar Plot disimpan di output/shap_bar_plot.png")

        print("SHAP Analysis selesai!")
    except Exception as e:
        print("Terjadi kesalahan saat menjalankan SHAP Analysis:", str(e))