import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib as jbl
from starter_analisis_depresi import X_train, X_test, y_train, y_test

def preprocess_data(df):
    dataset = "Depression_Student_Dataset.csv"
    df = pd.read_csv(dataset)

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

    jbl.dump((X_train, X_test, y_train, y_test, encoders), 'data_split.pkl')

    return X_train, X_test, y_train, y_test, encoders, X.columns.tolist()