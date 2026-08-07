import pandas as pd
from eda import EDA
from preprocessing import preprocess_data
from modelRF import evaluate_random_forest
from modelLR import train_logistic_regression
from fImportance import feature_importance
from fImportance import run_shap_analysis
from export import export_models

# 1. Load data
df = pd.read_csv("Depression_Student_Dataset.csv")

# 1. Exploratory Data Analysis
EDA(df) 

# 2. Preprocessing
X_train, X_test, y_train, y_test, encoders, feature_names = preprocess_data(df)

# 3. Train Random Forest
rf_model = evaluate_random_forest(X_train, X_test, y_train, y_test)

# 4. Train Logistic Regression
lr_model, scaler = train_logistic_regression(X_train, X_test, y_train, y_test)

# 5. Feature Importance
importance_df = feature_importance(X_train, X_test, y_train, y_test)

# 6. SHAP Analysis
run_shap_analysis(X_train, X_test, rf_model)

# 7. Export Models
export_models(rf_model, lr_model, scaler, encoders, feature_names)