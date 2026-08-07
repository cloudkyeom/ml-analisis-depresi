import joblib
import os

def export_models(rf_model, lr_model, scaler, encoders, feature_names):
    print("Mengekspor model...")

    os.makedirs('models', exist_ok=True)

    joblib.dump(rf_model, 'models/model_rf.pkl')
    joblib.dump(lr_model, 'models/model_lr.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')

    print("Model berhasil diekspor ke folder /models/")