import joblib
import os
from django.conf import settings

def get_recommmender_model():
    base_path = os.path.join(settings.BASE_DIR,'ML_models')

    model_path = os.path.join(base_path, 'lightfm_model.pkl')
    dataset_path = os.path.join(base_path, 'lightfm_dataset.pkl')
    item_features_path = os.path.join(base_path, 'lightfm_item_features.pkl')
    
    if not all(os.path.exists(p) for p in [model_path, dataset_path, item_features_path]):
        raise FileNotFoundError("One or more model files not found in 'models/' directory")

    rf_model = joblib.load(model_path)
    rf_dataset = joblib.load(dataset_path)
    rf_item_features = joblib.load(item_features_path)

    return rf_model, rf_dataset, rf_item_features
