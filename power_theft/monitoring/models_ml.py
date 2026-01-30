# models_ml.py
from sklearn.ensemble import RandomForestClassifier

def random_forest_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
