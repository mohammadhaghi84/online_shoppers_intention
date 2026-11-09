import joblib
import pandas as pd
from typing import Dict, Any, Tuple

MODEL_PATH = "shopping_intention_model.joblib"

try:
    _model = joblib.load(MODEL_PATH)
except Exception as e:
    _model = None

def predict_intention(data: Dict[str, Any]) -> Tuple[bool, float]:
    if _model is None:
        raise RuntimeError("مدل بارگذاری نشده است.")

    df_input = pd.DataFrame([data])

    numeric_cols = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'OperatingSystems', 'Browser', 'Region', 'TrafficType'
    ]
    for col in numeric_cols:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col])

    if 'Weekend' in df_input.columns:
        df_input['Weekend'] = df_input['Weekend'].astype(bool)

    prediction = bool(_model.predict(df_input)[0])
    probability = float(_model.predict_proba(df_input)[0][1])

    return prediction, probability
