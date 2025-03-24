import xgboost as xgb
import numpy as np

manual_data = np.array([
    [0.5, 1.2, 0.3, 2.4, 0.8, 1.5, 1.0, 0.6, 1.1],
    [0.2, 0.7, 1.8, 0.5, 1.0, 0.3, 0.9, 1.2, 0.4]
])

# 加载模型
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')

# XGBoost需要二维输入
probabilities = model.predict_proba(manual_data)[:, 1]
predictions = model.predict(manual_data)

print("XGBoost Predictions:", predictions)
print("Probabilities:", probabilities)