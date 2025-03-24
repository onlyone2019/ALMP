import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 加载测试数据
X_test = pd.read_excel('../dataset/test_set.xlsx', header=None).iloc[:, :9]
y_test = pd.read_excel('../dataset/test_set.xlsx', header=None).iloc[:, 9]

# 加载模型
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')  # 加载保存的模型

# 转换为DMatrix格式（可选）
dtest = xgb.DMatrix(X_test.values)

# 预测与评估
y_pred = model.predict(X_test.values)
y_proba = model.predict_proba(X_test.values)[:,1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

'''
Accuracy: 0.9820
              precision    recall  f1-score   support
           0       0.99      0.98      0.98       721
           1       0.98      0.99      0.98       721

    accuracy                           0.98      1442
   macro avg       0.98      0.98      0.98      1442
weighted avg       0.98      0.98      0.98      1442

AUC Score: 0.9947
'''