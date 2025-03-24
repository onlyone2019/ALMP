import xgboost as xgb

import pandas as pd
import numpy as np


X_train = pd.read_excel('../dataset/train_set.xlsx', header=None).iloc[:, :9]
y_train = pd.read_excel('../dataset/train_set.xlsx', header=None).iloc[:, 9]

model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1)
model.fit(X_train, y_train)
model.save_model('xgb_model.json')