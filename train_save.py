import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

BASE_PATH = "./UCI-HAR Dataset/"

def load_data(split):
    X = pd.read_csv(f"{BASE_PATH}{split}/X_{split}.txt", delim_whitespace=True, header=None)
    y = pd.read_csv(f"{BASE_PATH}{split}/y_{split}.txt", header=None).squeeze()
    return X, y

print("Loading data...")
X_train, y_train = load_data("train")
X_test, y_test = load_data("test")
X_cal, X_eval, y_cal, y_eval = train_test_split(X_test, y_test, test_size=0.70, random_state=42, stratify=y_test)

print("Training LR...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, 'lr_model.joblib')

print("Training XGBoost...")
le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train)
y_cal_xgb = le.transform(y_cal)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_train, y_train_xgb)

xgb_isotonic = CalibratedClassifierCV(xgb_model, cv='prefit', method='isotonic')
xgb_isotonic.fit(X_cal, y_cal_xgb)

joblib.dump(xgb_isotonic, 'xgb_isotonic.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("Models saved successfully!")
