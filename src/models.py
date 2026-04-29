import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import numpy as np
import yaml
from src.logger import logger

class HAR_MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class PyTorchModelWrapper:
    def __init__(self, model, le):
        self.model = model
        self.le = le
        self.model.eval()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.le.inverse_transform(np.argmax(proba, axis=1))

def train_all_models(X_train, y_train, X_cal, y_cal):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=config['models']['lr_max_iter'], random_state=42, multi_class='multinomial', solver='lbfgs')
    lr_model.fit(X_train, y_train)
    
    logger.info("Training XGBoost...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_cal_encoded = le.transform(y_cal)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=config['models']['xgb_estimators'], max_depth=config['models']['xgb_depth'], learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_encoded)
    
    logger.info("Calibrating XGBoost...")
    xgb_isotonic = CalibratedClassifierCV(xgb_model, cv='prefit', method='isotonic')
    xgb_isotonic.fit(X_cal, y_cal_encoded)
    
    logger.info("Training PyTorch NN...")
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    nn_model = HAR_MLP(input_dim, num_classes)
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=config['models']['nn_batch_size'], shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=config['models']['nn_lr'])
    
    nn_model.train()
    for epoch in range(config['models']['nn_epochs']):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    nn_wrapper = PyTorchModelWrapper(nn_model, le)
    
    # Save models
    logger.info("Saving models...")
    joblib.dump(lr_model, 'artifacts/lr_model.joblib')
    joblib.dump(xgb_isotonic, 'artifacts/xgb_isotonic.joblib')
    torch.save(nn_model.state_dict(), 'artifacts/nn_model.pth')
    joblib.dump(le, 'artifacts/label_encoder.joblib')
    
    return lr_model, xgb_isotonic, nn_wrapper, le

def load_all_models():
    lr_model = joblib.load('artifacts/lr_model.joblib')
    xgb_isotonic = joblib.load('artifacts/xgb_isotonic.joblib')
    le = joblib.load('artifacts/label_encoder.joblib')
    
    # Load NN
    input_dim = 561 # HAR fixed
    num_classes = 6
    nn_model = HAR_MLP(input_dim, num_classes)
    nn_model.load_state_dict(torch.load('artifacts/nn_model.pth', weights_only=True))
    nn_wrapper = PyTorchModelWrapper(nn_model, le)
    
    return lr_model, xgb_isotonic, nn_wrapper, le
