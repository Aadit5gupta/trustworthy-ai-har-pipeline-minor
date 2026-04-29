import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logger

def load_data(base_path="./UCI-HAR Dataset/", split="train"):
    X = pd.read_csv(f"{base_path}{split}/X_{split}.txt", sep='\s+', header=None)
    y = pd.read_csv(f"{base_path}{split}/y_{split}.txt", header=None).squeeze()
    return X, y

def get_splits(base_path="./UCI-HAR Dataset/", test_size=0.70, random_state=42):
    logger.info("Loading dataset...")
    X_train, y_train = load_data(base_path, "train")
    X_test, y_test = load_data(base_path, "test")
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_test, y_test, test_size=test_size, random_state=random_state, stratify=y_test
    )
    logger.info(f"Data splits ready. Train: {len(X_train)}, Cal: {len(X_cal)}, Eval: {len(X_eval)}")
    return X_train, y_train, X_cal, y_cal, X_eval, y_eval
