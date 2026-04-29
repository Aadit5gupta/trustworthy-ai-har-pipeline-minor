import joblib
from src.data import get_splits
from src.models import train_all_models
from src.logger import logger

def run():
    logger.info("Starting training pipeline...")
    X_train, y_train, X_cal, y_cal, X_eval, y_eval = get_splits()
    
    # Train all models and save to artifacts/
    lr_model, xgb_isotonic, nn_wrapper, le = train_all_models(X_train, y_train, X_cal, y_cal)
    
    # Save the background dataset for SHAP to use later
    # We save a small subset of training data (e.g., 500 samples) as background
    X_bg = X_train.sample(n=min(500, len(X_train)), random_state=42)
    joblib.dump(X_bg, 'artifacts/X_bg.joblib')
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    run()
