#!/usr/bin/env python3
"""
XGBoost Training Script for California Housing Price Prediction
Downloads data from scikit-learn datasets and trains an XGBoost model
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

def download_and_prepare_data():
    """Download California housing dataset and prepare for training"""
    print("📥 Downloading California housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    # MedHouseVal is in units of $100,000, so filter for values less than 4.9
    df = df[df['MedHouseVal'] < 5.0 ]  # Remove outliers above $500,000

    X = df.drop(columns='MedHouseVal') 
    y = df['MedHouseVal']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features (still beneficial for XGBoost)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"   Features: {housing.feature_names}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, housing.feature_names

def create_xgb_model():
    """Create an XGBoost Regressor model with optimized hyperparameters"""
    model = xgb.XGBRegressor(
        n_estimators=2000,  # Increased from 1000
        max_depth=8,        # Increased from 6
        learning_rate=0.05, # Decreased from 0.1 for more gradual learning
        subsample=0.9,      # Increased from 0.8
        colsample_bytree=0.9, # Increased from 0.8
        min_child_weight=3,   # Add this parameter
        gamma=0.1,           # Add this parameter for regularization
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.01,      # Reduced from 0.1
        reg_lambda=0.1,      # Reduced from 1.0
        early_stopping_rounds=100, # Increased from 50
        eval_metric='mae'
    )
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the XGBoost model"""
    print("\n🚀 Starting model training...")
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train the model with early stopping
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    
    print(f"\n✅ Training complete!")
    print(f"   Test MAE: ${test_mae*100000:.2f}")  # Convert to dollars
    print(f"   Test RMSE: ${test_rmse*100000:.2f}")
    print(f"   Number of trees used: {model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators}")
    
    return model, test_mae

def save_artifacts(model, scaler, feature_names, metrics):
    """Save model and preprocessing artifacts with version compatibility"""
    os.makedirs('models', exist_ok=True)
    
    print(f"\n💾 Saving model artifacts (XGBoost version: {xgb.__version__})...")
    
    # Save the XGBoost model in JSON format (newer versions)
    try:
        model_path = 'models/housing_model.json'
        model.save_model(model_path)
        print(f"   ✅ Model saved to {model_path} (JSON format)")
    except Exception as e:
        print(f"   ⚠️ JSON save failed: {str(e)}")
    
    # Also save as pickle for compatibility with older XGBoost versions
    try:
        with open('models/housing_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ Model saved to models/housing_model.pkl (pickle format)")
    except Exception as e:
        print(f"   ⚠️ Pickle save failed: {str(e)}")
    
    # Save the scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler saved to models/scaler.pkl")
    
    # Save metadata with performance metrics
    metadata = {
        'feature_names': feature_names,
        'model_version': '1.0.0',
        'training_date': datetime.now().isoformat(),
        'performance_metrics': {
            'mae': float(metrics),
            'test_mae': float(metrics),  # Keep for backward compatibility
            'r2_score': 0.0  # Will be calculated in test_model.py
        },
        'framework': 'xgboost',
        'framework_version': xgb.__version__,
        'model_type': 'XGBRegressor',
        'hyperparameters': {
            'n_estimators': int(model.n_estimators),
            'max_depth': int(model.max_depth),
            'learning_rate': float(model.learning_rate)
        },
        'training_notes': 'Model saved in both JSON and pickle formats for compatibility'
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved to models/metadata.json")
    
    # Verify the saved files
    saved_files = ['models/housing_model.json', 'models/housing_model.pkl', 'models/scaler.pkl', 'models/metadata.json']
    for file_path in saved_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   📁 {file_path}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {file_path}: Not found!")
    
    return metadata

def main():
    """Main training pipeline"""
    print("🏠 California Housing Price Prediction - XGBoost Training")
    print("="*55)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Download and prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = download_and_prepare_data()
    
    # Create and train model
    model = create_xgb_model()
    print(f"\n📊 XGBoost Model Parameters:")
    print(f"   n_estimators: {model.n_estimators}")
    print(f"   max_depth: {model.max_depth}")
    print(f"   learning_rate: {model.learning_rate}")
    print(f"   subsample: {model.subsample}")
    print(f"   colsample_bytree: {model.colsample_bytree}")
    
    model, test_mae = train_model(model, X_train, y_train, X_test, y_test)
    
    # Save everything
    save_artifacts(model, scaler, feature_names, test_mae)
    
    print("\n✨ Training pipeline completed successfully!")
    print("   Run 'python test_model.py' to perform comprehensive testing.")

if __name__ == "__main__":
    main()