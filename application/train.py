#!/usr/bin/env python3
"""
DNN Training Script for California Housing Price Prediction
Downloads data from scikit-learn datasets and trains a neural network
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def download_and_prepare_data():
    """Download California housing dataset and prepare for training"""
    print("📥 Downloading California housing dataset...")
    housing = fetch_california_housing()
    
    X = housing.data
    y = housing.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"   Features: {housing.feature_names}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, housing.feature_names

def create_dnn_model(input_shape):
    """Create a Deep Neural Network model"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(1)  # Output layer for regression
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the DNN model"""
    print("\n🚀 Starting model training...")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Training complete!")
    print(f"   Test MAE: ${test_mae*100000:.2f}")  # Convert to dollars
    print(f"   Test RMSE: ${np.sqrt(test_mse)*100000:.2f}")
    
    return history, test_mae

def save_artifacts(model, scaler, feature_names, metrics):
    """Save model and preprocessing artifacts"""
    os.makedirs('models', exist_ok=True)
    
    # Save the model in TensorFlow SavedModel format
    model_path = 'models/housing_model.keras'
    model.save(model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Save the scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved to models/scaler.pkl")
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'model_version': '1.0.0',
        'training_date': datetime.now().isoformat(),
        'test_mae': float(metrics),
        'framework': 'tensorflow',
        'framework_version': tf.__version__
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved to models/metadata.json")

def main():
    """Main training pipeline"""
    print("🏠 California Housing Price Prediction - DNN Training")
    print("="*50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Download and prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = download_and_prepare_data()
    
    # Create and train model
    model = create_dnn_model(X_train.shape[1])
    print("\n📊 Model Architecture:")
    model.summary()
    
    history, test_mae = train_model(model, X_train, y_train, X_test, y_test)
    
    # Save everything
    save_artifacts(model, scaler, feature_names, test_mae)
    
    print("\n✨ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()