#!/usr/bin/env python3
"""
DNN Training Script for California Housing Price Prediction
Downloads data from scikit-learn datasets and trains a neural network
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
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

def test_model_performance(model, X_test, y_test, scaler, feature_names):
    """Comprehensive model performance testing"""
    print("\n🧪 Running comprehensive model performance tests...")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Convert to dollar amounts for better interpretation
    mae_dollars = mae * 100000
    rmse_dollars = rmse * 100000
    
    print(f"📊 Performance Metrics:")
    print(f"   Mean Absolute Error (MAE): ${mae_dollars:.2f}")
    print(f"   Root Mean Square Error (RMSE): ${rmse_dollars:.2f}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Mean target value: ${np.mean(y_test) * 100000:.2f}")
    
    # Performance interpretation
    print(f"\n📈 Performance Analysis:")
    if r2 > 0.8:
        print("   ✅ Excellent model performance (R² > 0.8)")
    elif r2 > 0.6:
        print("   ✅ Good model performance (R² > 0.6)")
    elif r2 > 0.4:
        print("   ⚠️  Moderate model performance (R² > 0.4)")
    else:
        print("   ❌ Poor model performance (R² ≤ 0.4)")
    
    # Error distribution analysis
    residuals = y_test - y_pred
    print(f"   Mean residual: ${np.mean(residuals) * 100000:.2f}")
    print(f"   Std residual: ${np.std(residuals) * 100000:.2f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mae_dollars': mae_dollars,
        'rmse_dollars': rmse_dollars,
        'residuals': residuals,
        'predictions': y_pred
    }

def test_prediction_ranges(model, X_test, y_test, scaler):
    """Test model performance across different price ranges"""
    print("\n🎯 Testing performance across price ranges...")
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Define price ranges (in hundreds of thousands)
    ranges = [
        (0, 1, "Low ($0-$100k)"),
        (1, 2, "Medium-Low ($100k-$200k)"),
        (2, 3, "Medium ($200k-$300k)"),
        (3, 4, "Medium-High ($300k-$400k)"),
        (4, float('inf'), "High ($400k+)")
    ]
    
    for min_price, max_price, label in ranges:
        mask = (y_test >= min_price) & (y_test < max_price)
        if np.sum(mask) > 0:
            range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            range_r2 = r2_score(y_test[mask], y_pred[mask])
            count = np.sum(mask)
            
            print(f"   {label}: MAE=${range_mae*100000:.0f}, R²={range_r2:.3f}, n={count}")

def test_cross_validation(X_train, y_train, input_shape, cv_folds=5):
    """Perform cross-validation to test model stability"""
    print(f"\n🔄 Running {cv_folds}-fold cross-validation...")
    
    cv_scores = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"   Training fold {fold + 1}/{cv_folds}...")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Create and train model for this fold
        fold_model = create_dnn_model(input_shape)
        
        # Simplified training for CV (fewer epochs)
        fold_model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=20,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        y_pred = fold_model.predict(X_fold_val, verbose=0).flatten()
        r2 = r2_score(y_fold_val, y_pred)
        cv_scores.append(r2)
    
    print(f"   Cross-validation R² scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"   Mean CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return cv_scores

def test_feature_importance(X_train, y_train, X_test, y_test, feature_names):
    """Test feature importance using a baseline model"""
    print("\n🔍 Analyzing feature importance (using Random Forest baseline)...")
    
    # Train a Random Forest for feature importance analysis
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Get feature importance
    importance = rf_model.feature_importances_
    
    # Sort features by importance
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("   Top features by importance:")
    for i, (feature, imp) in enumerate(feature_importance[:5]):
        print(f"   {i+1}. {feature}: {imp:.4f}")
    
    return feature_importance

def test_prediction_stability(model, X_test, num_runs=5):
    """Test prediction stability across multiple runs"""
    print(f"\n🎲 Testing prediction stability over {num_runs} runs...")
    
    predictions = []
    for run in range(num_runs):
        pred = model.predict(X_test, verbose=0).flatten()
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate stability metrics
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    
    print(f"   Mean prediction std: ${np.mean(std_predictions) * 100000:.2f}")
    print(f"   Max prediction std: ${np.max(std_predictions) * 100000:.2f}")
    
    if np.mean(std_predictions) < 0.001:  # Very low variance
        print("   ✅ Predictions are highly stable")
    elif np.mean(std_predictions) < 0.01:
        print("   ✅ Predictions are reasonably stable")
    else:
        print("   ⚠️  Predictions show some instability")

def test_edge_cases(model, scaler, feature_names):
    """Test model behavior on edge cases"""
    print("\n⚠️  Testing edge cases and boundary conditions...")
    
    # Create some edge case scenarios
    edge_cases = {
        'minimum_values': [0.5, 1, 1, 0.1, 1, 0.5, 32.5, -124.3],
        'maximum_values': [15, 50, 20, 5, 10000, 20, 42, -114],
        'median_values': [3.87, 29, 5.4, 1.1, 1425, 3, 34.2, -118.5]
    }
    
    for case_name, values in edge_cases.items():
        try:
            input_array = np.array(values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled, verbose=0)
            predicted_price = prediction[0][0] * 100000
            
            print(f"   {case_name}: ${predicted_price:.0f}")
            
            # Sanity checks
            if predicted_price < 0:
                print(f"     ❌ Warning: Negative prediction!")
            elif predicted_price > 10000000:  # $10M
                print(f"     ⚠️  Warning: Unusually high prediction!")
            
        except Exception as e:
            print(f"   {case_name}: Error - {str(e)}")

def visualize_results(y_test, y_pred, residuals, save_plots=False):
    """Create visualization plots for model evaluation"""
    print("\n📈 Creating evaluation visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test * 100000, y_pred * 100000, alpha=0.6)
    axes[0, 0].plot([y_test.min() * 100000, y_test.max() * 100000], 
                   [y_test.min() * 100000, y_test.max() * 100000], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price ($)')
    axes[0, 0].set_ylabel('Predicted Price ($)')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    
    # 2. Residuals plot
    axes[0, 1].scatter(y_pred * 100000, residuals * 100000, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residuals Plot')
    
    # 3. Residuals histogram
    axes[1, 0].hist(residuals * 100000, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    
    # 4. Q-Q plot for residuals normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')
    
    plt.tight_layout()
    
    if save_plots:
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("   Plots saved to plots/model_evaluation.png")
    
    plt.show()

def run_comprehensive_tests(model, X_train, X_test, y_train, y_test, scaler, feature_names):
    """Run all test functions comprehensively"""
    print("\n🚀 COMPREHENSIVE MODEL TESTING SUITE")
    print("="*70)
    
    # 1. Basic performance metrics
    perf_metrics = test_model_performance(model, X_test, y_test, scaler, feature_names)
    
    # 2. Performance across price ranges
    test_prediction_ranges(model, X_test, y_test, scaler)
    
    # 3. Cross-validation (on a subset for speed)
    if len(X_train) > 1000:
        subset_size = 1000
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        cv_scores = test_cross_validation(
            X_train[indices], y_train[indices], X_train.shape[1]
        )
    else:
        cv_scores = test_cross_validation(X_train, y_train, X_train.shape[1])
    
    # 4. Feature importance analysis
    feature_importance = test_feature_importance(X_train, y_train, X_test, y_test, feature_names)
    
    # 5. Prediction stability
    test_prediction_stability(model, X_test)
    
    # 6. Edge cases
    test_edge_cases(model, scaler, feature_names)
    
    # 7. Visualization
    try:
        visualize_results(y_test, perf_metrics['predictions'], perf_metrics['residuals'])
    except ImportError:
        print("   ⚠️  Matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"   ⚠️  Visualization error: {str(e)}")
    
    # Summary report
    print("\n📋 TESTING SUMMARY")
    print("="*30)
    print(f"✅ Model R² Score: {perf_metrics['r2']:.4f}")
    print(f"✅ Test MAE: ${perf_metrics['mae_dollars']:.2f}")
    print(f"✅ Test RMSE: ${perf_metrics['rmse_dollars']:.2f}")
    print(f"✅ Cross-validation mean R²: {np.mean(cv_scores):.4f}")
    print(f"✅ Most important feature: {feature_importance[0][0]}")
    
    return {
        'performance': perf_metrics,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }

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
    
    # Run comprehensive tests
    run_comprehensive_tests(model, X_train, X_test, y_train, y_test, scaler, feature_names)
    
    print("\n✨ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()