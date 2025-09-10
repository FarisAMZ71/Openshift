#!/usr/bin/env python3
"""
Standalone Model Testing Script
Tests a trained model's performance with comprehensive evaluation
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras

def load_test_data():
    """Load test data for evaluation"""
    print("📥 Loading test data...")
    housing = fetch_california_housing()
    
    X = housing.data
    y = housing.target
    
    # Use same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_test, y_test, housing.feature_names

def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    print("🔄 Loading model artifacts...")
    
    try:
        # Load model
        model = keras.models.load_model('models/housing_model.keras')
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ All artifacts loaded successfully")
        return model, scaler, metadata
    
    except Exception as e:
        print(f"❌ Failed to load artifacts: {str(e)}")
        return None, None, None

def detailed_performance_analysis(model, X_test, y_test, scaler, feature_names):
    """Detailed performance analysis with business insights"""
    print("\n📊 DETAILED PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Scale test data and make predictions
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Core metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Business metrics
    mae_dollars = mae * 100000
    rmse_dollars = rmse * 100000
    mean_price = np.mean(y_test) * 100000
    median_price = np.median(y_test) * 100000
    
    print(f"🎯 Core Performance Metrics:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Mean Absolute Error: ${mae_dollars:,.2f}")
    print(f"   Root Mean Square Error: ${rmse_dollars:,.2f}")
    print(f"   Mean Absolute Percentage Error: {(mae/np.mean(y_test))*100:.2f}%")
    
    print(f"\n💰 Business Context:")
    print(f"   Mean actual price: ${mean_price:,.2f}")
    print(f"   Median actual price: ${median_price:,.2f}")
    print(f"   MAE as % of mean price: {(mae_dollars/mean_price)*100:.2f}%")
    
    # Error distribution
    residuals = y_test - y_pred
    abs_residuals = np.abs(residuals)
    
    print(f"\n📈 Error Distribution:")
    print(f"   25% of predictions within: ${np.percentile(abs_residuals, 25)*100000:,.2f}")
    print(f"   50% of predictions within: ${np.percentile(abs_residuals, 50)*100000:,.2f}")
    print(f"   75% of predictions within: ${np.percentile(abs_residuals, 75)*100000:,.2f}")
    print(f"   90% of predictions within: ${np.percentile(abs_residuals, 90)*100000:,.2f}")
    print(f"   95% of predictions within: ${np.percentile(abs_residuals, 95)*100000:,.2f}")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mae_dollars': mae_dollars,
        'rmse_dollars': rmse_dollars,
        'predictions': y_pred,
        'residuals': residuals
    }

def segment_analysis(y_test, y_pred, X_test, feature_names):
    """Analyze performance across different segments"""
    print("\n🎯 SEGMENT PERFORMANCE ANALYSIS")
    print("="*45)
    
    # Price range analysis
    price_ranges = [
        (0, 1, "Budget ($0-$100k)"),
        (1, 2, "Affordable ($100k-$200k)"),
        (2, 3, "Mid-range ($200k-$300k)"),
        (3, 4, "Premium ($300k-$400k)"),
        (4, 5, "Luxury ($400k-$500k)"),
        (5, float('inf'), "Ultra-luxury ($500k+)")
    ]
    
    print("📊 Performance by Price Range:")
    for min_price, max_price, label in price_ranges:
        mask = (y_test >= min_price) & (y_test < max_price)
        if np.sum(mask) > 10:  # Only analyze if enough samples
            segment_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            segment_r2 = r2_score(y_test[mask], y_pred[mask])
            count = np.sum(mask)
            
            print(f"   {label}:")
            print(f"     MAE: ${segment_mae*100000:,.0f}, R²: {segment_r2:.3f}, n={count}")
    
    # Geographic analysis (if we have lat/lon features)
    if 'Latitude' in feature_names and 'Longitude' in feature_names:
        print("\n🌍 Geographic Performance Analysis:")
        lat_idx = list(feature_names).index('Latitude')
        lon_idx = list(feature_names).index('Longitude')
        
        # Northern vs Southern California
        median_lat = np.median(X_test[:, lat_idx])
        north_mask = X_test[:, lat_idx] >= median_lat
        south_mask = X_test[:, lat_idx] < median_lat
        
        north_mae = mean_absolute_error(y_test[north_mask], y_pred[north_mask])
        south_mae = mean_absolute_error(y_test[south_mask], y_pred[south_mask])
        
        print(f"   Northern CA: MAE=${north_mae*100000:,.0f}, n={np.sum(north_mask)}")
        print(f"   Southern CA: MAE=${south_mae*100000:,.0f}, n={np.sum(south_mask)}")

def model_reliability_tests(model, X_test, scaler, num_runs=10):
    """Test model reliability and consistency"""
    print("\n🔬 MODEL RELIABILITY TESTS")
    print("="*35)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Multiple prediction runs
    predictions = []
    for i in range(num_runs):
        pred = model.predict(X_test_scaled, verbose=0).flatten()
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Consistency metrics
    mean_std = np.mean(np.std(predictions, axis=0))
    max_std = np.max(np.std(predictions, axis=0))
    
    print(f"🎲 Prediction Consistency:")
    print(f"   Mean standard deviation: ${mean_std * 100000:.2f}")
    print(f"   Maximum standard deviation: ${max_std * 100000:.2f}")
    
    if mean_std < 0.001:
        print("   ✅ Highly consistent predictions")
    elif mean_std < 0.01:
        print("   ✅ Reasonably consistent predictions")
    else:
        print("   ⚠️  Some prediction variability detected")
    
    # Input sensitivity test
    print(f"\n🧪 Input Sensitivity Test:")
    original_pred = model.predict(X_test_scaled[:100], verbose=0)
    
    # Add small noise to inputs
    noise_level = 0.01
    X_noisy = X_test_scaled[:100] + np.random.normal(0, noise_level, X_test_scaled[:100].shape)
    noisy_pred = model.predict(X_noisy, verbose=0)
    
    sensitivity = np.mean(np.abs(original_pred - noisy_pred))
    print(f"   Sensitivity to {noise_level*100}% input noise: ${sensitivity * 100000:.2f}")
    
    if sensitivity < 0.05:
        print("   ✅ Low sensitivity to input noise")
    elif sensitivity < 0.1:
        print("   ⚠️  Moderate sensitivity to input noise")
    else:
        print("   ❌ High sensitivity to input noise")

def business_impact_analysis(mae_dollars, rmse_dollars, r2):
    """Analyze business impact of model performance"""
    print("\n💼 BUSINESS IMPACT ANALYSIS")
    print("="*35)
    
    print(f"📈 Model Quality Assessment:")
    if r2 > 0.85:
        quality = "Excellent"
        recommendation = "Ready for production deployment"
    elif r2 > 0.7:
        quality = "Good"
        recommendation = "Suitable for production with monitoring"
    elif r2 > 0.5:
        quality = "Fair"
        recommendation = "Consider model improvements before production"
    else:
        quality = "Poor"
        recommendation = "Significant improvements needed"
    
    print(f"   Overall quality: {quality}")
    print(f"   Recommendation: {recommendation}")
    
    print(f"\n💰 Financial Impact:")
    print(f"   Typical prediction error: ±${mae_dollars:,.0f}")
    print(f"   95% confidence interval: ±${rmse_dollars * 1.96:,.0f}")
    
    # Risk assessment
    if mae_dollars < 30000:
        risk_level = "Low"
    elif mae_dollars < 50000:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    print(f"   Business risk level: {risk_level}")

def create_performance_report(results, metadata):
    """Create a comprehensive performance report"""
    print("\n📋 COMPREHENSIVE PERFORMANCE REPORT")
    print("="*50)
    
    print(f"🏠 Model Information:")
    print(f"   Model version: {metadata.get('model_version', 'Unknown')}")
    print(f"   Training date: {metadata.get('training_date', 'Unknown')}")
    print(f"   Framework: {metadata.get('framework', 'Unknown')} v{metadata.get('framework_version', 'Unknown')}")
    
    print(f"\n📊 Performance Summary:")
    print(f"   R² Score: {results['r2']:.4f}")
    print(f"   Mean Absolute Error: ${results['mae_dollars']:,.2f}")
    print(f"   Root Mean Square Error: ${results['rmse_dollars']:,.2f}")
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save detailed results
    report = {
        'timestamp': timestamp,
        'model_info': metadata,
        'performance_metrics': {
            'r2_score': float(results['r2']),
            'mae_dollars': float(results['mae_dollars']),
            'rmse_dollars': float(results['rmse_dollars'])
        }
    }
    
    os.makedirs('reports', exist_ok=True)
    report_file = f"reports/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main testing pipeline"""
    print("🧪 HOUSING PRICE MODEL - COMPREHENSIVE TESTING")
    print("="*55)
    
    # Load everything
    X_test, y_test, feature_names = load_test_data()
    model, scaler, metadata = load_model_artifacts()
    
    if model is None:
        print("❌ Cannot proceed without model artifacts")
        return
    
    # Run all tests
    print(f"\n🎯 Testing on {len(X_test)} samples...")
    
    # 1. Detailed performance analysis
    results = detailed_performance_analysis(model, X_test, y_test, scaler, feature_names)
    
    # 2. Segment analysis
    segment_analysis(y_test, results['predictions'], X_test, feature_names)
    
    # 3. Reliability tests
    model_reliability_tests(model, X_test, scaler)
    
    # 4. Business impact
    business_impact_analysis(results['mae_dollars'], results['rmse_dollars'], results['r2'])
    
    # 5. Generate report
    create_performance_report(results, metadata)
    
    print("\n✨ Testing completed successfully!")
    print("\n💡 Quick Start for API Testing:")
    print("   python test_api.py")

if __name__ == "__main__":
    main()