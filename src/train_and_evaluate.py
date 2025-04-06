#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
import gc

# Add the src directory to the path so we can import our model
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our model functions
from models.prediction_model import load_data, prepare_features, split_time_data, create_model, train_model, save_model
from models.model_selection import ModelSelector

def create_evaluation_plots(y_true, y_pred, title_prefix, output_dir):
    """Create and save evaluation plots."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Scatter plot of predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.title(f'{title_prefix}: Predicted vs Actual')
    plt.xlabel('Actual per_pago')
    plt.ylabel('Predicted per_pago')
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}_scatter.png'))
    plt.close()
    
    # Histogram of prediction errors
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'{title_prefix}: Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}_errors.png'))
    plt.close()

def evaluate_model_performance(y_true, y_pred, title_prefix):
    """Calculate and print model performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{title_prefix} Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def analyze_feature_importance(model, feature_names, output_dir):
    """Analyze and plot feature importance."""
    # Get feature importances from the RandomForestRegressor
    feature_importances = model.named_steps['regressor'].feature_importances_
    
    # Get the feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    all_features = np.concatenate([numeric_features, categorical_features])
    
    # Create a DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': feature_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importances (top 20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    return importance_df

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', f'model_evaluation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_data('data/base_cobranca')
    print(f"Total data size: {len(df)}")
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    df.info()
    
    # Save dataset info to file
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
        f.write(f"Total data size: {len(df)}\n\n")
        f.write("Dataset Info:\n")
        df.info(buf=f)
    
    print("\nSplitting data into training and out-of-time sets...")
    train_data, oot_data = split_time_data(df)
    
    # Clear memory
    del df
    gc.collect()
    
    # Save data split info
    with open(os.path.join(output_dir, 'data_split_info.txt'), 'w') as f:
        f.write(f"Training data size: {len(train_data)}\n")
        f.write(f"Out-of-time data size: {len(oot_data)}\n")
    
    # Plot distribution of per_pago in both sets
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(train_data['per_pago'], kde=True)
    plt.title('Distribution of per_pago in Training Data')
    plt.xlabel('per_pago')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.histplot(oot_data['per_pago'], kde=True)
    plt.title('Distribution of per_pago in Out-of-Time Data')
    plt.xlabel('per_pago')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_pago_distribution.png'))
    plt.close()
    
    print("\nPreparing features...")
    X_train, y_train, numeric_columns, categorical_columns = prepare_features(train_data)
    X_oot, y_oot, _, _ = prepare_features(oot_data)
    
    # Clear memory
    del train_data
    del oot_data
    gc.collect()
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print("\nStarting model selection and hyperparameter tuning...")
    model_selector = ModelSelector(output_dir)
    best_model, best_type = model_selector.compare_models(X_train, y_train, X_val, y_val)
    
    # Clear memory
    del X_train
    del X_val
    del y_train
    del y_val
    gc.collect()
    
    # Make predictions on out-of-time data
    y_oot_pred = best_model.predict(X_oot)
    
    # Evaluate model performance
    oot_metrics = evaluate_model_performance(y_oot, y_oot_pred, "Out-of-Time Data")
    
    # Create evaluation plots
    create_evaluation_plots(y_oot, y_oot_pred, "Out-of-Time Data", output_dir)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(best_model, numeric_columns + categorical_columns, output_dir)
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Out-of-Time': [oot_metrics['mse'], oot_metrics['rmse'], oot_metrics['mae'], oot_metrics['r2']]
    })
    
    metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
    
    # Save feature importance to file
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Save the model
    save_model(best_model, os.path.join(output_dir, 'per_pago_prediction_model.joblib'))
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    print(f"Model saved to {os.path.join(output_dir, 'per_pago_prediction_model.joblib')}")

if __name__ == "__main__":
    main() 