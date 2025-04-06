#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from models.prediction_model import load_data, prepare_features

MODEL_PATH = 'output/model_evaluation_20250406_102230/best_xgb_model.joblib'
def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', f'predictions_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model_path = MODEL_PATH
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load the data
    print("Loading data...")
    df = load_data('data/base_cobranca')
    df['previous_appearances'] = df.groupby('documento').cumcount()
    # Prepare features
    print("Preparing features...")
    X, y, numeric_columns, categorical_columns = prepare_features(df)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'id': df['id'],
        'documento': df['documento'],
        'mes_pagamento': df['mes_pagamento'],
        'per_pago': y,
        'predicted_per_pago': predictions
    })
    
    # Add original features for reference
    for col in numeric_columns + categorical_columns:
        results_df[col] = df[col]
    
    # Save results
    output_file = os.path.join(output_dir, 'predictions.csv')
    print(f"Saving predictions to {output_file}")
    results_df.to_csv(output_file, index=False)
    
    print("\nPrediction Summary:")
    print(f"Total records processed: {len(results_df)}")
    print(f"Average predicted per_pago: {results_df['predicted_per_pago'].mean():.4f}")
    print(f"Average actual per_pago: {results_df['per_pago'].mean():.4f}")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 