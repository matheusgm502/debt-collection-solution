#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from models.classification_model import load_data, prepare_features, convert_to_payment_class

# Paths to the classification model and label encoder
MODEL_PATH = 'output/classification_model_evaluation/best_xgb_model.joblib'
LABEL_ENCODER_PATH = 'output/classification_model_evaluation/label_encoder.joblib'

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('output', f'classification_predictions_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model and label encoder
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    print(f"Loading label encoder from {LABEL_ENCODER_PATH}")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # Load the data
    print("Loading data...")
    df = load_data('data/base_cobranca')
    
    # Create previous_appearances feature (this is done in prepare_features but we need it for the results)
    print("Creating previous_appearances feature...")
    df = df.sort_values(['documento', 'mes_pagamento'])
    df['previous_appearances'] = df.groupby('documento').cumcount()
    
    # Prepare features
    print("Preparing features...")
    X, y, numeric_columns, categorical_columns, _ = prepare_features(df)
    
    # Make predictions
    print("Making predictions...")
    predicted_classes = model.predict(X)
    
    # Convert numeric predictions back to class names
    predicted_class_names = label_encoder.inverse_transform(predicted_classes)
    
    # Convert actual per_pago values to class names for comparison
    actual_class_names = df['per_pago'].apply(convert_to_payment_class)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'id': df['id'],
        'documento': df['documento'],
        'mes_pagamento': df['mes_pagamento'],
        'per_pago': df['per_pago'],
        'actual_class': actual_class_names,
        'predicted_class': predicted_class_names
    })
    
    # Add prediction probabilities if available
    try:
        prediction_probas = model.predict_proba(X)
        proba_df = pd.DataFrame(
            prediction_probas, 
            columns=[f'prob_{class_name}' for class_name in label_encoder.classes_]
        )
        results_df = pd.concat([results_df, proba_df], axis=1)
    except:
        print("Note: Probability predictions not available for this model")
    
    # Add original features for reference
    for col in numeric_columns + categorical_columns:
        results_df[col] = df[col]
    
    # Save results
    output_file = os.path.join(output_dir, 'classification_predictions.csv')
    print(f"Saving predictions to {output_file}")
    results_df.to_csv(output_file, index=False)
    
    # Print prediction summary
    print("\nPrediction Summary:")
    print(f"Total records processed: {len(results_df)}")
    
    # Class distribution
    print("\nActual Class Distribution:")
    actual_dist = results_df['actual_class'].value_counts(normalize=True)
    print(actual_dist)
    
    print("\nPredicted Class Distribution:")
    pred_dist = results_df['predicted_class'].value_counts(normalize=True)
    print(pred_dist)
    
    # Accuracy metrics
    accuracy = (results_df['actual_class'] == results_df['predicted_class']).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    for class_name in label_encoder.classes_:
        class_mask = results_df['actual_class'] == class_name
        if class_mask.sum() > 0:
            class_accuracy = (results_df.loc[class_mask, 'actual_class'] == 
                            results_df.loc[class_mask, 'predicted_class']).mean()
            print(f"{class_name}: {class_accuracy:.4f}")
    
    # Save summary metrics to a separate file
    summary_file = os.path.join(output_dir, 'classification_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Classification Prediction Summary\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total records processed: {len(results_df)}\n\n")
        
        f.write("Actual Class Distribution:\n")
        f.write(actual_dist.to_string())
        f.write("\n\nPredicted Class Distribution:\n")
        f.write(pred_dist.to_string())
        f.write(f"\n\nOverall Accuracy: {accuracy:.4f}\n\n")
        
        f.write("Class-wise Accuracy:\n")
        for class_name in label_encoder.classes_:
            class_mask = results_df['actual_class'] == class_name
            if class_mask.sum() > 0:
                class_accuracy = (results_df.loc[class_mask, 'actual_class'] == 
                                results_df.loc[class_mask, 'predicted_class']).mean()
                f.write(f"{class_name}: {class_accuracy:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 