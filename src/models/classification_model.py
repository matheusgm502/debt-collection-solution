import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """Load and prepare the data for modeling."""
    df = pd.read_csv(file_path, delimiter='\t')
    return df

def convert_to_payment_class(per_pago):
    """Convert continuous payment ratio to discrete classes."""
    if per_pago == 0:
        return 'no_payment'
    elif per_pago <= 0.33:
        return 'low_payment'
    elif per_pago <= 0.66:
        return 'medium_payment'
    else:
        return 'full_payment'

def prepare_features(df):
    """Prepare features for the classification model."""
    # Sort and create previous appearances feature
    df = df.sort_values(['documento', 'mes_pagamento'])
    df['previous_appearances'] = df.groupby('documento').cumcount()
    
    # Define features
    numeric_columns = ['dias_atraso', 'saldo_vencido', 'previous_appearances', 
                      'VAR_5', 'VAR_166', 'IDADE', 'VAR_260', 'VAR_21', 'VAR_258']
    categorical_columns = ['VAR_114', 'segmento_veiculo', 'VAR_2', 'VAR_135', 
                         'UF', 'VAR_307', 'VAR_314', 'VAR_120']

    print(f"Using {len(numeric_columns)} numeric features: {numeric_columns}")
    print(f"Using {len(categorical_columns)} categorical features: {categorical_columns}")
    
    # Create feature matrix X
    X = df[numeric_columns + categorical_columns]
    
    # Convert target to classes
    y = df['per_pago'].apply(convert_to_payment_class)
    
    # Initialize LabelEncoder for the target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, numeric_columns, categorical_columns, le

def split_time_data(df):
    """Split data into training and out-of-time validation sets based on months."""
    df['mes_pagamento'] = pd.to_datetime(df['mes_pagamento'].astype(str), format='%Y%m')
    
    oot_start = pd.to_datetime('202210', format='%Y%m')
    oot_end = pd.to_datetime('202301', format='%Y%m')
    
    train_data = df[df['mes_pagamento'] < oot_start]
    oot_data = df[(df['mes_pagamento'] >= oot_start) & (df['mes_pagamento'] <= oot_end)]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Out-of-time data size: {len(oot_data)}")
    
    return train_data, oot_data

def create_classification_model(numeric_columns, categorical_columns):
    """Create and return the classification model pipeline."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    return model

def analyze_feature_importance(model, feature_names, output_dir=None):
    """Extract and visualize feature importance from the model."""
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    all_features = np.concatenate([numeric_features, categorical_features])
    
    importances = classifier.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_feature_importance.png'))
        plt.close()
        
        feature_importance.to_csv(os.path.join(output_dir, 'classification_feature_importance.csv'), 
                                index=False)
    
    return feature_importance

def train_classification_model(X_train, y_train, X_oot, y_oot, model, label_encoder, output_dir=None):
    """Train the classification model and evaluate performance."""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_oot_pred = model.predict(X_oot)
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Print classification reports
    print("\nTraining Data Classification Report:")
    print(classification_report(y_train, y_train_pred, 
                              target_names=class_names))
    
    print("\nOut-of-Time Data Classification Report:")
    print(classification_report(y_oot, y_oot_pred, 
                              target_names=class_names))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training confusion matrix
        sns.heatmap(confusion_matrix(y_train, y_train_pred), 
                   annot=True, fmt='d', ax=ax1, cmap='Blues')
        ax1.set_title('Training Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # OOT confusion matrix
        sns.heatmap(confusion_matrix(y_oot, y_oot_pred), 
                   annot=True, fmt='d', ax=ax2, cmap='Blues')
        ax2.set_title('Out-of-Time Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_confusion_matrices.png'))
        plt.close()
        
        # Save the label encoder with the model
        joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    
    return model, y_train_pred, y_oot_pred

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

def main():
    # Load data
    df = load_data('data/base_cobranca')
    
    # Split data into training and out-of-time sets
    train_data, oot_data = split_time_data(df)
    
    # Prepare features
    X_train, y_train, numeric_columns, categorical_columns, label_encoder = prepare_features(train_data)
    X_oot, y_oot, _, _, _ = prepare_features(oot_data)
    
    # Create output directory
    output_dir = 'output/classification_model_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and train model
    model = create_classification_model(numeric_columns, categorical_columns)
    trained_model, y_train_pred, y_oot_pred = train_classification_model(
        X_train, y_train, X_oot, y_oot, model, label_encoder, output_dir
    )
    
    # Save the model
    save_model(trained_model, 'output/per_pago_classification_model.joblib')

if __name__ == "__main__":
    main() 