import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """Load and prepare the data for modeling."""
    df = pd.read_csv(file_path, delimiter='\t')
    return df

def prepare_features(df):
    """Prepare features for the model using all available variables."""
    # Identify numeric and categorical columns
    # Exclude columns that are not useful for prediction
    df = df.sort_values(['documento', 'mes_pagamento'])
    df['previous_appearances'] = df.groupby('documento').cumcount()
    numeric_columns = ['dias_atraso','saldo_vencido','previous_appearances', 'VAR_5', 'VAR_166', 'IDADE', 'VAR_260', 'VAR_21', 'VAR_258']
    categorical_columns = ['VAR_114', 'segmento_veiculo', 'VAR_2', 'VAR_135', 'UF', 'VAR_307', 'VAR_314', 'VAR_120']

    print(f"Using {len(numeric_columns)} numeric features: {numeric_columns}")
    print(f"Using {len(categorical_columns)} categorical features: {categorical_columns}")
    
    # Create feature matrix X and target variable y
    X = df[numeric_columns + categorical_columns]
    y = df['per_pago']
    
    return X, y, numeric_columns, categorical_columns

def split_time_data(df):
    """Split data into training and out-of-time validation sets based on months."""
    # Convert mes_pagamento to datetime for comparison
    df['mes_pagamento'] = pd.to_datetime(df['mes_pagamento'].astype(str), format='%Y%m')
    
    # Define out-of-time period
    oot_start = pd.to_datetime('202210', format='%Y%m')
    oot_end = pd.to_datetime('202301', format='%Y%m')
    
    # Split data
    train_data = df[df['mes_pagamento'] < oot_start]
    oot_data = df[(df['mes_pagamento'] >= oot_start) & (df['mes_pagamento'] <= oot_end)]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Out-of-time data size: {len(oot_data)}")
    
    return train_data, oot_data

def create_model(numeric_columns, categorical_columns):
    """Create and return the model pipeline with all available features."""
    # Create preprocessing steps with null value handling
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        # Use 'MISSING' as a fill value for categorical features
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    
    # Create the model pipeline with more trees and deeper trees
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,  # More trees
            max_depth=15,      # Deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    return model

def analyze_feature_importance(model, feature_names, output_dir=None):
    """Extract and print feature importance from the model."""
    # Get the regressor from the pipeline
    regressor = model.named_steps['regressor']
    
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    all_features = np.concatenate([numeric_features, categorical_features])
    
    # Get feature importance
    importances = regressor.feature_importances_
    
    # Create a DataFrame with feature names and importance scores
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    # Create feature importance plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot feature importances (top 20)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    return feature_importance

def train_model(X_train, y_train, X_oot, y_oot, model, output_dir=None):
    """Train the model and evaluate on both training and out-of-time data."""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on training data
    y_train_pred = model.predict(X_train)
    
    # Make predictions on out-of-time data
    y_oot_pred = model.predict(X_oot)
    
    # Calculate metrics for training data
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for out-of-time data
    oot_mse = mean_squared_error(y_oot, y_oot_pred)
    oot_r2 = r2_score(y_oot, y_oot_pred)
    
    print("Training Data Metrics:")
    print(f"Mean Squared Error: {train_mse:.4f}")
    print(f"R² Score: {train_r2:.4f}")
    
    print("\nOut-of-Time Data Metrics:")
    print(f"Mean Squared Error: {oot_mse:.4f}")
    print(f"R² Score: {oot_r2:.4f}")
    
    # Analyze feature importance if output directory is provided
    if output_dir:
        analyze_feature_importance(model, X_train.columns, output_dir)
    
    return model, y_train_pred, y_oot_pred

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

def main():
    # Load data
    df = load_data('data/base_cobranca')
    
    # Split data into training and out-of-time sets
    train_data, oot_data = split_time_data(df)
    
    # Prepare features for training data
    X_train, y_train, numeric_columns, categorical_columns = prepare_features(train_data)
    
    # Prepare features for out-of-time data
    X_oot, y_oot, _, _ = prepare_features(oot_data)
    
    # Create output directory for model evaluation
    output_dir = 'output/model_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and train model
    model = create_model(numeric_columns, categorical_columns)
    trained_model, y_train_pred, y_oot_pred = train_model(X_train, y_train, X_oot, y_oot, model, output_dir)
    
    # Save the model
    save_model(trained_model, 'models/per_pago_prediction_model.joblib')

if __name__ == "__main__":
    main() 