import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import gc
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

def load_data(file_path):
    """Load and prepare the data for modeling."""
    df = pd.read_csv(file_path, delimiter='\t')
    return df

def convert_to_payment_class(per_pago):
    """Convert continuous payment ratio to discrete classes."""
    if per_pago == 0:
        return 'no_payment'
    elif per_pago > 0 and per_pago <1:
        return 'partial_payment'
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
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create output directory
    output_dir = 'output/classification_model_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model selector and perform model comparison
    print("\nStarting model selection and hyperparameter optimization...")
    model_selector = ClassificationModelSelector(output_dir)
    best_model, best_type = model_selector.compare_models(X_train, y_train, X_val, y_val)
    
    # Train the best model on the full training set and evaluate on OOT data
    print("\nTraining best model on full training set...")
    trained_model, y_train_pred, y_oot_pred = train_classification_model(
        X_train, y_train, X_oot, y_oot, best_model, label_encoder, output_dir
    )
    
    # Save the model
    save_model(trained_model, os.path.join(output_dir, 'per_pago_classification_model.joblib'))
    
    print(f"\nModel selection and training complete! Results saved to {output_dir}")
    print(f"Best model type: {best_type.upper()}")

class ClassificationModelSelector:
    def __init__(self, output_dir=None):
        """Initialize the classification model selector with output directory."""
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join('output', f'classification_model_selection_{timestamp}')
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define parameter grids for each model type
        self.rf_param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt'],
            'classifier__class_weight': ['balanced']
        }
        
        self.xgb_param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8],
            'classifier__colsample_bytree': [0.8],
            'classifier__min_child_weight': [1],
            'classifier__scale_pos_weight': [1, 2]  # Handle class imbalance
        }
        
        self.lr_param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__max_iter': [1000],
            'classifier__class_weight': ['balanced'],
            'classifier__solver': ['lbfgs']
        }
    
    def create_model_pipeline(self, model_type, preprocessor):
        """Create a model pipeline with the specified model type."""
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=42)
        elif model_type == 'xgb':
            model = XGBClassifier(random_state=42)
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    def grid_search(self, X_train, y_train, X_val, y_val, model_type, param_grid, cv=5):
        """Perform grid search with progress bar and memory optimization."""
        print(f"\nStarting grid search for {model_type.upper()}...")
        
        # Create base pipeline
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
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
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Initialize best model and score
        best_score = 0  # For classification, higher is better
        best_model = None
        best_params = None
        results_list = []
        
        # Calculate total combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        # Create progress bar
        pbar = tqdm(total=total_combinations, desc=f'Grid Search {model_type.upper()}')
        
        # Process each parameter combination
        for params in param_combinations:
            # Create and fit pipeline
            pipeline = self.create_model_pipeline(model_type, preprocessor)
            pipeline.set_params(**params)
            
            # Fit the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            val_pred = pipeline.predict(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, val_pred)
            f1 = f1_score(y_val, val_pred, average='weighted')
            
            # Update best model if necessary (using F1 score as metric)
            if f1 > best_score:
                best_score = f1
                best_model = pipeline
                best_params = params
            
            # Store results
            results_list.append({
                'params': params,
                'accuracy': accuracy,
                'f1_score': f1
            })
            
            # Update progress bar
            pbar.update(1)
            
            # Clear memory
            del pipeline
            del val_pred
            gc.collect()
        
        pbar.close()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Save results
        self._save_results({
            'model_type': model_type,
            'best_params': best_params,
            'best_f1_score': best_score,
            'validation_accuracy': accuracy_score(y_val, best_model.predict(X_val)),
            'validation_f1': f1_score(y_val, best_model.predict(X_val), average='weighted'),
            'cv_results': results_df
        }, best_model, model_type)
        
        return best_model, {
            'model_type': model_type,
            'best_params': best_params,
            'best_f1_score': best_score,
            'validation_accuracy': accuracy_score(y_val, best_model.predict(X_val)),
            'validation_f1': f1_score(y_val, best_model.predict(X_val), average='weighted'),
            'cv_results': results_df
        }
    
    def _save_results(self, results, model, model_type):
        """Save model and results to files."""
        # Save model
        model_path = os.path.join(self.output_dir, f'best_{model_type}_model.joblib')
        joblib.dump(model, model_path)
        
        # Save results
        results_path = os.path.join(self.output_dir, f'{model_type}_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'model_type': results['model_type'],
                'best_params': results['best_params'],
                'best_f1_score': results['best_f1_score'],
                'validation_accuracy': results['validation_accuracy'],
                'validation_f1': results['validation_f1']
            }, f, indent=4)
        
        # Save detailed CV results
        cv_results_path = os.path.join(self.output_dir, f'{model_type}_cv_results.csv')
        results['cv_results'].to_csv(cv_results_path, index=False)
    
    def compare_models(self, X_train, y_train, X_val, y_val):
        """Compare Random Forest, XGBoost, and Logistic Regression models."""
        print("Starting model comparison...")
        
        # Grid search for Random Forest
        print("\nTraining Random Forest model...")
        rf_model, rf_results = self.grid_search(
            X_train, y_train, X_val, y_val,
            'rf', self.rf_param_grid
        )
        
        # Clear memory
        gc.collect()
        
        # Grid search for XGBoost
        print("\nTraining XGBoost model...")
        xgb_model, xgb_results = self.grid_search(
            X_train, y_train, X_val, y_val,
            'xgb', self.xgb_param_grid
        )
        
        # Clear memory
        gc.collect()
        
        # Grid search for Logistic Regression
        print("\nTraining Logistic Regression model...")
        lr_model, lr_results = self.grid_search(
            X_train, y_train, X_val, y_val,
            'lr', self.lr_param_grid
        )
        
        # Compare results
        comparison = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Logistic Regression'],
            'Validation Accuracy': [
                rf_results['validation_accuracy'],
                xgb_results['validation_accuracy'],
                lr_results['validation_accuracy']
            ],
            'Validation F1': [
                rf_results['validation_f1'],
                xgb_results['validation_f1'],
                lr_results['validation_f1']
            ]
        })
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison.to_csv(comparison_path, index=False)
        
        print("\nModel Comparison Results:")
        print(comparison.to_string(index=False))
        
        # Return the best model based on validation F1 score
        models_and_scores = [
            (rf_model, 'rf', rf_results['validation_f1']),
            (xgb_model, 'xgb', xgb_results['validation_f1']),
            (lr_model, 'lr', lr_results['validation_f1'])
        ]
        
        best_model, best_type, best_score = max(models_and_scores, key=lambda x: x[2])
        
        print(f"\nBest model: {best_type.upper()} (F1 Score: {best_score:.4f})")
        return best_model, best_type

if __name__ == "__main__":
    main() 