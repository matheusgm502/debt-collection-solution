import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
import joblib
import os
import json
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import gc

class ModelSelector:
    def __init__(self, output_dir=None):
        """Initialize the model selector with output directory."""
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join('output', f'model_selection_{timestamp}')
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define smaller parameter grids for memory efficiency
        self.rf_param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 15],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2],
            'regressor__max_features': ['sqrt']
        }
        
        self.xgb_param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 5],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0],
            'regressor__min_child_weight': [1, 3]
        }
    
    def create_model_pipeline(self, model_type, preprocessor):
        """Create a model pipeline with the specified model type."""
        if model_type == 'rf':
            model = RandomForestRegressor(random_state=42)
        elif model_type == 'xgb':
            model = XGBRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    
    def grid_search(self, X_train, y_train, X_val, y_val, model_type, param_grid, cv=5):
        """Perform grid search with progress bar and memory optimization."""
        print(f"\nStarting grid search for {model_type.upper()}...")
        
        # Create base pipeline
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_cols)
            ])
        
        # Initialize best model and score
        best_score = float('inf')
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
            mse = mean_squared_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)
            
            # Update best model if necessary
            if mse < best_score:
                best_score = mse
                best_model = pipeline
                best_params = params
            
            # Store results
            results_list.append({
                'params': params,
                'mse': mse,
                'r2': r2
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
            'best_cv_score': best_score,
            'validation_mse': best_score,
            'validation_r2': r2_score(y_val, best_model.predict(X_val)),
            'cv_results': results_df
        }, best_model, model_type)
        
        return best_model, {
            'model_type': model_type,
            'best_params': best_params,
            'best_cv_score': best_score,
            'validation_mse': best_score,
            'validation_r2': r2_score(y_val, best_model.predict(X_val)),
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
                'best_cv_score': results['best_cv_score'],
                'validation_mse': results['validation_mse'],
                'validation_r2': results['validation_r2']
            }, f, indent=4)
        
        # Save detailed CV results
        cv_results_path = os.path.join(self.output_dir, f'{model_type}_cv_results.csv')
        results['cv_results'].to_csv(cv_results_path, index=False)
    
    def compare_models(self, X_train, y_train, X_val, y_val):
        """Compare Random Forest and XGBoost models."""
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
        
        # Compare results
        comparison = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'],
            'Validation MSE': [rf_results['validation_mse'], xgb_results['validation_mse']],
            'Validation R²': [rf_results['validation_r2'], xgb_results['validation_r2']],
            'CV Score': [rf_results['best_cv_score'], xgb_results['best_cv_score']]
        })
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison.to_csv(comparison_path, index=False)
        
        print("\nModel Comparison Results:")
        print(comparison.to_string(index=False))
        
        # Return the best model based on validation R²
        best_model = rf_model if rf_results['validation_r2'] > xgb_results['validation_r2'] else xgb_model
        best_type = 'rf' if rf_results['validation_r2'] > xgb_results['validation_r2'] else 'xgb'
        
        print(f"\nBest model: {best_type.upper()}")
        return best_model, best_type 