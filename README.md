# Payment Prediction Model

This project contains a machine learning model to predict the payment ratio (`per_pago`) for vehicle-related debts.

## Model Description

The model predicts the proportion of debt that will be paid (`per_pago`) based on various features including:
- Days of delay (`dias_atraso`)
- Outstanding balance (`saldo_vencido`)
- Vehicle segment (`segmento_veiculo`)

## Validation Strategy

The model uses an out-of-time validation approach:
- Training data: All data before October 2022
- Out-of-time validation: Data from October 2022 to January 2023

This approach ensures the model's performance is evaluated on future data, providing a more realistic assessment of its predictive power.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training and Evaluation

To train and evaluate the model with detailed metrics and visualizations:

```bash
python src/train_and_evaluate.py
```

This will:
- Load the data from `data/base_cobranca`
- Split data into training and out-of-time validation sets
- Train the model on data before October 2022
- Evaluate the model on data from October 2022 to January 2023
- Generate detailed performance metrics and visualizations
- Save all results to a timestamped directory in `output/`
- Save the trained model to the output directory

### Basic Training

For a simpler training process without visualizations:

```bash
python src/models/prediction_model.py
```

This will:
- Load the data from `data/base_cobranca`
- Split data into training and out-of-time validation sets
- Train the model on data before October 2022
- Evaluate the model on data from October 2022 to January 2023
- Save the trained model to `models/per_pago_prediction_model.joblib`
- Print performance metrics for both training and out-of-time data

### Streamlit Dashboard

To run the interactive dashboard that displays users and their payment probabilities:

```bash
./src/app/run_app.sh
```

Or directly with Streamlit:

```bash
streamlit run src/app/app.py
```

The dashboard provides:
- Model selection from available trained models
- Data filtering by vehicle segment and payment month
- Summary metrics of the filtered data
- Top 100 users sorted by payment probability
- Visualizations of payment probabilities
- Option to download the filtered data with predictions

## Model Details

The model uses a Random Forest Regressor with the following preprocessing steps:
- Standard scaling for numeric features
- One-hot encoding for categorical features
- Pipeline to combine preprocessing and model training

## Evaluation Output

The comprehensive evaluation script (`train_and_evaluate.py`) generates:

1. **Performance Metrics**:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score

2. **Visualizations**:
   - Distribution of `per_pago` in training and out-of-time data
   - Scatter plots of predicted vs actual values
   - Histograms of prediction errors
   - Feature importance plot

3. **Data Files**:
   - Dataset information
   - Data split information
   - Model metrics in CSV format
   - Feature importance in CSV format
   - Trained model in joblib format

All outputs are saved to a timestamped directory in `output/` for easy tracking of different model versions.

## Data Format

The input data should be a tab-separated file with the following columns:
- `mes_pagamento`: Payment month in YYYYMM format
- `dias_atraso`: Number of days of delay
- `saldo_vencido`: Outstanding balance
- `segmento_veiculo`: Vehicle segment (e.g., 'leves', 'motos')
- `per_pago`: Target variable (proportion of debt paid) 