#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Payment Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set title and description
st.title("Payment Prediction Dashboard")
st.markdown("""
This dashboard displays users and their probability of paying based on the prediction model.
""")

# Load the data directly from the CSV file
try:
    # Load the data
    df = pd.read_csv("output/classification_predictions_20250406_114904/classification_predictions.csv")
    df=df[df['mes_pagamento']>=202210]
    st.sidebar.success("Data loaded successfully!")
    
    # Check if required columns exist
    required_columns = ['id', 'documento', 'segmento_veiculo', 'dias_atraso', 'saldo_vencido','actual_class','predicted_class']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.sidebar.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()
        
    # Check if per_pago exists (for evaluation if available)
    has_actual_payment = 'per_pago' in df.columns
    
    # Data filtering options
    st.sidebar.header("Filter Data")
    
    # Filter by vehicle segment
    segment_options = ['All'] + sorted(df['segmento_veiculo'].unique().tolist())
    selected_segment = st.sidebar.selectbox("Vehicle Segment", segment_options)
    
    # Filter by payment month if available
    if 'mes_pagamento' in df.columns:
        month_options = ['All'] + sorted(df['mes_pagamento'].unique().tolist())
        selected_month = st.sidebar.selectbox("Payment Month", month_options)
    else:
        selected_month = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['segmento_veiculo'] == selected_segment]
    if selected_month != 'All' and 'mes_pagamento' in df.columns:
        filtered_df = filtered_df[filtered_df['mes_pagamento'] == selected_month]
    
    # Display data summary
    st.header("Data Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(filtered_df))
    col2.metric("Average Days of Delay", f"{filtered_df['dias_atraso'].mean():.1f}")
    col3.metric("Average Outstanding Balance", f"R$ {filtered_df['saldo_vencido'].mean():.2f}")
    
    # Display users and their payment probabilities
    st.header("Users and Payment Probabilities")
    
    # Sort by predicted class
    filtered_df_sorted = filtered_df.sort_values('predicted_class', ascending=False)
    
    # Display the top 100 users
    st.subheader("Top 100 Users by Predicted Payment Class")
    display_columns = ['id', 'documento', 'segmento_veiculo', 'dias_atraso', 'saldo_vencido', 'predicted_class']
    if has_actual_payment:
        display_columns.append('actual_class')
        
    top_users = filtered_df_sorted.head(100)[display_columns]
    column_names = {
        'id': 'ID', 
        'documento': 'Document', 
        'segmento_veiculo': 'Vehicle Segment', 
        'dias_atraso': 'Days of Delay', 
        'saldo_vencido': 'Outstanding Balance', 
        'predicted_class': 'Predicted Payment Class',
        'actual_class': 'Actual Payment Class'
    }
    top_users.columns = [column_names.get(col, col) for col in top_users.columns]
    st.dataframe(top_users, use_container_width=True)
    
    # Visualizations
    st.header("Visualizations")
    
    # Create two columns for visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Distribution of payment classes
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='predicted_class', ax=ax)
        ax.set_title('Distribution of Predicted Payment Classes')
        ax.set_xlabel('Payment Class')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with viz_col2:
        # Box plot of days of delay by predicted class
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='predicted_class', y='dias_atraso', data=filtered_df, ax=ax)
        ax.set_title('Days of Delay by Predicted Payment Class')
        ax.set_xlabel('Predicted Payment Class')
        ax.set_ylabel('Days of Delay')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Payment class by vehicle segment
    st.subheader("Payment Classes by Vehicle Segment")
    fig, ax = plt.subplots(figsize=(10, 6))
    pd.crosstab(filtered_df['segmento_veiculo'], filtered_df['predicted_class']).plot(kind='bar', ax=ax)
    ax.set_title('Payment Classes by Vehicle Segment')
    ax.set_xlabel('Vehicle Segment')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # If actual payment data is available, show model performance
    if has_actual_payment:
        st.subheader("Model Performance")
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = filtered_df['predicted_class']
        y_true = filtered_df['actual_class']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Accuracy", f"{accuracy:.3f}")
        metric_col2.metric("Precision", f"{precision:.3f}")
        metric_col3.metric("Recall", f"{recall:.3f}")
        metric_col4.metric("F1 Score", f"{f1:.3f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    # Download data
    st.header("Download Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"payment_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
except Exception as e:
    st.error(f"Error processing data: {str(e)}")