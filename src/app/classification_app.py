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
    page_title="Payment Classification Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set title and description
st.title("Payment Classification Dashboard")
st.markdown("""
This dashboard displays users and their predicted payment classes based on the classification model.
""")

# Sidebar for data filtering
st.sidebar.header("Settings")

# File upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with predictions", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data loaded successfully!")
        
        # Check if required columns exist
        required_columns = ['id', 'documento', 'segmento_veiculo', 'dias_atraso', 'saldo_vencido', 'predicted_class']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.sidebar.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()
            
        # Check if actual class exists (for evaluation if available)
        has_actual_class = 'actual_class' in df.columns
        
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
        
        # Filter by predicted class
        class_options = ['All'] + sorted(df['predicted_class'].unique().tolist())
        selected_class = st.sidebar.selectbox("Predicted Payment Class", class_options)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_segment != 'All':
            filtered_df = filtered_df[filtered_df['segmento_veiculo'] == selected_segment]
        if selected_month != 'All' and 'mes_pagamento' in df.columns:
            filtered_df = filtered_df[filtered_df['mes_pagamento'] == selected_month]
        if selected_class != 'All':
            filtered_df = filtered_df[filtered_df['predicted_class'] == selected_class]
        
        # Display data summary
        st.header("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users", len(filtered_df))
        col2.metric("Average Days of Delay", f"{filtered_df['dias_atraso'].mean():.1f}")
        col3.metric("Average Outstanding Balance", f"R$ {filtered_df['saldo_vencido'].mean():.2f}")
        
        # Calculate class distribution
        class_dist = filtered_df['predicted_class'].value_counts()
        col4.metric("Most Common Class", f"{class_dist.index[0]} ({class_dist.values[0]} users)")
        
        # Display class distribution
        st.subheader("Payment Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='predicted_class', ax=ax)
        plt.xticks(rotation=45)
        plt.title('Distribution of Predicted Payment Classes')
        st.pyplot(fig)
        
        # Display users and their predictions
        st.header("Users and Predictions")
        
        # Sort by outstanding balance (descending)
        filtered_df_sorted = filtered_df.sort_values('saldo_vencido', ascending=False)
        
        # Display the top 100 users
        st.subheader("Top 100 Users by Outstanding Balance")
        display_columns = ['id', 'documento', 'segmento_veiculo', 'dias_atraso', 
                         'saldo_vencido', 'predicted_class']
        if has_actual_class:
            display_columns.append('actual_class')
            
        top_users = filtered_df_sorted.head(100)[display_columns]
        column_names = {
            'id': 'ID', 
            'documento': 'Document', 
            'segmento_veiculo': 'Vehicle Segment', 
            'dias_atraso': 'Days of Delay', 
            'saldo_vencido': 'Outstanding Balance', 
            'predicted_class': 'Predicted Class',
            'actual_class': 'Actual Class'
        }
        top_users.columns = [column_names.get(col, col) for col in top_users.columns]
        st.dataframe(top_users, use_container_width=True)
        
        # Additional visualizations
        st.header("Additional Insights")
        
        # Create two columns for visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Payment class by vehicle segment
            plt.figure(figsize=(10, 6))
            segment_class_counts = pd.crosstab(filtered_df['segmento_veiculo'], 
                                             filtered_df['predicted_class'])
            segment_class_props = segment_class_counts.div(segment_class_counts.sum(axis=1), axis=0)
            segment_class_props.plot(kind='bar', stacked=True)
            plt.title('Payment Class Distribution by Vehicle Segment')
            plt.xlabel('Vehicle Segment')
            plt.ylabel('Proportion')
            plt.legend(title='Payment Class', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            st.pyplot(plt)
        
        with viz_col2:
            # Box plot of days of delay by predicted class
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=filtered_df, x='predicted_class', y='dias_atraso')
            plt.title('Days of Delay by Predicted Payment Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
        
        # If actual classes are available, show model performance
        if has_actual_class:
            st.header("Model Performance")
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(filtered_df['actual_class'], filtered_df['predicted_class'])
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sorted(filtered_df['predicted_class'].unique()),
                       yticklabels=sorted(filtered_df['actual_class'].unique()))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Class')
            plt.ylabel('Actual Class')
            st.pyplot(plt)
            
            # Display classification report
            st.subheader("Classification Report")
            report = classification_report(filtered_df['actual_class'], 
                                        filtered_df['predicted_class'])
            st.text(report)
        
        # Download data
        st.header("Download Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"payment_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a CSV file with the prediction data to begin.") 