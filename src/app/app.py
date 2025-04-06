#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from whatsapp_api.main import send_whatsapp_message

# Set page configuration
st.set_page_config(
    page_title="Dashboard de Cobrança de Dívidas",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set title and description
st.title("Dashboard de Cobrança de Dívidas")
st.markdown("""
Este dashboard exibe o plano de ação para a cobrança de dívidas, com base nas predições do modelo de classificação.
""")

# Load the data directly from the CSV file
try:
    # Load the data
    df = pd.read_csv("output/classification_predictions_20250406_114904/classification_predictions.csv")
    df=df[df['mes_pagamento']>=202210]
    st.sidebar.success("Data loaded successfully!")
    
    # Create treemap data
    treemap_data = df.groupby(['predicted_class', 'segmento_veiculo']).size().reset_index(name='count')
    treemap_data = treemap_data.dropna(subset=['segmento_veiculo'])
    
    # Define consistent mapping for prediction classes
    prediction_mapping = {
        'full_payment': 'Message Reminder',
        'partial_payment': 'Message for Renegociation',
        'no_payment': 'Collection Team'
    }
    
    # Define consistent color mapping for prediction classes
    color_mapping = {
        'Message Reminder': '#8dd3c7',  # Light teal
        'Message for Renegociation': '#ffffb3',  # Light yellow
        'Collection Team': '#fb8072'  # Light red
    }
    
    # Apply prediction mapping to treemap data
    treemap_data['predicted_class'] = treemap_data['predicted_class'].replace(prediction_mapping)

    treemap_data['segmento_veiculo'] = treemap_data['segmento_veiculo'].replace({
        'Carro': 'Car',
        'Moto': 'Motorcycle',
        'Caminhão': 'Truck',
        'Caminhão': 'Truck',
        'Caminhão': 'Truck',
    })
    # Create Plotly treemap
    
    fig = px.treemap(treemap_data,
                    path=['predicted_class', 'segmento_veiculo'],
                    values='count',
                    color='predicted_class',
                    color_discrete_map=color_mapping)
    
    # Update layout for better visibility
    fig.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>" +
                     "Count: %{value}<br>" +
                     "<extra></extra>"
    )
    
    # Add summary statistics
    st.subheader("Consolidado")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate summary statistics
    total_debt = df['saldo_vencido'].sum()
    avg_debt = df['saldo_vencido'].mean()
    total_clients = len(df)
    no_payment_count = len(df[df['predicted_class'] == 'no_payment'])
    
    # Display metrics
    with col1:
        st.metric("Dívida em atraso", f"R$ {total_debt/1000:,.0f}k")
    with col2:
        st.metric("Média de dívida", f"R$ {avg_debt:,.0f}")
    with col3:
        st.metric("Total de clientes", f"{total_clients:,}")
    with col4:
        st.metric("Previsão de não pagamento", f"{(no_payment_count/total_clients)*100:0.2f}%")
    # Create two columns for the charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribuição de Ações por perfil do Usuário")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Add debt distribution by prediction class
        st.subheader("Distribuição de dívidas por perfil de pagamento")
        
        # Calculate debt by prediction class
        debt_by_class = df.groupby('predicted_class')['saldo_vencido'].sum().reset_index()
        
        # Apply the same prediction mapping to debt_by_class
        debt_by_class['predicted_class'] = debt_by_class['predicted_class'].replace(prediction_mapping)
        
        # Create bar chart with consistent colors
        fig_debt = px.bar(
            debt_by_class,
            x='predicted_class',
            y='saldo_vencido',
            labels={'predicted_class': 'Ação', 'saldo_vencido': 'Dívida em atraso (R$)'},
            color='predicted_class',
            color_discrete_map=color_mapping
        )
        
        st.plotly_chart(fig_debt, use_container_width=True)

    # Add table for no_payment clients
    st.subheader("Lista de clientes para coleta (por ordem de prioridade)")
    
    # Filter for no_payment clients and sort by debt size
    no_payment_df = df[df['predicted_class'] == 'no_payment'].copy()
    

    no_payment_df = no_payment_df.sort_values(by='saldo_vencido', ascending=False)
    
    # Select columns to display
    display_columns = ['id','saldo_vencido', 'segmento_veiculo']
        
    # Rename columns for better display
    column_names = {
        'id': 'ID do cliente',
        'saldo_vencido': 'Dívida em atraso',
        'segmento_veiculo': 'Tipo de veículo'
    }
    
    # Display the table
    st.dataframe(
        no_payment_df[display_columns].rename(columns=column_names),
        use_container_width=True
    )

    # Add message delivery tracking
    st.subheader("Acompanhamento de entrega de mensagens")
    
    # Initialize session state for message tracking if it doesn't exist
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []
    
    # Display message history
    if st.session_state.message_history:
        message_df = pd.DataFrame(st.session_state.message_history)
        
        # Calculate success rate
        success_rate = (message_df['success'].sum() / len(message_df)) * 100
        
        # Display success rate
        st.metric("Message Delivery Success Rate", f"{success_rate:.1f}%")
        
        # Display message history table
        st.dataframe(
            message_df[['timestamp', 'recipient', 'prediction_class', 'success']],
            use_container_width=True
        )
    else:
        st.info("Nenhuma mensagem foi enviada ainda. Envie mensagens para acompanhar as taxas de entrega.")

    if st.button("Disparar mensagens de cobrança"):
        # Track successful and failed messages
        successful_messages = 0
        failed_messages = 0
        df_to_send = df[:10]
        for index, row in df_to_send.iterrows():
            message = None
            if not pd.isna(row['first_name']):
                if row['predicted_class'] == 'full_payment':
                    message = f"""Hello, {row['first_name']}
                    We noticed that you have not paid your last payment.
                    Please make sure to pay your payment on time to avoid any late fees.
                    Thank you.
                    """
                if row['predicted_class'] == 'partial_payment':
                    message = f"""Hello, {row['first_name']}
                    We noticed that you have not paid your last payment.
                    I am here to help you renegociate your payment.
                    Please let me know if you are interested in renegociating your payment.
                    Thank you.
                    """

                if message:
                    with st.spinner(f"Sending message to {row['first_name']}..."):
                        success = send_whatsapp_message('+'+str(int(row['phone_number'])), message)
                        
                        if success:
                            successful_messages += 1
                        else:
                            failed_messages += 1
                            
                        # Record message in history
                        st.session_state.message_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'recipient': row['first_name'],
                            'phone': row['phone_number'],
                            'prediction_class': row['predicted_class'],
                            'success': success
                        })
                                    
        # Display summary of batch operation
        st.info(f"Disparo de mensagem completo: {successful_messages} mensagens entregues, {failed_messages} mensagens falhas de um total de {len(df_to_send)} clientes.")

except Exception as e:
    st.error(f"Error processing data: {str(e)}")