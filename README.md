# Sistema de Previsão de Pagamentos e Gestão de Cobranças

## Introdução

Este projeto consiste em uma solução completa para previsão de comportamento de pagamento e gestão de cobranças, desenvolvida para auxiliar empresas a otimizar seu processo de recuperação de crédito. O sistema utiliza técnicas avançadas de machine learning para classificar clientes de acordo com sua probabilidade de pagamento, permitindo uma abordagem mais estratégica e personalizada nas ações de cobrança.

### Principais Funcionalidades

- Previsão do comportamento de pagamento dos clientes através de modelo XGBoost
- Classificação dos clientes em diferentes perfis de pagamento
- Interface web interativa desenvolvida em Streamlit para visualização e gestão
- Sistema automatizado de envio de mensagens de cobrança via WhatsApp
- Acompanhamento em tempo real das métricas de sucesso das cobranças

### Tecnologias Utilizadas

O projeto foi desenvolvido utilizando as seguintes tecnologias:

- Python 3.13
- Streamlit para interface web
- Pandas e NumPy para manipulação de dados
- Scikit-learn e XGBoost para machine learning
- Matplotlib, Seaborn e Plotly para visualizações
- Integração com API do WhatsApp para envio de mensagens

Esta solução visa aumentar a eficiência do processo de cobrança, reduzir custos operacionais e melhorar a taxa de recuperação de crédito através de uma abordagem data-driven e automatizada.

## Desenvolvimento do Modelo

### Pré-processamento dos Dados

O desenvolvimento do modelo de previsão seguiu as seguintes etapas de pré-processamento:

1. **Limpeza dos Dados**
   - Remoção de valores nulos e duplicados
   - Remoção de colunas com mais de 40% de valores faltantes
   - Padronização de formatos de datas e valores monetários

2. **Engenharia de Features**
   - Criação de features temporais (aparições anteriores de um cliente no histórico de atrasos)
   - Imputação de uma flag MISSING para variáveis categóricas faltantes
   - Codificação de variáveis categóricas usando one-hot encoding
   - Imputação da mediana para variáveis numéricas
   - Normalização de variáveis numéricas usando o standardscaler
   - Transformação da variável alvo em flag (full_payment, partial_payment e no_payment)

3. **Seleção de Features**
   - Análise de correlação entre variáveis numérica
   [[image.png]]
   - Análise de diferença de grupos para variáveis categóricas

### Processo de Modelagem

O processo de desenvolvimento do modelo seguiu as seguintes etapas:

1. **Divisão dos Dados**
   - Separação em conjuntos de treino e oot.
   - Separação dos dados de treino em treino e teste.
   - Utilização de gridsearch para otimizar parametros testando modelos XGBoostClassifier, RandomForestClassifier e Regressão Logística
   - Estratificação para manter a distribuição das classes

2. **Treinamento do Modelo**
   - Utilização do algoritmo XGBoost para classificação
   - Otimização de hiperparâmetros via validação cruzada
   - Implementação de early stopping para evitar overfitting

3. **Avaliação do Modelo**
   - Métricas principais: precisão, recall e F1-score
   - Análise da matriz de confusão
   - Validação cruzada para robustez dos resultados
   - Monitoramento de métricas específicas do negócio

4. **Interpretabilidade**
   - Análise de importância das features
   - Geração de SHAP values para entender decisões do modelo
   - Documentação das regras de negócio aprendidas


