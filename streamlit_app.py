import streamlit as st
import requests
import pandas as pd
import os


def main():
    st.title('Predição de Churn de Clientes Telco')
    st.write('Preencha os dados do cliente para prever o churn usando o modelo em produção.')

    # Mapeamento para exibir labels em português
    labels = {
        "gender": "Gênero",
        "SeniorCitizen": "Idoso? (0=Não, 1=Sim)",
        "Partner": "Possui Parceiro(a)?",
        "Dependents": "Possui Dependentes?",
        "tenure": "Meses de Contrato",
        "PhoneService": "Serviço de Telefone",
        "MultipleLines": "Múltiplas Linhas",
        "InternetService": "Tipo de Internet",
        "OnlineSecurity": "Segurança Online",
        "OnlineBackup": "Backup Online",
        "DeviceProtection": "Proteção de Dispositivo",
        "TechSupport": "Suporte Técnico",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Filmes",
        "Contract": "Tipo de Contrato",
        "PaperlessBilling": "Fatura Digital",
        "PaymentMethod": "Método de Pagamento",
        "MonthlyCharges": "Cobrança Mensal",
        "TotalCharges": "Cobrança Total"
    }

    with st.form(key='churn_form'):
        gender = st.selectbox(labels["gender"], ['Masculino', 'Feminino'])
        SeniorCitizen = st.selectbox(labels["SeniorCitizen"], [0, 1])
        Partner = st.selectbox(labels["Partner"], ['Sim', 'Não'])
        Dependents = st.selectbox(labels["Dependents"], ['Sim', 'Não'])
        tenure = st.number_input(labels["tenure"], min_value=0, max_value=100, value=1)
        PhoneService = st.selectbox(labels["PhoneService"], ['Sim', 'Não'])
        MultipleLines = st.selectbox(labels["MultipleLines"], ['Sim', 'Não', 'Sem serviço de telefone'])
        InternetService = st.selectbox(labels["InternetService"], ['DSL', 'Fibra óptica', 'Nenhum'])
        OnlineSecurity = st.selectbox(labels["OnlineSecurity"], ['Sim', 'Não', 'Sem internet'])
        OnlineBackup = st.selectbox(labels["OnlineBackup"], ['Sim', 'Não', 'Sem internet'])
        DeviceProtection = st.selectbox(labels["DeviceProtection"], ['Sim', 'Não', 'Sem internet'])
        TechSupport = st.selectbox(labels["TechSupport"], ['Sim', 'Não', 'Sem internet'])
        StreamingTV = st.selectbox(labels["StreamingTV"], ['Sim', 'Não', 'Sem internet'])
        StreamingMovies = st.selectbox(labels["StreamingMovies"], ['Sim', 'Não', 'Sem internet'])
        Contract = st.selectbox(labels["Contract"], ['Mensal', 'Um ano', 'Dois anos'])
        PaperlessBilling = st.selectbox(labels["PaperlessBilling"], ['Sim', 'Não'])
        PaymentMethod = st.selectbox(labels["PaymentMethod"], [
            'Cheque eletrônico', 'Cheque enviado', 'Transferência bancária (automática)', 'Cartão de crédito (automático)'])
        MonthlyCharges = st.number_input(labels["MonthlyCharges"], min_value=0.0, value=0.0)
        TotalCharges = st.number_input(labels["TotalCharges"], min_value=0.0, value=0.0)
        submit = st.form_submit_button('Prever Churn')

    if submit:
        # Mapeia valores para o formato esperado pelo modelo
        data = {
            "gender": 'Male' if gender == 'Masculino' else 'Female',
            "SeniorCitizen": SeniorCitizen,
            "Partner": 'Yes' if Partner == 'Sim' else 'No',
            "Dependents": 'Yes' if Dependents == 'Sim' else 'No',
            "tenure": tenure,
            "PhoneService": 'Yes' if PhoneService == 'Sim' else 'No',
            "MultipleLines": (
                'Yes' if MultipleLines == 'Sim' else ('No' if MultipleLines == 'Não' else 'No phone service')
            ),
            "InternetService": (
                'DSL' if InternetService == 'DSL' else ('Fiber optic' if InternetService == 'Fibra óptica' else 'No')
            ),
            "OnlineSecurity": (
                'Yes' if OnlineSecurity == 'Sim' else ('No' if OnlineSecurity == 'Não' else 'No internet service')
            ),
            "OnlineBackup": (
                'Yes' if OnlineBackup == 'Sim' else ('No' if OnlineBackup == 'Não' else 'No internet service')
            ),
            "DeviceProtection": (
                'Yes' if DeviceProtection == 'Sim' else ('No' if DeviceProtection == 'Não' else 'No internet service')
            ),
            "TechSupport": (
                'Yes' if TechSupport == 'Sim' else ('No' if TechSupport == 'Não' else 'No internet service')
            ),
            "StreamingTV": (
                'Yes' if StreamingTV == 'Sim' else ('No' if StreamingTV == 'Não' else 'No internet service')
            ),
            "StreamingMovies": (
                'Yes' if StreamingMovies == 'Sim' else ('No' if StreamingMovies == 'Não' else 'No internet service')
            ),
            "Contract": (
                'Month-to-month' if Contract == 'Mensal' else ('One year' if Contract == 'Um ano' else 'Two year')
            ),
            "PaperlessBilling": 'Yes' if PaperlessBilling == 'Sim' else 'No',
            "PaymentMethod": (
                'Electronic check' if PaymentMethod == 'Cheque eletrônico' else
                'Mailed check' if PaymentMethod == 'Cheque enviado' else
                'Bank transfer (automatic)' if PaymentMethod == 'Transferência bancária (automática)' else
                'Credit card (automatic)'
            ),
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }
        df = pd.DataFrame([data])
        payload = {"input_data": {"columns": list(df.columns), "data": df.values.tolist()}}
        st.write('Payload enviado ao endpoint:')
        st.json(payload)

        endpoint_url = st.secrets.get("ENDPOINT_URL") or os.environ.get("ENDPOINT_URL")
        api_key = st.secrets.get("API_KEY") or os.environ.get("API_KEY")
        if not endpoint_url or not api_key:
            st.error("Configure ENDPOINT_URL e API_KEY em .streamlit/secrets.toml ou variáveis de ambiente.")
            return
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Resultado do modelo: {result}")
            else:
                st.error(f"Erro na requisição: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erro ao conectar ao endpoint: {e}")

if __name__ == "__main__":
    main()
