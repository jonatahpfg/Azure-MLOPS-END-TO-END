import json
import logging
import os
import joblib
import pandas as pd
import numpy as np
import mlflow
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init():
    global model, preprocessor
    try:
        logger.info("Inicializando serviço de inferência...")
        model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
        
        # Carrega modelo via MLflow
        model_path = os.path.join(model_dir, "model")
        model = mlflow.sklearn.load_model(model_path)
        logger.info(f"✅ Modelo carregado: {type(model).__name__}")
        
        # Carrega preprocessor
        preprocessor_path = os.path.join(model_path, "preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info("✅ Preprocessor carregado")
        else:
            preprocessor = None
            logger.warning("⚠️ Preprocessor não encontrado")
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar: {str(e)}")
        raise

# Schema com as 19 features reais
input_sample = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"
}])

output_sample = np.array([{"customer_id": 0, "churn_prediction": 1, "churn_probability": 0.85, "risk_level": "CRITICAL"}])

def get_risk_level(prob):
    if prob < 0.3: return "LOW"
    if prob < 0.6: return "MEDIUM"
    if prob < 0.8: return "HIGH"
    return "CRITICAL"

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        # Garante que colunas numéricas estejam no tipo correto
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        # Remove nomes de colunas e envia direto para o pipeline
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "customer_id": i,
                "churn_prediction": int(pred),
                "churn_probability": round(float(prob), 4),
                "risk_level": get_risk_level(prob)
            })
        return results

    except Exception as e:
        logger.error(f"❌ Erro na inferência: {str(e)}")
        return {"error": str(e)}