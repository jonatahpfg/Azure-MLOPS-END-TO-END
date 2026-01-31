"""
===================================================================
SCRIPT DE TESTE DO ENDPOINT
===================================================================
Descrição: Testa o endpoint de inferência deployado
- Envia requisições de teste
- Valida respostas
- Mede latência

Uso:
    python test_endpoint.py --endpoint_url <url> --api_key <key>

===================================================================
"""

import os
import sys
import json
import logging
import time
import requests
import pandas as pd
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Testa endpoint de inferência")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Número de amostras de teste (padrão: 5)"
    )
    parser.add_argument(
        "--subscription_id",
        type=str,
        required=False,
        help="Azure Subscription ID (opcional, pode usar login local)"
    )
    parser.add_argument(
        "--resource_group",
        type=str,
        required=False,
        help="Azure Resource Group (opcional)"
    )
    parser.add_argument(
        "--workspace_name",
        type=str,
        required=False,
        help="Azure ML Workspace Name (opcional)"
    )
    return parser.parse_args()


def generate_sample_data(num_samples: int) -> dict:
    """Gera dados de teste reais para validar o preprocessor na nuvem."""
    logger.info(f"Gerando {num_samples} amostras de teste reais...")
    
    # Criamos uma lista de dicionários com as 19 colunas obrigatórias
    samples = []
    for _ in range(num_samples):
        samples.append({
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", 
            "Dependents": "No", "tenure": 1, "PhoneService": "No", 
            "MultipleLines": "No phone service", "InternetService": "DSL", 
            "OnlineSecurity": "No", "OnlineBackup": "Yes", 
            "DeviceProtection": "No", "TechSupport": "No", 
            "StreamingTV": "No", "StreamingMovies": "No", 
            "Contract": "Month-to-month", "PaperlessBilling": "Yes", 
            "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, 
            "TotalCharges": "29.85"
        })
    
    # O Azure ML espera os dados dentro de uma chave 'data'
    return {"data": samples}

def test_endpoint(endpoint_url: str, api_key: str, data: dict) -> dict:
    """
    Testa o endpoint com dados fornecidos.
    """
    try:
        logger.info(f"Enviando requisição para: {endpoint_url}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        start_time = time.time()
        response = requests.post(
            endpoint_url,
            json=data,
            headers=headers,
            timeout=30
        )
        latency = (time.time() - start_time) * 1000  # em ms
        response.raise_for_status()
        result = response.json()
        logger.info(f"✅ Resposta recebida (latência: {latency:.2f}ms)")
        return {
            "success": True,
            "latency_ms": latency,
            "status_code": response.status_code,
            "result": result
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erro na requisição: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def analyze_results(results: list):
    """
    Analisa resultados dos testes.
    
    Args:
        results: Lista de resultados
    """
    logger.info("\n" + "="*70)
    logger.info("ANÁLISE DOS RESULTADOS")
    logger.info("="*70)
    
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    
    logger.info(f"✅ Sucessos: {len(successes)}/{len(results)}")
    logger.info(f"❌ Falhas: {len(failures)}/{len(results)}")
    
    if successes:
        latencies = [r["latency_ms"] for r in successes]
        logger.info(f"\n📊 Latência:")
        logger.info(f"   - Mínima: {min(latencies):.2f}ms")
        logger.info(f"   - Máxima: {max(latencies):.2f}ms")
        logger.info(f"   - Média: {np.mean(latencies):.2f}ms")
        logger.info(f"   - P95: {np.percentile(latencies, 95):.2f}ms")
    
    logger.info("="*70)


def validate_predictions(result: dict):
    """
    Valida formato das predições.
    
    Args:
        result: Resultado do endpoint
    """
    try:
        if isinstance(result, list):
            for pred in result:
                assert "churn_prediction" in pred, "Falta 'churn_prediction'"
                assert "churn_probability" in pred, "Falta 'churn_probability'"
                
                # Valida valores
                assert pred["churn_prediction"] in [0, 1], "Predição deve ser 0 ou 1"
                assert 0 <= pred["churn_probability"] <= 1, "Probabilidade deve estar entre 0 e 1"
            
            logger.info("✅ Formato das predições validado")
        else:
            logger.warning(" Formato de resposta inesperado")
            
    except AssertionError as e:
        logger.error(f"❌ Validação falhou: {str(e)}")


def main():
    """Função principal."""
    args = parse_args()
    logger.info("="*70)
    logger.info("TESTE DO ENDPOINT DE INFERÊNCIA")
    logger.info("="*70)

    # Conecta ao Azure ML
    subscription_id = args.subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = args.resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = args.workspace_name or os.environ.get("AZURE_WORKSPACE_NAME")
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

    # Recupera nome do endpoint do ambiente
    endpoint_name = os.environ.get("DEPLOYED_ENDPOINT_NAME")
    if not endpoint_name:
        logger.error("Variável de ambiente DEPLOYED_ENDPOINT_NAME não definida!")
        return
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    scoring_uri = endpoint.scoring_uri
    keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
    api_key = keys.primary_key

    # Gera dados de teste
    test_data = generate_sample_data(args.num_samples)
    num_tests = 10
    logger.info(f"\nExecutando {num_tests} testes...")
    results = []
    for i in range(num_tests):
        logger.info(f"\n🧪 Teste {i+1}/{num_tests}")
        result = test_endpoint(scoring_uri, api_key, test_data)
        results.append(result)
        if i == 0 and result.get("success"):
            validate_predictions(result.get("result"))
        time.sleep(0.5)
    analyze_results(results)
    
    num_success = sum(1 for r in results if r.get("success"))
    if num_success < num_tests:
        logger.error(f"❌ Apenas {num_success}/{num_tests} testes foram bem-sucedidos. Interrompendo pipeline!")
        sys.exit(1)
    else:
        logger.info(f"\n✅ Todos os {num_tests} testes passaram! Endpoint está estável e pronto para produção.")


if __name__ == "__main__":
    main()
