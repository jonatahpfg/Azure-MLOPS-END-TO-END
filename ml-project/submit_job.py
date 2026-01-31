"""
===================================================================
SCRIPT DE SUBMISSÃO DO PIPELINE
===================================================================
Descrição: Submete o pipeline de Churn ao Azure ML
- Conecta ao workspace
- Registra dados no Data Lake
- Cria ambiente
- Submete pipeline job
- Monitora execução

Uso:
    python submit_job.py --subscription_id <id> \
                         --resource_group <rg> \
                         --workspace_name <ws> \
                         --model_type XGBoost
===================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import Data, Environment
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from pipelines.churn_pipeline import create_pipeline_job

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parseia argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(description="Submissão de Pipeline no Azure ML")
    
    parser.add_argument(
        "--subscription_id",
        type=str,
        required=True,
        help="Azure Subscription ID"
    )
    
    parser.add_argument(
        "--resource_group",
        type=str,
        required=True,
        help="Azure Resource Group"
    )
    
    parser.add_argument(
        "--workspace_name",
        type=str,
        required=True,
        help="Azure ML Workspace Name"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="XGBoost",
        choices=['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression', 'all'],
        help="Tipo de modelo a ser treinado (padrão: XGBoost, ou 'all' para todos)"
    )
    
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="azureml://datastores/datalakedados/paths/bronze/Telco_Customer_Churn.csv",
        help="Caminho para os dados brutos no Data Lake (padrão: azureml://datastores/datalakedados/paths/bronze/Telco_Customer_Churn.csv)"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="telco-churn-experiment",
        help="Nome do experimento (padrão: telco-churn-experiment)"
    )
    
    parser.add_argument(
        "--wait_for_completion",
        type=str,
        default="true",
        choices=['true', 'false'],
        help="Aguardar conclusão do pipeline (padrão: true)"
    )
    
    return parser.parse_args()


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str) -> MLClient:
    """
    Cria cliente do Azure ML.
    
    Args:
        subscription_id: ID da subscription
        resource_group: Nome do resource group
        workspace_name: Nome do workspace
        
    Returns:
        MLClient configurado
    """
    try:
        logger.info("Autenticando com Azure...")
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        logger.info(f"✅ Cliente conectado ao workspace: {workspace_name}")
        return ml_client
        
    except Exception as e:
        logger.error(f"Erro ao criar MLClient: {str(e)}")
        raise


def register_data_asset(ml_client: MLClient, data_path: str) -> str:
    """
    Registra dados no Azure ML como Data Asset.
    
    Args:
        ml_client: Cliente do Azure ML
        data_path: Caminho dos dados (pode ser local ou URI do Data Lake)
        
    Returns:
        URI do data asset registrado
    """
    try:
        logger.info(f"Registrando dados de: {data_path}")
        
        # Cria Data Asset
        data_asset = Data(
            name="telco_churn_raw",
            description="Telco Customer Churn - Dados Brutos (Bronze Layer)",
            path=data_path,
            type=AssetTypes.URI_FILE,
            tags={
                "layer": "bronze",
                "source": "kaggle",
                "format": "csv"
            }
        )
        
        # Registra no workspace
        registered_data = ml_client.data.create_or_update(data_asset)
        
        logger.info(f"✅ Dados registrados: {registered_data.name}:{registered_data.version}")
        logger.info(f"   URI: {registered_data.path}")
        
        return registered_data.id
        
    except Exception as e:
        logger.error(f"Erro ao registrar dados: {str(e)}")
        raise


def create_or_get_environment(ml_client: MLClient) -> str:
    """
    Cria ou recupera ambiente do Azure ML.
    
    Args:
        ml_client: Cliente do Azure ML
        
    Returns:
        String de referência ao ambiente
    """
    import hashlib
    try:
        # Gera hash SHA-256 dos primeiros 8 caracteres do conda.yml
        with open("./environments/conda.yml", "rb") as f:
            conda_bytes = f.read()
        conda_hash = hashlib.sha256(conda_bytes).hexdigest()[:8]
        env_name = f"prod-churn-env--{conda_hash}"
        logger.info(f"Verificando existência do ambiente: {env_name}")
        try:
            env = ml_client.environments.get(name=env_name, label="latest")
            logger.info(f" Ambiente já existe: {env_name}:{env.version}")
            return f"azureml:{env_name}:{env.version}"
        except Exception:
            logger.info(f"Ambiente não encontrado, criando novo: {env_name}")
            env = Environment(
                name=env_name,
                description="Ambiente para pipeline de Churn Prediction com MLflow",
                conda_file="./environments/conda.yml",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
                tags={
                    "project": "telco-churn",
                    "framework": "sklearn-xgboost-lightgbm",
                    "conda_hash": conda_hash
                }
            )
            created_env = ml_client.environments.create_or_update(env)
            logger.info(f" Ambiente criado: {env_name}:{created_env.version}")
            return f"azureml:{env_name}:{created_env.version}"
    except Exception as e:
        logger.error(f"Erro ao criar/recuperar ambiente: {str(e)}")
        raise

def submit_pipeline(ml_client, data_asset_id, model_type, experiment_name, environment):
    import os
    from datetime import datetime
    try:
        logger.info(f"Criando pipeline (Modo Segurança)")

        # Nomeação dinâmica
        github_run_id = os.environ.get("GITHUB_RUN_ID")
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_suffix = github_run_id if github_run_id else timestamp
        dynamic_experiment_name = f"{experiment_name}-{unique_suffix}"
        dynamic_display_name = f"Churn Pipeline - {model_type} - {unique_suffix}"

        # 1. PASSE TUDO DENTRO DA CHAMADA (Inclusive o model_type)
        pipeline_job = create_pipeline_job(
            raw_data_uri=Input(type=AssetTypes.URI_FILE, path=data_asset_id),
            model_type=model_type, # <-- Passe aqui diretamente
            environment=environment,
            config_path=Input(type=AssetTypes.URI_FILE, path="./config/grid_search.yml")
        )
        
        pipeline_job.experiment_name = dynamic_experiment_name
        pipeline_job.display_name = dynamic_display_name
        pipeline_job.description = f"Pipeline completo de Churn com {model_type} (Medallion Architecture - Modo Segurança)"
        logger.info("Submetendo pipeline job ao Azure ML...")
        logger.info("="*70)
        logger.info("CONFIGURAÇÕES DO PIPELINE:")
        logger.info(f"   - Modelo: {model_type}")
        logger.info(f"   - Experimento: {dynamic_experiment_name}")
        logger.info(f"   - Compute: clustertreino")
        logger.info(f"   - Outputs: Gerenciados automaticamente pelo Azure")
        logger.info("="*70)
        submitted_job = ml_client.jobs.create_or_update(
            pipeline_job,
            experiment_name=dynamic_experiment_name
        )
        logger.info(f"\n Pipeline submetido com sucesso!")
        logger.info(f"   - Job Name: {submitted_job.name}")
        logger.info(f"   - Job ID: {submitted_job.id}")
        logger.info(f"   - Status: {submitted_job.status}")
        logger.info(f"   - Studio URL: {submitted_job.studio_url}")
        return submitted_job
    except Exception as e:
        logger.error(f"Erro ao submeter pipeline: {str(e)}")
        raise


def monitor_job(ml_client: MLClient, job):
    """
    Monitora execução do pipeline job.
    
    Args:
        ml_client: Cliente do Azure ML
        job: Job submetido
    """
    try:
        logger.info("\n Monitorando execução do pipeline...")
        logger.info("   (Pressione Ctrl+C para parar o monitoramento)")
        logger.info("="*70)
        # Stream logs
        ml_client.jobs.stream(job.name)
        # Busca job atualizado
        final_job = ml_client.jobs.get(job.name)
        logger.info("\n" + "="*70)
        logger.info(f"Status final do pipeline: {final_job.status}")
        logger.info(f" Azure ML Studio: {final_job.studio_url}")
        logger.info("="*70)
        if final_job.status == "Completed":
            logger.info(" PIPELINE CONCLUÍDO COM SUCESSO!")
            logger.info("="*70)
            logger.info("\n PRÓXIMOS PASSOS:")
            logger.info("1. Verifique as métricas no Azure ML Studio")
            logger.info("2. Analise os dados da camada Gold no Data Lake")
            logger.info("3. Revise o modelo registrado no Model Registry")
            logger.info("4. Se o modelo passou no gate de qualidade, execute o deploy:")
            logger.info("   python src/deploy/deploy_model.py --subscription_id <id> ...")
            return True
        else:
            logger.error(f"❌ Pipeline falhou com status: {final_job.status}")
            logger.error("   Verifique os logs no Azure ML Studio para detalhes")
            return False
    except KeyboardInterrupt:
        logger.info("\n\n Monitoramento interrompido pelo usuário")
        logger.info(f"   O pipeline continua executando em segundo plano")
        logger.info(f"   Acompanhe em: {job.studio_url}")
        return False
    except Exception as e:
        logger.error(f"Erro ao monitorar job: {str(e)}")
        return False


def main():
    """
    Função principal de execução.
    """
    try:
        # Parse argumentos
        args = parse_args()
        
        logger.info("="*70)
        logger.info("SUBMISSÃO DO PIPELINE DE CHURN PREDICTION")
        logger.info("="*70)
        
        # 1. Cria cliente do Azure ML
        ml_client = get_ml_client(
            args.subscription_id,
            args.resource_group,
            args.workspace_name
        )
        
        # 2. Cria/recupera ambiente
        environment = create_or_get_environment(ml_client)
        
        # 3. Registra dados no Data Lake
        data_asset_id = register_data_asset(ml_client, args.raw_data_path)
        
        # 4. Submete pipeline
        job = submit_pipeline(
        ml_client,
        data_asset_id,
        args.model_type,
        args.experiment_name,
        environment  
        )
            
        # 5. Monitora execução (se solicitado)
        success = True
        if args.wait_for_completion.lower() == 'true':
            success = monitor_job(ml_client, job)
        else:
            logger.info("\n📌 Pipeline submetido em modo assíncrono")
            logger.info(f"   Acompanhe em: {job.studio_url}")
        if not success:
            logger.error("❌ O pipeline não foi concluído com sucesso. Abortando processo.")
            sys.exit(1)
        logger.info("\n✅ PROCESSO CONCLUÍDO!")
    except Exception as e:
        logger.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
