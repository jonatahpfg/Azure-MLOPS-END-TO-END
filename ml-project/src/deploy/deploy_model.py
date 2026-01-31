def cleanup_endpoint(ml_client, endpoint_name):
    """
    Remove o endpoint se já existir para liberar cota de núcleos.
    """
    from azure.core.exceptions import ResourceNotFoundError
    try:
        logger.info(f"Verificando existência do endpoint '{endpoint_name}' para limpeza de cota...")
        ml_client.online_endpoints.get(name=endpoint_name)
        logger.info(f"Endpoint '{endpoint_name}' existe. Excluindo para liberar cota...")
        ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
        logger.info(f"Endpoint '{endpoint_name}' removido com sucesso.")
    except ResourceNotFoundError:
        logger.info(f"Endpoint '{endpoint_name}' não existe. Nenhuma limpeza necessária.")
    except Exception as e:
        logger.error(f"Erro ao tentar remover endpoint: {str(e)}")
        raise
"""
===================================================================
SCRIPT DE DEPLOY - MANAGED ONLINE ENDPOINT
===================================================================
Descrição: Deploy do modelo em Azure ML Managed Online Endpoint
- Busca última versão do modelo no Registry
- Cria/atualiza Managed Online Endpoint
- Configura deployment com auto-scaling
- Aloca 100% do tráfego

===================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings,
    OnlineRequestSettings
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

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
    parser = argparse.ArgumentParser(description="Deploy de modelo no Azure ML")
    
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
        "--model_name",
        type=str,
        default="TelcoChurnModel",
        help="Nome do modelo no Registry (padrão: TelcoChurnModel)"
    )
    
    parser.add_argument(
        "--endpoint_name",
        type=str,
        default="churn-prediction-endpoint",
        help="Nome do endpoint (padrão: churn-prediction-endpoint)"
    )
    
    parser.add_argument(
        "--deployment_name",
        type=str,
        default="churn-deployment-v1",
        help="Nome do deployment (padrão: churn-deployment-v1)"
    )
    
    parser.add_argument(
        "--instance_type",
        type=str,
        default="Standard_D2as_v4",
        help="Tipo de VM para deployment (padrão: Standard_D2as_v4)"
    )
    
    parser.add_argument(
        "--instance_count",
        type=int,
        default=1,
        help="Número de instâncias (padrão: 1)"
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


def get_latest_model_version(ml_client: MLClient, model_name: str) -> Model:
    """
    Busca o melhor modelo (campeão) baseado no maior composite_score entre todos os modelos TelcoChurnModel*.
    
    Args:
        ml_client: Cliente do Azure ML
        model_name: Nome base do modelo (ignorado, mantido para compatibilidade)
    Returns:
        Model object do campeão
    """
    try:
        logger.info("Buscando o melhor modelo entre todas as versões de TelcoChurnModel")
        all_models = ml_client.models.list(name=model_name)
        best_model = None
        best_score = -float('inf')
        if not all_models:
            logger.warning("Nenhum modelo TelcoChurnModel encontrado.")
        for model in all_models:
            tags = getattr(model, 'tags', {}) or {}
            logger.info(f"Modelo encontrado: {model.name} | Versão: {model.version} | Tags: {tags}")
            score_str = tags.get('composite_score')
            if score_str is None:
                continue
            try:
                score = float(score_str)
            except Exception:
                logger.warning(f"Tag composite_score inválida para modelo {model.name} v{model.version}: {score_str}")
                continue
            if score > best_score:
                best_score = score
                best_model = model
        if best_model is None:
            raise ValueError("Nenhum modelo TelcoChurnModel com composite_score encontrado no Registry.")
        logger.info(f"Campeão selecionado: {best_model.name} (versão {best_model.version}) com score {best_score:.4f}")
        logger.info(f"   - ID: {best_model.id}")
        logger.info(f"   - Tags: {best_model.tags}")
        return best_model
    except Exception as e:
        logger.error(f"Erro ao buscar modelo campeão: {str(e)}")
        raise


def create_or_update_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    """
    Cria ou atualiza Managed Online Endpoint.
    
    Args:
        ml_client: Cliente do Azure ML
        endpoint_name: Nome do endpoint
        
    Returns:
        ManagedOnlineEndpoint criado/atualizado
    """
    try:
        # Verifica se endpoint já existe
        try:
            existing_endpoint = ml_client.online_endpoints.get(name=endpoint_name)
            logger.info(f"Endpoint '{endpoint_name}' já existe. Atualizando...")
            return existing_endpoint
            
        except ResourceNotFoundError:
            logger.info(f"Criando novo endpoint: {endpoint_name}")
            
            # Cria novo endpoint
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description="Endpoint para previsão de Churn de clientes Telco",
                auth_mode="key",  # Autenticação por chave API
                tags={
                    "project": "telco-churn",
                    "environment": "production",
                    "model_type": "classification"
                }
            )
            
            # Cria endpoint
            endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            
            logger.info(f"✅ Endpoint criado: {endpoint_name}")
            logger.info(f"   - Scoring URI: {endpoint_result.scoring_uri}")
            
            return endpoint_result
        
    except Exception as e:
        logger.error(f"Erro ao criar/atualizar endpoint: {str(e)}")
        raise


def create_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model: Model,
    instance_type: str,
    instance_count: int
) -> ManagedOnlineDeployment:
    """
    Cria deployment do modelo no endpoint com ambiente customizado.
    """
    try:
        logger.info(f"Criando ambiente pré-construído para o deployment: {deployment_name}")
        env_name = f"env-{deployment_name}"

    # --- LÓGICA DE CARREGAMENTO FORÇADO ---
        from pathlib import Path
        
        # GPS: src/deploy -> src -> ml-project -> environments/conda.yml
        script_path = Path(__file__).resolve()
        conda_path = script_path.parent.parent.parent / "environments" / "conda.yml"
        
        logger.info(f" Verificando arquivo em: {conda_path}")

        if not conda_path.exists():
            raise FileNotFoundError(f"❌ O arquivo não existe no caminho: {conda_path}")

        # TESTE DE LEITURA: Se falhar aqui, o log do GitHub avisará na hora!
        with open(conda_path, "r") as f:
            conda_content = f.read()
            if len(conda_content.strip()) < 10:
                raise ValueError(f"❌ ERRO CRÍTICO: O arquivo {conda_path} foi lido, mas está VAZIO!")
        
        logger.info(f"✅ Arquivo lido com sucesso ({len(conda_content)} caracteres).")

        # --- CRIAÇÃO DO AMBIENTE ---
    
        env = Environment(
            name=env_name,
            image="mcr.microsoft.com/azureml/inference-base-2204:latest",
            version="6", 
            conda_file=str(conda_path.absolute()) 
        )
        
        logger.info(f"🚀 Enviando ambiente v{env.version} para a Azure...")
        env = ml_client.environments.create_or_update(env)

        # --- CONFIGURAÇÃO DO CÓDIGO ---
        # Como o score.py está na mesma pasta deste script (src/deploy), o caminho é o diretório pai do arquivo atual
        code_dir = Path(__file__).parent.resolve()
        logger.info(f"📂 Pacote de código localizado em: {code_dir}")

        logger.info(f"Criando deployment: {deployment_name}")
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            environment=env,
            code_configuration=CodeConfiguration(
                code=str(code_dir.absolute()),
                scoring_script="score.py" 
            ),
            instance_type=instance_type,
            instance_count=instance_count,
            liveness_probe=ProbeSettings(
                initial_delay=120,
                period=10,
                timeout=2,
                success_threshold=1,
                failure_threshold=3
            ),
            readiness_probe=ProbeSettings(
                initial_delay=120,
                period=10,
                timeout=2,
                success_threshold=1,
                failure_threshold=3
            ),
            request_settings=OnlineRequestSettings(
                request_timeout_ms=60000,
                max_concurrent_requests_per_instance=1
            )
        )
        logger.info("Iniciando deployment (pode levar alguns minutos)...")
        deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
        logger.info(f"✅ Deployment criado: {deployment_name}")
        return deployment_result
    except Exception as e:
        logger.error(f"Erro ao criar deployment: {str(e)}")
        raise


def allocate_traffic(ml_client: MLClient, endpoint_name: str, deployment_name: str) -> None:
    """
    Aloca 100% do tráfego para o deployment.
    
    Args:
        ml_client: Cliente do Azure ML
        endpoint_name: Nome do endpoint
        deployment_name: Nome do deployment
    """
    try:
        logger.info(f"Alocando 100% do tráfego para: {deployment_name}")
        
        # Busca endpoint
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        
        # Aloca tráfego
        endpoint.traffic = {deployment_name: 100}
        
        # Atualiza endpoint
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"✅ Tráfego alocado: {deployment_name} = 100%")
        
    except Exception as e:
        logger.error(f"Erro ao alocar tráfego: {str(e)}")
        raise


def get_endpoint_details(ml_client: MLClient, endpoint_name: str) -> None:
    """
    Exibe detalhes do endpoint para o usuário.
    
    Args:
        ml_client: Cliente do Azure ML
        endpoint_name: Nome do endpoint
    """
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        
        logger.info("\n" + "="*70)
        logger.info("DETALHES DO ENDPOINT")
        logger.info("="*70)
        logger.info(f"Nome: {endpoint.name}")
        logger.info(f"Scoring URI: {endpoint.scoring_uri}")
        logger.info(f"Swagger URI: {endpoint.openapi_uri}")
        logger.info(f"Estado: {endpoint.provisioning_state}")
        logger.info(f"Autenticação: {endpoint.auth_mode}")
        
        # Busca chaves de autenticação
        try:
            keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
            logger.info(f"\n CHAVE PRIMÁRIA: {keys.primary_key}")
            logger.info(f" CHAVE SECUNDÁRIA: {keys.secondary_key}")
        except:
            logger.warning("Não foi possível recuperar chaves de autenticação")
        
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Erro ao obter detalhes: {str(e)}")


def main():
    """
    Função principal de execução.
    """
    try:
        # Parse argumentos
        args = parse_args()
        
        logger.info("="*70)
        logger.info("INICIANDO DEPLOY DO MODELO")
        logger.info("="*70)
        
        # 1. Cria cliente do Azure ML
        ml_client = get_ml_client(
            args.subscription_id,
            args.resource_group,
            args.workspace_name
        )

        # Limpa endpoint existente para liberar cota
        cleanup_endpoint(ml_client, args.endpoint_name)
        time.sleep(60)  # Aguarda liberação de cota na Azure

        # Busca última versão do modelo
        model = get_latest_model_version(ml_client, args.model_name)

        # Cria endpoint (novo)
        endpoint = create_or_update_endpoint(ml_client, args.endpoint_name)

        # Cria deployment com ambiente pré-construído
        deployment = create_deployment(
            ml_client,
            args.endpoint_name,
            args.deployment_name,
            model,
            args.instance_type,
            args.instance_count
        )

        # Aloca tráfego
        allocate_traffic(ml_client, args.endpoint_name, args.deployment_name)

        # Exibe detalhes
        get_endpoint_details(ml_client, args.endpoint_name)

        # Exporta o nome do endpoint para o GitHub Actions, se GITHUB_OUTPUT estiver definido
        github_output = os.environ.get('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f"endpoint_name={args.endpoint_name}\n")
            logger.info(f"Nome do endpoint exportado para GITHUB_OUTPUT: {args.endpoint_name}")

        logger.info("\n" + "="*70)
        logger.info("✅ DEPLOY CONCLUÍDO COM SUCESSO!")
        logger.info("="*70)

        logger.info("\n📌 PRÓXIMOS PASSOS:")
        logger.info("1. Guarde as chaves de API no Azure Key Vault")
        logger.info( "2. Configure monitoramento com Application Insights")
        logger.info("3. Teste o endpoint com dados reais")
        logger.info("4. Configure alertas para latência e erros")
        logger.info("5. Implemente blue-green deployment para próximas versões")

    except Exception as e:
        logger.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
