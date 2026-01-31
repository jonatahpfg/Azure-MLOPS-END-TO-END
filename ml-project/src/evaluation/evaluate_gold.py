"""
===================================================================
EVALUATION & GOLD LAYER - SELEÇÃO DO MELHOR MODELO
===================================================================
Descrição: 
- Avalia modelo no conjunto de teste
- Compara com modelos anteriores registrados
- Registra APENAS se for o MELHOR até agora
- Cria camada Gold (tabela analítica)

===================================================================
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Avaliação e Gold Layer")
    
    parser.add_argument("--input_test", type=str, required=True,
                       help="Caminho para dados de teste (Silver)")
    parser.add_argument("--input_model", type=str, required=True,
                       help="Caminho para modelo treinado")
    parser.add_argument("--output_gold", type=str, required=True,
                       help="Caminho para salvar Gold layer")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Nome do modelo no Registry")
    
    return parser.parse_args()


def load_test_data(test_path: str) -> tuple:
    """
    Carrega dados de teste.
    
    Args:
        test_path: Caminho para test.parquet
        
    Returns:
        X_test, y_test
    """
    print(" [1/7] Carregando dados de teste...")
    
    # Azure ML monta como diretório, procura arquivo .parquet
    if os.path.isdir(test_path):
        parquet_files = list(Path(test_path).glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Nenhum arquivo .parquet encontrado em {test_path}")
        test_path = str(parquet_files[0])
    
    df_test = pd.read_parquet(test_path)
    print(f"   ✅ {len(df_test)} amostras carregadas")
    
    X_test = df_test.drop('Churn', axis=1)
    y_test = df_test['Churn']
    
    return X_test, y_test


def load_model(model_path: str):
    """
    Carrega modelo treinado.
    
    Args:
        model_path: Caminho para modelo .joblib
        
    Returns:
        Modelo carregado
    """
    print(" [2/7] Carregando modelos treinados...")
    # Validação robusta do caminho
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERRO STREAMING] O caminho do modelo não existe: {model_path}")
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"[ERRO STREAMING] O caminho do modelo não é um diretório: {model_path}")
    joblib_files = list(Path(model_path).rglob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(f"[ERRO STREAMING] Nenhum arquivo .joblib encontrado de forma recursiva em {model_path}. Verifique se o output do treino está correto e se não houve erro de montagem de volume.")
    print(f"   {len(joblib_files)} arquivos encontrados: {[Path(f).name for f in joblib_files]}")
    models = []
    for file in joblib_files:
        try:
            model = joblib.load(file)
            models.append((str(file), model))
        except Exception as e:
            print(f"[ERRO STREAMING] Falha ao carregar {file}: {e}")
    if not models:
        raise RuntimeError("[ERRO STREAMING] Nenhum modelo pôde ser carregado. Verifique se os arquivos .joblib estão íntegros e acessíveis.")
    return models


def calculate_metrics(y_test, y_pred, y_proba, model_name) -> dict:
    """
    Calcula todas as métricas de avaliação.
    
    Args:
        y_test: Labels verdadeiros
        y_pred: Predições (classe)
        y_proba: Probabilidades
        
    Returns:
        Dicionário com métricas
    """
    print(" [3/7] Calculando métricas...")
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    # Score composto para comparação: F1 (50%) + Recall (30%) + ROC-AUC (20%)
    metrics["composite_score"] = (
        metrics["f1_score"] * 0.5 +
        metrics["recall"] * 0.3 +
        metrics["roc_auc"] * 0.2
    )
    
    print(f"\n📈 MÉTRICAS NO TESTE:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")


    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)
    # Salvar e logar no MLflow
    cm_path = f"confusion_matrix_{model_name}.json"
    with open(cm_path, "w") as f:
        json.dump(cm.tolist(), f)
    try:
        import mlflow
        mlflow.log_artifact(cm_path)
    except Exception as e:
        print(f"[WARN] Não foi possível logar confusion_matrix no MLflow: {e}")

    return metrics


def get_best_model_score(ml_client: MLClient, model_name: str) -> float:
    """
                    model_name=model_name,
    
    Args:
        ml_client: Cliente do Azure ML
        model_name: Nome base do modelo
        
    Returns:
        Melhor score composto (ou 0.0 se nenhum modelo registrado)
    """
    print(" [4/7] Buscando modelos registrados anteriormente...")
    
    try:
        all_models = ml_client.models.list()
        
        best_score = 0.0
        best_model_info = None
        
        for model in all_models:
            if model.name.startswith("TelcoChurnModel"):
                score = float(model.tags.get("composite_score", 0.0))
                
                if score > best_score:
                    best_score = score
                    best_model_info = {
                        "name": model.name,
                        "version": model.version,
                        "score": score
                    }
        
        if best_model_info:
            print(f"   Melhor modelo atual:")
            print(f"      Nome: {best_model_info['name']}")
            print(f"      Versão: {best_model_info['version']}")
            print(f"      Score Composto: {best_model_info['score']:.4f}")
            return best_score
        else:
            print("   ℹ Nenhum modelo registrado anteriormente")
            return 0.0
            
    except Exception as e:
        print(f"   ⚠️ Erro ao buscar modelos: {str(e)}")
        return 0.0


def should_register_model(current_score: float, best_previous_score: float) -> bool:
    """
    Decide se o modelo atual deve ser registrado.
    
    Args:
        current_score: Score composto do modelo atual
        best_previous_score: Melhor score entre modelos anteriores
        
    Returns:
        True se deve registrar, False caso contrário
    """
    print(f"\n [5/7] Decisão de registro:")
    print(f"   Score atual:      {current_score:.4f}")
    print(f"   Melhor anterior:  {best_previous_score:.4f}")
    
    if current_score > best_previous_score:
        improvement = ((current_score - best_previous_score) / best_previous_score * 100) if best_previous_score > 0 else 100
        print(f"   ✅ MODELO APROVADO! Melhoria de {improvement:.2f}%")
        return True
    else:
        decline = ((best_previous_score - current_score) / best_previous_score * 100)
        print(f"   ❌ MODELO REJEITADO! Piora de {decline:.2f}%")
        return False


def register_model_to_registry(
    ml_client: MLClient,
    model,
    model_name: str,
    metrics: dict,
    model_path: str
):
    """
    Registra modelo no Azure ML Model Registry.
    
    Args:
        ml_client: Cliente do Azure ML
        model: Modelo treinado
        model_name: Nome para registro
        metrics: Dicionário com métricas
        model_path: Caminho onde modelo foi salvo
    """
    print(" [6/7] Registrando modelo no Model Registry...")
    
    try:
        # Salvar modelo localmente primeiro
        local_model_path = "best_model.joblib"
        joblib.dump(model, local_model_path)

        # Criar tags com métricas
        tags = {
            "status": "champion",
            "accuracy": f"{metrics['accuracy']:.4f}",
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "f1_score": f"{metrics['f1_score']:.4f}",
            "roc_auc": f"{metrics['roc_auc']:.4f}",
            "composite_score": f"{metrics['composite_score']:.4f}",
            "training_date": datetime.now().isoformat(),
            "framework": "scikit-learn"
        }

        # Registrar via MLflow (integrado com Azure ML)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

        # Recuperar o modelo registrado do tipo MLFLOW
        registered_model = ml_client.models.get(name=model_name, label="latest")
        registered_model.tags = tags
        registered_model.description = f"Modelo de Churn com score composto de {metrics['composite_score']:.4f}"
        ml_client.models.create_or_update(registered_model)

        print(f"   ✅ Modelo registrado: {model_name}")
        print(f"   🏆 Status: CHAMPION")
        print(f"   📊 Score Composto: {metrics['composite_score']:.4f}")

    except Exception as e:
        print(f"  Erro ao registrar modelo: {str(e)}")
        if "AuthorizationFailed" in str(e):
            print("   [RBAC] Falha de permissão ao registrar modelo. Verifique se o usuário possui a role 'AzureML Data Scientist' no portal Azure.")
        raise


def create_gold_layer(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_path: str,
    metrics: dict,
    model_scores: list = None
):
    """
    Cria camada Gold (tabela analítica para BI).
    
    Args:
        X_test: Features de teste
        y_test: Labels verdadeiros
        y_pred: Predições
        y_proba: Probabilidades
        output_path: Caminho para salvar Gold layer
        metrics: Métricas do modelo
    """
    print(" [7/7] Criando camada GOLD (Analytical Layer)...")
    
    # Criar DataFrame analítico
    df_gold = X_test.copy()
    df_gold['True_Churn'] = y_test.values
    df_gold['Predicted_Churn'] = y_pred
    df_gold['Churn_Probability'] = y_proba
    
    # Segmentação de risco
    df_gold['Risk_Segment'] = pd.cut(
        df_gold['Churn_Probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Adicionar flag de predição correta
    df_gold['Prediction_Correct'] = (df_gold['True_Churn'] == df_gold['Predicted_Churn']).astype(int)
    
    # Adicionar timestamp
    df_gold['evaluation_date'] = datetime.now().isoformat()
    
    # Adicionar métricas gerais como colunas
    df_gold['model_accuracy'] = metrics['accuracy']
    df_gold['model_f1_score'] = metrics['f1_score']
    df_gold['model_composite_score'] = metrics['composite_score']
    
    # Salvar
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "churn_analytical_gold.parquet")
    df_gold.to_parquet(output_file, index=False)

    print(f"    Gold layer salva em: {output_file}")
    print(f"    {len(df_gold)} registros")
    print(f"\n    Distribuição de Risco:")
    print(df_gold['Risk_Segment'].value_counts())

    # Salvar também métricas em JSON
    metrics_file = os.path.join(output_path, "model_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Métricas salvas em: {metrics_file}")

    # Salvar tabela comparativa de scores dos modelos
    if model_scores is not None:
        scores_df = pd.DataFrame(model_scores)
        scores_file = os.path.join(output_path, "model_scores_comparison.csv")
        scores_df.to_csv(scores_file, index=False)
        print(f"   Tabela comparativa de scores salva em: {scores_file}")


def main():
    """Função principal."""
    args = parse_args()
    
    print("="*70)
    print("AVALIAÇÃO DE MODELO E CRIAÇÃO DA CAMADA GOLD")
    print("="*70)
    
    try:
        # Conectar ao Azure ML
        credential = DefaultAzureCredential()
        
        # Obter informações do workspace das variáveis de ambiente
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
        workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")
        
        if not all([subscription_id, resource_group, workspace_name]):
            print(" Variáveis de ambiente não configuradas. Usando valores padrão.")
            subscription_id = "a9ff0e55-7ef2-474d-8511-e8208d1d6267"
            resource_group = "modelo_prod"
            workspace_name = "modelo_treino"
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Carregar dados e todos os modelos
        X_test, y_test = load_test_data(args.input_test)
        models = load_model(args.input_model)

        # Torneio de modelos: avaliar todos e escolher o campeão
        model_scores = []
        for model_path, model in models:
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                model_name = Path(model_path).stem
                metrics = calculate_metrics(y_test, y_pred, y_proba, model_name)
                model_scores.append({
                    "model_path": model_path,
                    **metrics
                })
            except Exception as e:
                print(f"[WARN] Falha ao avaliar modelo {model_path}: {e}")

        if not model_scores:
            raise RuntimeError("Nenhum modelo pôde ser avaliado.")

        # Escolher campeão
        champion = max(model_scores, key=lambda m: m["composite_score"])
        print(f"\n Modelo campeão: {Path(champion['model_path']).name} (Score: {champion['composite_score']:.4f})")

        # Buscar melhor modelo anterior (com tratamento de autorização)
        best_previous_score = 0.0
        try:
            best_previous_score = get_best_model_score(ml_client, args.model_name)
        except Exception as e:
            import logging
            logging.warning(f"[AzureML] Falha ao buscar modelos anteriores: {e}")

        # Decidir se registra (com tratamento de autorização)
        try:
            if should_register_model(champion["composite_score"], best_previous_score):
                register_model_to_registry(
                    ml_client=ml_client,
                    model=[m for p, m in models if p == champion["model_path"]][0],
                    model_name=args.model_name,
                    metrics=champion,
                    model_path=champion["model_path"]
                )
            else:
                print("\n Modelo não será registrado (performance inferior)")
        except Exception as e:
            import logging
            logging.warning(f"[AzureML] Falha ao registrar modelo: {e}")

        # Criar Gold layer (camada analítica e tabela comparativa)
        y_pred = [m for p, m in models if p == champion["model_path"]][0].predict(X_test)
        y_proba = [m for p, m in models if p == champion["model_path"]][0].predict_proba(X_test)[:, 1]
        create_gold_layer(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            output_path=args.output_gold,
            metrics=champion,
            model_scores=model_scores
        )
        
        print("\n" + "="*70)
        print(" AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f"\n ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
