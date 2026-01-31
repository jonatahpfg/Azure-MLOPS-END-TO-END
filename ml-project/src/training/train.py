from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Função para criar e ajustar o preprocessor
def create_and_fit_preprocessor(X_train):
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    numeric_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ]
    all_features = categorical_features + numeric_features
    missing = [col for col in all_features if col not in X_train.columns]
    if missing:
        logger.warning(f'Colunas ausentes no treino: {missing}')
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), [col for col in numeric_features if col in X_train.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), [col for col in categorical_features if col in X_train.columns])
    ])
    preprocessor.fit(X_train)
    return preprocessor
"""
===================================================================
SCRIPT DE TREINAMENTO DE MODELOS (SILVER -> MODELO)
===================================================================
Descricao: Etapa 2 - Treina modelos de ML com GridSearch e MLflow
- Input: Dados processados da camada Silver (train.parquet)
- Processo: SMOTE, GridSearch, validacao cruzada, logging MLflow
- Output: Modelo treinado e registrado no MLflow

===================================================================
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Suprime warnings desnecessarios
warnings.filterwarnings('ignore')

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de modelos com MLflow")
    
    parser.add_argument("--input_train", type=str, required=True,
                        help="Caminho para dados de treino (Silver layer)")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression', 'all'],
                        help="Tipo de modelo a ser treinado ou 'all' para todos")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Caminho para arquivo de configuracao (grid_search.yml)")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Caminho de saida para o modelo treinado")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed para reprodutibilidade")

    parser.add_argument("--apply_smote", type=str, required=False, default=None,
                        help="(Ignorado) Compatibilidade com pipeline antigo. SMOTE agora é automático no pipeline.")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    try:
        logger.info(f"Carregando configuracao de: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Configuracao carregada com sucesso")
        return config
    except Exception as e:
        logger.error(f"Erro ao carregar configuracao: {str(e)}")
        raise


def load_train_data(file_path: str) -> tuple:
    try:
        logger.info(f"Carregando dados de treino de: {file_path}")
        df = pd.read_parquet(file_path)
        X_train = df.drop('Churn', axis=1)
        y_train = df['Churn']
        logger.info(f"Dados carregados. Shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Distribuicao target: {y_train.value_counts().to_dict()}")
        return X_train, y_train
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise


def get_model_and_params(model_type: str, config: dict, y_train: pd.Series) -> tuple:
    try:
        logger.info(f"Configurando modelo: {model_type}")
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1
        logger.info(f"Scale pos weight calculado: {scale_pos_weight:.2f}")
        
        if model_type == 'RandomForest':
            model = RandomForestClassifier()
            param_grid = config['RandomForest']
        elif model_type == 'XGBoost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            param_grid = config['XGBoost']
            param_grid['scale_pos_weight'] = [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2]
        elif model_type == 'LightGBM':
            model = LGBMClassifier(verbose=-1)
            param_grid = config['LightGBM']
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=2000)
            param_grid = config['LogisticRegression']
        else:
            raise ValueError(f"Modelo nao suportado: {model_type}")
        
        logger.info(f"Modelo configurado. Parametros para busca: {len(param_grid)} chaves")
        return model, param_grid
    except Exception as e:
        logger.error(f"Erro ao configurar modelo: {str(e)}")
        raise


def perform_grid_search(model, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series, config: dict, random_state: int) -> GridSearchCV:
    try:
        logger.info("Iniciando GridSearch com Cross-Validation, Preprocessor, SMOTE no Pipeline...")
        gs_config = config.get('grid_search_config', {})
        cv = gs_config.get('cv', 5)
        scoring = gs_config.get('scoring', 'f1')
        n_jobs = gs_config.get('n_jobs', -1)
        verbose = gs_config.get('verbose', 2)
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        # Cria e ajusta o preprocessor
        preprocessor = create_and_fit_preprocessor(X_train)
        imb_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=random_state, k_neighbors=5)),
            ('classifier', model)
        ])
        param_grid_prefixed = {f'classifier__{k}': v for k, v in param_grid.items()}
        grid_search = GridSearchCV(
            estimator=imb_pipeline,
            param_grid=param_grid_prefixed,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        logger.info(f"Executando GridSearch com {cv} folds, metrica: {scoring}")
        grid_search.fit(X_train, y_train)
        logger.info(f"GridSearch concluido!")
        logger.info(f"   - Melhor score (CV): {grid_search.best_score_:.4f}")
        logger.info(f"   - Melhores parametros: {grid_search.best_params_}")
        # Salva o preprocessor ajustado para logging posterior
        joblib.dump(preprocessor, 'preprocessor.joblib')
        return grid_search
    except Exception as e:
        logger.error(f"Erro durante GridSearch: {str(e)}")
        raise


def log_to_mlflow(model_type: str, grid_search: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series, output_model: str, custom_model_name: str = None, preprocessor=None) -> None:
    try:
        logger.info("Registrando experimento no MLflow...")
        with mlflow.start_run():
            mlflow.log_param("model_type", model_type)
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(param, value)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_train)
            metrics = {
                'cv_score': grid_search.best_score_,
                'train_accuracy': accuracy_score(y_train, y_pred),
                'train_precision': precision_score(y_train, y_pred, zero_division=0),
                'train_recall': recall_score(y_train, y_pred, zero_division=0),
                'train_f1': f1_score(y_train, y_pred, zero_division=0),
            }
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"   - {metric_name}: {metric_value:.4f}")

            # Feature Importance (se disponivel no classificador do pipeline)
            classifier = best_model.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                top_n = min(20, len(importances))
                indices = np.argsort(importances)[-top_n:]
                plt.figure(figsize=(10, 8))
                plt.barh(range(top_n), importances[indices])
                plt.yticks(range(top_n), [f'Feature {i}' for i in indices])
                plt.xlabel('Importance')
                plt.title(f'Top {top_n} Feature Importances - {model_type}')
                fi_path = f'feature_importance_{model_type}.png' if custom_model_name is None else f'feature_importance_{custom_model_name}.png'
                plt.savefig(fi_path)
                mlflow.log_artifact(fi_path)
                plt.close()

            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, y_pred)
            try:
                mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model", signature=signature)
                logger.info("Modelo salvo como artefato no MLflow.")
            except Exception as e:
                logger.error(f"Erro ao registrar modelo no MLflow: {str(e)}")

            parent = os.path.dirname(output_model)
            if parent:
                os.makedirs(parent, exist_ok=True)
            joblib.dump(best_model, output_model)
            mlflow.log_artifact(output_model)
            logger.info(f"Modelo salvo localmente em: {output_model}")

            # Loga o preprocessor.joblib dentro da pasta do modelo no MLflow
            preprocessor_path = 'preprocessor.joblib'
            if os.path.exists(preprocessor_path):
                mlflow.log_artifact(preprocessor_path, artifact_path='model')
                logger.info(f"Preprocessor salvo e logado em: {preprocessor_path} (dentro da pasta do modelo)")
            else:
                logger.warning(f"Preprocessor.joblib não encontrado para log_artifact.")
    except Exception as e:
        logger.error(f"Erro ao registrar no MLflow: {str(e)}")
        raise


def main():
    try:
        args = parse_args()
        logger.info("="*70)
        logger.info(f"INICIANDO TREINAMENTO - MODELO: {args.model_type}")
        logger.info("="*70)
        
        config = load_config(args.config_path)
        X_train, y_train = load_train_data(args.input_train)

        # Cria e ajusta o preprocessor
        preprocessor = create_and_fit_preprocessor(X_train)

        if args.model_type == 'all':
            model_keys = [k for k in config.keys() if k in ['RandomForest', 'XGBoost', 'LightGBM', 'LogisticRegression']]
            if args.output_model.endswith('.joblib'):
                best_score = -float('inf')
                best_model_name = None
                best_grid_search = None
                for model_name in model_keys:
                    logger.info("="*70)
                    logger.info(f"TREINANDO MODELO: {model_name}")
                    logger.info("="*70)
                    model, param_grid = get_model_and_params(model_name, config, y_train)
                    grid_search = perform_grid_search(model, param_grid, X_train, y_train, config, args.random_state)
                    score = grid_search.best_score_
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_grid_search = grid_search
                parent_dir = os.path.dirname(args.output_model)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                log_to_mlflow(best_model_name, best_grid_search, X_train, y_train, args.output_model, custom_model_name=best_model_name, preprocessor=preprocessor)
            else:
                output_dir = args.output_model
                os.makedirs(output_dir, exist_ok=True)
                for model_name in model_keys:
                    logger.info("="*70)
                    logger.info(f"TREINANDO MODELO: {model_name}")
                    logger.info("="*70)
                    model, param_grid = get_model_and_params(model_name, config, y_train)
                    grid_search = perform_grid_search(model, param_grid, X_train, y_train, config, args.random_state)
                    output_model_path = os.path.join(output_dir, f"model_{model_name}.joblib")
                    log_to_mlflow(model_name, grid_search, X_train, y_train, output_model_path, custom_model_name=model_name, preprocessor=preprocessor)
        else:
            model, param_grid = get_model_and_params(args.model_type, config, y_train)
            grid_search = perform_grid_search(model, param_grid, X_train, y_train, config, args.random_state)
            parent_dir = os.path.dirname(args.output_model)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            output_model_path = os.path.join(args.output_model, "model.joblib")
            log_to_mlflow(args.model_type, grid_search, X_train, y_train, output_model_path, preprocessor=preprocessor)
        
        logger.info("="*70)
        logger.info("TREINAMENTO CONCLUIDO COM SUCESSO!")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"ERRO CRITICO: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
