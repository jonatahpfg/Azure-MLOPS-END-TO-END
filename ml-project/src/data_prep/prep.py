"""
===================================================================
SCRIPT DE PREPARAÇÃO DE DADOS (BRONZE -> SILVER)
===================================================================
Descrição: Etapa 1 da Medallion Architecture
- Input: Dados brutos da camada Bronze (Telco_Customer_Churn.csv)
- Processo: Limpeza, transformação e feature engineering
- Output: Dados processados na camada Silver (train.parquet, test.parquet)
          + Artefato de preprocessamento (preprocessor.joblib)


===================================================================
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
    parser = argparse.ArgumentParser(description="Preparação de dados Bronze -> Silver")
    
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Caminho para os dados brutos (Bronze layer)"
    )
    
    parser.add_argument(
        "--output_train",
        type=str,
        required=True,
        help="Caminho de saída para dados de treino (Silver layer)"
    )
    
    parser.add_argument(
        "--output_test",
        type=str,
        required=True,
        help="Caminho de saída para dados de teste (Silver layer)"
    )
    
    parser.add_argument(
        "--output_preprocessor",
        type=str,
        required=True,
        help="Caminho de saída para o preprocessor serializado"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporção de dados para teste (padrão: 0.2)"
    )
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (padrão: 42)"
    )
    
    return parser.parse_args()


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Carrega dados brutos da camada Bronze.
    
    Args:
        file_path: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados brutos
    """
    try:
        logger.info(f"Carregando dados de: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpeza básica dos dados.
    
    Operações:
    - Remove coluna customerID (não é feature)
    - Trata valores vazios em TotalCharges
    - Remove duplicatas
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame limpo
    """
    try:
        logger.info("Iniciando limpeza de dados...")
        
        # Remove customerID (não é feature preditiva)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            logger.info("Coluna 'customerID' removida")
        
        # Converte TotalCharges para numérico (pode ter espaços vazios)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Preenche valores nulos com 0 (clientes novos sem histórico)
            missing_count = df['TotalCharges'].isna().sum()
            if missing_count > 0:
                logger.info(f"Preenchendo {missing_count} valores nulos em TotalCharges com 0")
                df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Remove duplicatas
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removidas {removed_rows} linhas duplicadas")
        
        logger.info(f"Limpeza concluída. Shape final: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Erro durante limpeza: {str(e)}")
        raise


def binarize_target(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Binariza a variável alvo (Yes/No -> 1/0).
    
    Args:
        df: DataFrame com coluna target
        target_col: Nome da coluna alvo
        
    Returns:
        DataFrame com target binarizado
    """
    try:
        if target_col in df.columns:
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
            logger.info(f"Variável alvo '{target_col}' binarizada")
            logger.info(f"Distribuição: {df[target_col].value_counts().to_dict()}")
        return df
    except Exception as e:
        logger.error(f"Erro ao binarizar target: {str(e)}")
        raise


def identify_feature_types(df: pd.DataFrame, target_col: str = 'Churn') -> tuple:
    """
    Identifica features numéricas e categóricas.
    
    Args:
        df: DataFrame
        target_col: Nome da coluna alvo (para excluir)
        
    Returns:
        Tupla (numeric_features, categorical_features)
    """
    try:
        # Separa features da target
        features = [col for col in df.columns if col != target_col]
        
        # Identifica numéricas (já em formato numérico)
        numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Identifica categóricas (object ou com poucos valores únicos)
        categorical_features = df[features].select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Features numéricas ({len(numeric_features)}): {numeric_features}")
        logger.info(f"Features categóricas ({len(categorical_features)}): {categorical_features}")
        
        return numeric_features, categorical_features
        
    except Exception as e:
        logger.error(f"Erro ao identificar tipos de features: {str(e)}")
        raise


def create_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Cria pipeline de preprocessamento.
    
    Pipeline:
    - Numéricas: MinMaxScaler (normalização para [0,1])
    - Categóricas: OneHotEncoder (com handle_unknown='ignore')
    
    Args:
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas
        
    Returns:
        ColumnTransformer configurado
    """
    try:
        logger.info("Criando pipeline de preprocessamento...")
        
        # Pipeline para features numéricas
        numeric_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ])
        
        # Pipeline para features categóricas
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combina ambos os pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'  # Mantém outras colunas (se houver)
        )
        
        logger.info("Preprocessor criado com sucesso")
        return preprocessor
        
    except Exception as e:
        logger.error(f"Erro ao criar preprocessor: {str(e)}")
        raise


def split_and_save_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    output_train: str,
    output_test: str,
    output_preprocessor: str,
    test_size: float,
    random_state: int,
    target_col: str = 'Churn'
) -> None:
    """
    Separa dados em treino/teste, aplica transformações e salva.
    
    Args:
        df: DataFrame completo
        preprocessor: Pipeline de preprocessamento
        output_train: Caminho de saída para treino
        output_test: Caminho de saída para teste
        output_preprocessor: Caminho de saída para preprocessor
        test_size: Proporção de teste
        random_state: Seed
        target_col: Nome da coluna alvo
    """
    try:
        logger.info("Separando dados em treino e teste...")
        
        # Separa features e target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Split estratificado (mantém proporção de Churn)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # Ajusta preprocessor nos dados de TREINO apenas (evita data leakage)
        logger.info("Ajustando preprocessor nos dados de treino...")
        preprocessor.fit(X_train)

        # Garante que diretórios de saída existem
        os.makedirs(os.path.dirname(output_train), exist_ok=True)
        os.makedirs(os.path.dirname(output_test), exist_ok=True)
        os.makedirs(os.path.dirname(output_preprocessor), exist_ok=True)

        # Salva dados originais (com nomes de colunas originais) em Parquet
        logger.info(f"Salvando dados de treino em: {output_train}")
        train_df = X_train.copy()
        train_df['Churn'] = y_train.values
        train_df.to_parquet(output_train, index=False)

        logger.info(f"Salvando dados de teste em: {output_test}")
        test_df = X_test.copy()
        test_df['Churn'] = y_test.values
        test_df.to_parquet(output_test, index=False)

        # Salva preprocessor para uso posterior (deploy)
        logger.info(f"Salvando preprocessor em: {output_preprocessor}")
        joblib.dump(preprocessor, output_preprocessor)

        logger.info("✅ Preparação de dados concluída com sucesso!")
        logger.info(f"   - Treino: {train_df.shape[0]} amostras")
        logger.info(f"   - Teste: {test_df.shape[0]} amostras")
        logger.info(f"   - Features: {X_train.shape[1]}")
        
    except Exception as e:
        logger.error(f"Erro ao processar e salvar dados: {str(e)}")
        raise


def main():
    """
    Função principal de execução.
    """
    try:
        # Parse argumentos
        args = parse_args()
        
        logger.info("="*70)
        logger.info("INICIANDO PREPARAÇÃO DE DADOS (BRONZE -> SILVER)")
        logger.info("="*70)
        
        # 1. Carrega dados brutos
        df = load_raw_data(args.input_data)
        
        # 2. Limpeza de dados
        df = clean_data(df)
        
        # 3. Binariza target
        df = binarize_target(df)
        
        # 4. Identifica tipos de features
        numeric_features, categorical_features = identify_feature_types(df)
        
        # 5. Cria preprocessor
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        
        # 6. Split, transforma e salva
        split_and_save_data(
            df=df,
            preprocessor=preprocessor,
            output_train=args.output_train,
            output_test=args.output_test,
            output_preprocessor=args.output_preprocessor,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info("="*70)
        logger.info("PROCESSO CONCLUÍDO COM SUCESSO!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"ERRO CRÍTICO: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
