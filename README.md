# 📡 Telco Customer Churn: Azure MLOps End-to-End
Este repositório apresenta um projeto completo de MLOps em Azure, implementando um pipeline end-to-end de Machine Learning para previsão de churn de clientes de telecomunicações.
O projeto segue as práticas mais modernas de automação, governança, rastreabilidade e deploy em produção.
O foco é demonstrar como modelos de ML são desenvolvidos, versionados, treinados e disponibilizados em ambiente corporativo utilizando Azure Machine Learning, MLflow e CI/CD.

## 🎯 Objetivos do Projeto
Pipeline Profissional: Construir um fluxo automatizado de ML para previsão de churn.
Práticas de Mercado: Aplicar conceitos de MLOps usados por grandes empresas.
Domínio Azure: Demonstrar competência técnica no Azure Machine Learning SDK v2.
Governança: Garantir reprodutibilidade e rastreabilidade total de experimentos.
Deploy Gerenciado: Disponibilizar o modelo via Managed Online Endpoint.
Interface de Consumo: Criar uma aplicação Streamlit para usuários finais.

## 🏗️ Arquitetura e Estrutura Medallion
O projeto organiza o ciclo de vida dos dados em camadas para garantir a integridade do processo:

Camada	Processo	O que visualizar na Azure?
Bronze	Ingestão	O arquivo bruto Telco_Customer_Churn.csv no Datastore.
Silver	Preparação (prep.py)	Limpeza e tratamento de nulos salvos como arquivos .parquet
Gold	Treinamento (train.py)	Modelos treinados e registrados com hiperparâmetros

## 🛠️ Infraestrutura e Ferramentas
Azure Machine Learning (SDK v2): Orquestração completa do ciclo de vida.
MLflow: Tracking de métricas, parâmetros e registro do modelo (Model Registry).
Managed Online Endpoints: Hospedagem escalável da API de predição.
Azure Key Vault: Gestão segura de segredos e autenticação.
GitHub Actions: Automação total via CI/CD (Pipeline automatizado).
Conda / Docker: Ambientes isolados e reprodutíveis.
Streamlit: Interface amigável para consumo real do modelo.

## Estrutura do Projeto
```
modelos.ipynb                # Notebook exploratório e de prototipao
Telco_Customer_Churn.csv     # Base de dados original
ml-project/
  requirements.txt           # Dependências Python
  submit_job.py              # Submisso do pipeline de treino
  config/
    grid_search.yml          # Configuraçãoo de hiperparmetros
  environments/
    conda.yml                # Ambiente reprodutvel para Azure ML
  pipelines/
    churn_pipeline.py        # Definio do pipeline de ML
  src/
    data_prep/               # Scripts de preparao de dados
    deploy/
      deploy_model.py        # Deploy do modelo em endpoint
      score.py               # Script de inferência para Azure ML
      test_endpoint.py       # Teste automatizado do endpoint
    evaluation/
      evaluate_gold.py       # Avaliao do modelo
    training/
      train.py               # Treinamento do modelo
.github/
  workflows/
    ml-pipeline.yml          # CI/CD automatizado
```

## 🔍 Governança e Rastreabilidade (MLflow)
Dentro do Azure ML Studio, cada treinamento é uma "Run" única onde você pode visualizar:

Métricas em Tempo Real: F1-Score, Acurácia, Recall e Precisão logados automaticamente.
Artefatos de Modelo: O arquivo .pkl do modelo acompanhado do preprocessor.joblib (garantindo que a transformação de dados viaje com o modelo).
Feature Importance: Gráfico automático mostrando quais variáveis (como tipo de contrato e tempo de serviço) mais impactam o churn.
Reprodutibilidade: O exato ambiente Conda e a versão do código usados no treino são registrados.

## 🏆 Resultados e Performance
Modelo Campeão: model_LogisticRegression (Score Composto = 0.7521)
O modelo apresentou um desempenho superior na detecção de churn com foco em sensibilidade:
Matriz de Confusão:
[713 320] (Negativos)
[ 52 320] (Positivos)

Recall: 0.8602
Excelente capacidade de detectar clientes com real probabilidade de churn, permitindo ações proativas de retenção.
- Demonstra domínio real de Azure + MLOps, não apenas notebooks

## Práticas de Governança e Mercado
- **Versionamento de assets:** Dados, modelos e ambientes são versionados e registrados no Azure ML, garantindo rastreabilidade e reprodutibilidade.
- **Limpeza automática de recursos:** Scripts de deploy removem endpoints antigos para liberar cota e evitar custos desnecessários.
- **Ambientes reprodutíveis:** Uso de conda.yml e requirements.txt para garantir que o ambiente de execução seja idêntico em desenvolvimento, teste e produção.
- **Monitoramento e logging:** Logs detalhados em todos os scripts, integração com Application Insights e Key Vault para segurança.
- **Automação CI/CD:** Workflows GitHub Actions para lint, teste, treino, deploy e validação do endpoint, seguindo o que o mercado exige em DevOps/MLOps.
- **Interface amigável:** Streamlit para consumo do modelo, facilitando a integração com times de negócio.

## Vantagens do Projeto
- **Escalabilidade:** Pronto para múltiplos modelos, pipelines e ambientes.
- **Segurança:** Autenticação via Azure, segredos protegidos, governança de recursos.
- **Flexibilidade:** Modularidade dos scripts permite fácil adaptação para outros casos de uso.
- **Aderência ao mercado:** Estrutura e práticas alinhadas com demandas reais de empresas que usam Azure ML, CI/CD e governança de dados/modelos.

---




