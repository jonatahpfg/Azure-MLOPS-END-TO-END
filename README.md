📡 Telco Customer Churn: Azure MLOps End-to-EndEste repositório apresenta um projeto completo de MLOps em Azure, implementando um pipeline end-to-end de Machine Learning para previsão de churn de clientes de telecomunicações. O projeto segue as práticas mais modernas de automação, governança, rastreabilidade e deploy em produção.O foco é demonstrar como modelos de ML são desenvolvidos, versionados, treinados e disponibilizados em ambiente corporativo utilizando Azure Machine Learning, MLflow e CI/CD.

🎯 Objetivos do ProjetoPipeline Profissional: Construir um fluxo automatizado de ML para previsão de churn.Práticas de Mercado: Aplicar conceitos de MLOps usados por grandes empresas.Domínio Azure: Demonstrar competência técnica no Azure Machine Learning SDK v2.Governança: Garantir reprodutibilidade e rastreabilidade total de experimentos.Deploy Gerenciado: Disponibilizar o modelo via Managed Online Endpoint.Interface de Consumo: Criar uma aplicação Streamlit para usuários finais.

🏗️ Arquitetura e Estrutura MedallionO projeto organiza o ciclo de vida dos dados em camadas para garantir a integridade do processo:CamadaProcessoO que visualizar na Azure?BronzeIngestãoO arquivo bruto Telco_Customer_Churn.csv no Datastore.SilverPreparação (prep.py)Limpeza e tratamento de nulos salvos como ficheiros .parquet.GoldTreinamento (train.py)Modelos treinados e registrados com hiperparâmetros otimizados.

🛠️ Infraestrutura e FerramentasAzure Machine Learning (SDK v2): Orquestração completa do ciclo de vida.MLflow: Tracking de métricas, parâmetros e registro do modelo (Model Registry).Managed Online Endpoints: Hospedagem escalável da API de predição.Azure Key Vault: Gestão segura de segredos e autenticação.GitHub Actions: Automação total via CI/CD (Pipeline automatizado).Conda / Docker: Ambientes isolados e reprodutíveis.Streamlit: Interface amigável para consumo real do modelo.

📁 Estrutura de PastasPlaintextmodelos.ipynb                # Notebook exploratório e prototipação
Telco_Customer_Churn.csv     # Base captada na camada Bronze
ml-project/
  requirements.txt           # Dependências Python
  submit_job.py              # Submissão do pipeline de treino
  config/
    grid_search.yml          # Configuração de hiperparâmetros
  environments/
    conda.yml                # Ambiente reprodutível Azure ML
  pipelines/
    churn_pipeline.py        # Definição do pipeline de ML
  src/
    data_prep/
      prep.py                # Preparação e limpeza (Silver Layer)
    deploy/
      deploy_model.py        # Deploy do Managed Endpoint
      score.py               # Script de inferência (Inference Logic)
      test_endpoint.py       # Validação do serviço em produção
    evaluation/
      evaluate_gold.py       # Avaliação comparativa de modelos
    training/
      train.py               # Script de treino com MLflow e SMOTE
.github/
  workflows/
    ml-pipeline.yml          # Fluxo de CI/CD automatizado

🔍 Governança e Rastreabilidade (MLflow)Dentro do Azure ML Studio, cada treinamento é uma "Run" única onde você pode visualizar:Métricas em Tempo Real: F1-Score, Acurácia, Recall e Precisão logados automaticamente.Artefatos de Modelo: O arquivo .pkl do modelo acompanhado do preprocessor.joblib (garantindo que a transformação de dados viaje com o modelo).Feature Importance: Gráfico automático mostrando quais variáveis (como tipo de contrato e tempo de serviço) mais impactam o churn.Reprodutibilidade: O exato ambiente Conda e a versão do código usados no treino são registrados.<br>
🏆 Resultados e PerformanceModelo Campeão: model_LogisticRegression (Score Composto = 0.7521)O modelo apresentou um desempenho superior na detecção de churn com foco em sensibilidade:Matriz de Confusão:
[713  320] (Negativos)<br> 
[ 52  320] (Positivos)  Recall: 0.8602 (Excelente capacidade de detectar clientes com real probabilidade de churn, permitindo ações proativas de retenção).


