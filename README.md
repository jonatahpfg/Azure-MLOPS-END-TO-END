
📡 Telco Customer Churn f Azure MLOps End-to-End

Este repositrio apresenta um projeto completo de MLOps em Azure, implementando um pipeline end-to-end de Machine Learning para previso de churn de clientes de telecomunicaes, seguindo prticas modernas de automao, governana, rastreabilidade e deploy em produo.

O foco do projeto  demonstrar como modelos de ML so desenvolvidos, versionados, treinados, registrados e disponibilizados em ambiente corporativo, utilizando Azure Machine Learning, MLflow e CI/CD.

🎯 **Objetivo do Projeto**

- Construir um pipeline profissional de ML para previso de churn
- Aplicar boas prticas de MLOps usadas no mercado
- Demonstrar domnio de Azure Machine Learning (SDK v2)
- Garantir reprodutibilidade, rastreabilidade e governana
- Disponibilizar o modelo via endpoint gerenciado
- Criar uma interface simples de consumo (Streamlit)

🧠 **Viso Geral da Arquitetura**

- Ingesto e preparao de dados
- Treinamento de mltiplos modelos com variao de hiperparmetros
- Rastreamento de experimentos com MLflow
- Registro do melhor modelo no Model Registry
- Deploy automatizado em Managed Online Endpoint
- Validao automtica do endpoint
- Automao completa via GitHub Actions (CI/CD)
- Consumo do modelo via API e Streamlit

🏗️ **Infraestrutura e Ferramentas**

- Azure Machine Learning (SDK v2)
- MLflow (Tracking + Model Registry)
- Azure Managed Online Endpoints
- Azure Key Vault (segredos e autenticao)
- Service Principal para automao segura
- GitHub Actions para CI/CD
- Conda / Ambientes reprodutveis
- Streamlit para interface simples e demonstrativa de consumo

## Estrutura do Projeto
```
modelos.ipynb                # Notebook exploratrio e de prototipao
Telco_Customer_Churn.csv     # Base de dados original, ela é captada no Datalakedados na camada bronze
ml-project/
  requirements.txt           # Dependncias Python
  submit_job.py              # Submisso do pipeline de treino
  config/
    grid_search.yml          # Configurao de hiperparmetros
  environments/
    conda.yml                # Ambiente reprodutvel para Azure ML
  pipelines/
    churn_pipeline.py        # Definio do pipeline de ML
  src/
    data_prep/               # Scripts de preparao de dados
    deploy/
      deploy_model.py        # Deploy do modelo em endpoint
      score.py               # Script de inferncia para Azure ML
      test_endpoint.py       # Teste automatizado do endpoint
    evaluation/
      evaluate_gold.py       # Avaliao do modelo
    training/
      train.py               # Treinamento do modelo
.github/
  workflows/
    ml-pipeline.yml          # CI/CD automatizado
```

🔍 **Governana e Boas Prticas de Mercado**

✔ Versionamento completo
  - Dados
  - Ambientes
  - Modelos
  - Experimentos

✔ Rastreabilidade
  - Cada treino  registrado no MLflow
  - Mtricas, parmetros e artefatos versionados

✔ Ambientes reprodutveis
  - Conda YAML garante consistncia entre dev, teste e produo

✔ Automao e CI/CD
  - Pipeline automatizado com GitHub Actions
  - Treino, deploy e validao sem interveno manual

✔ Gesto de custos
  - Limpeza automtica de endpoints antigos
  - Controle de cota no Azure ML

✔ Segurana
  - Autenticao via Azure AD
  - Segredos protegidos no Key Vault
  - Nenhuma credencial hardcoded

🚀 **Diferenciais do Projeto**

- Estrutura 100% alinhada ao que empresas usam em produo
- Fcil adaptao para outros problemas de negcio
- Modular, escalvel e auditvel
- Ideal para validar boas práticas das funções Data Scientist / ML Engineer / MLOps
- Demonstra domnio de Azure + MLOps.

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


---Modelo campeão: model_LogisticRegression (Score composto = 0.7521)
                        [713 320]
                        [ 52 320] excelente recall de  0.8602, modelo muito bom para detectar clientes com real probabilidade de churn, possibilitando medidas para resolução do problema.



