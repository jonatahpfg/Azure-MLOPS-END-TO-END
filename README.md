# Telco Customer Churn - AZURE MLOps End-to-End

Este projeto implementa um pipeline completo de Machine Learning para previsão de churn de clientes de telecom, utilizando práticas modernas de MLOps, governança e automação em Azure Machine Learning.

## Visão Geral
- **Pipeline automatizado:** Treinamento, avaliação, registro e deploy do modelo são realizados via scripts e workflows CI/CD (GitHub Actions).
- **Governança:** O projeto segue boas práticas de versionamento de dados, modelos e ambientes, com controle de cota, limpeza automática de endpoints e ambientes reprodutíveis.
- **Infraestrutura Azure ML:** Utiliza Managed Online Endpoints, ambientes customizados (conda.yml), monitoramento e autenticação segura via Azure Key Vault e Service Principal.
- **Interface de consumo:** Inclui uma aplicação Streamlit para consumo do modelo em produção, com autenticação via chave e visualização amigável dos resultados.

## Estrutura do Projeto
```
modelos.ipynb                # Notebook exploratório e de prototipação
Telco_Customer_Churn.csv     # Base de dados original
ml-project/
  requirements.txt           # Dependências Python
  submit_job.py              # Submissão do pipeline de treino
  config/
    grid_search.yml          # Configuração de hiperparâmetros
  environments/
    conda.yml                # Ambiente reprodutível para Azure ML
  pipelines/
    churn_pipeline.py        # Definição do pipeline de ML
  src/
    data_prep/               # Scripts de preparação de dados
    deploy/
      deploy_model.py        # Deploy do modelo em endpoint
      score.py               # Script de inferência para Azure ML
      test_endpoint.py       # Teste automatizado do endpoint
    evaluation/
      evaluate_gold.py       # Avaliação do modelo
    training/
      train.py               # Treinamento do modelo
.github/
  workflows/
    ml-pipeline.yml          # CI/CD automatizado
```

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

## Como Executar
1. Configure os segredos do Azure no GitHub (.streamlit/secrets.toml para Streamlit).
2. Execute o pipeline via GitHub Actions ou localmente com submit_job.py.
3. Realize o deploy do modelo com deploy_model.py.
4. Teste o endpoint com test_endpoint.py ou via Streamlit.

## Observações
- O projeto está pronto para ser expandido com monitoramento, explainability, retraining e integração com outros sistemas.
- Todos os scripts e assets seguem padrões de mercado para facilitar auditoria, compliance e integração contínua.

---


