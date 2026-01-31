"""
===================================================================
AZURE ML PIPELINE - CHURN PREDICTION 
===================================================================
Descrição: Pipeline simplificado de ML usando Azure ML SDK v2
- Etapa 1: Data Preparation (Bronze -> Silver)
- Etapa 2: Model Training (Silver -> Model)
- Etapa 3: Evaluation (Model -> Gold)

MODO SEGURANÇA:
- Outputs SEM caminhos explícitos (Azure gerencia automaticamente)
- Parâmetros hardcoded nas strings de comando
- Apenas raw_data_path e config_path como inputs do pipeline
- Evita erros de permissão/validação de datastore

===================================================================
"""


from azure.ai.ml import Input, Output, command, dsl
from azure.ai.ml.constants import AssetTypes

# ================= COMPONENTES DEFINIDOS FORA DO PIPELINE =================

def get_components(environment):
    prep_step_component = command(
        name="data_preparation",
        display_name="Data Preparation (Bronze -> Silver)",
        description="Limpa, transforma e engenharia de features",
        code="./src/data_prep",
        command="""
        python prep.py \
            --input_data ${{inputs.raw_data}} \
            --output_train ${{outputs.train_data}} \
            --output_test ${{outputs.test_data}} \
            --output_preprocessor ${{outputs.preprocessor}} \
            --test_size 0.2 \
            --random_state 123
        """,
        inputs={
            "raw_data": Input(type=AssetTypes.URI_FILE),
        },
        outputs={
            "train_data": Output(type=AssetTypes.URI_FILE),
            "test_data": Output(type=AssetTypes.URI_FILE),
            "preprocessor": Output(type=AssetTypes.URI_FILE),
        },
        environment=environment,
        compute="clustertreino"
    )

    train_step_component = command(
        name="model_training_v2",
        display_name="Model Training with GridSearchCV",
        description="Treina modelo selecionado com GridSearch e MLflow",
        code="./src/training",
        command="""
        python train.py \
            --input_train ${{inputs.train_data}} \
            --model_type ${{inputs.model_type}} \
            --config_path ${{inputs.config_path}} \
            --random_state 123 \
            --output_model ${{outputs.trained_model}}
        """,
        inputs={
            "train_data": Input(type=AssetTypes.URI_FILE),
            "config_path": Input(type=AssetTypes.URI_FILE),
            "model_type": Input(type="string"),
        },
        outputs={
            "trained_model": Output(type=AssetTypes.URI_FOLDER),
        },
        environment=environment,
        compute="clustertreino",
        allow_reuse=False  # Desativa reuso para evitar problemas de cache
    )

    evaluate_step_component = command(
        name="model_evaluation",
        display_name="Model Evaluation & Gold Layer",
        description="Avalia modelo, cria Gold layer e registra melhor modelo",
        code="./src/evaluation",
        command="""
        python evaluate_gold.py \
            --input_test ${{inputs.test_data}} \
            --input_model ${{inputs.trained_model}} \
            --output_gold ${{outputs.gold_data}} \
            --model_name "TelcoChurnModel"
        """,
        inputs={
            "test_data": Input(type=AssetTypes.URI_FILE),
            "trained_model": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={
            "gold_data": Output(type=AssetTypes.URI_FOLDER),
        },
        environment=environment,
        compute="clustertreino",
        allow_reuse=False  # Desativa reuso para evitar problemas de cache
    )
    return prep_step_component, train_step_component, evaluate_step_component

def create_pipeline_job(raw_data_uri, environment, model_type, config_path="./config/grid_search.yml"):
    
    prep_f, train_f, eval_f = get_components(environment)

    
    @dsl.pipeline(
        name="telco_churn_pipeline_safe",
        default_compute="clustertreino",
    )
    def churn_prediction_pipeline(raw_data_path, config_path, model_type):
        # Chamamos as funções dos componentes diretamente do escopo superior
        prep_node = prep_f(raw_data=raw_data_path)

        train_node = train_f(
            train_data=prep_node.outputs.train_data,
            config_path=config_path,
            model_type=model_type
        )
        train_node.allow_reuse = False

        eval_node = eval_f(
            test_data=prep_node.outputs.test_data,
            trained_model=train_node.outputs.trained_model,
        )
        eval_node.allow_reuse = False

        return {"model": train_node.outputs.trained_model}

   
    return churn_prediction_pipeline(
        raw_data_path=raw_data_uri,
        config_path=config_path,
        model_type=model_type  
    )
