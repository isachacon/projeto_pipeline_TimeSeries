import pandas as pd
import numpy as np
import duckdb
import lightgbm as lgb
import mlflow
import os
import sys
import argparse
from typing import Optional, List
from pathlib import Path 

# Adiciona o diretório 'src' ao PATH do sistema para que os módulos possam ser importados
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importa as funções de cada módulo
from src.data_processing import ingest_and_treat_data, feature_engineering
from src.model_training import train_lgbm_model
from src.model_inference import make_predictions
from src.model_loading import load_all_artifacts_and_params
from src.utils import create_offset 

# --- 0. Configurações Globais e Parâmetros ---
# Caminhos agora são relativos à raiz do projeto (onde 'pipeline' está)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminhos para os dados brutos (data/src)
PATH_CLIENTES = os.path.join(BASE_DIR, 'data', 'src', 'clientes.csv')
PATH_CLIMA = os.path.join(BASE_DIR, 'data', 'src', 'clima.csv')
PATH_CONSUMO = os.path.join(BASE_DIR, 'data', 'src', 'consumo.csv')

# Caminho para a pasta de saída do DuckDB (data/out)
DUCKDB_OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'out')

# Configuração do MLflow Tracking URI (models/artifacts)
MLFLOW_TRACKING_URI = Path(os.path.join(BASE_DIR, 'models', 'artifacts')).as_uri()

# Parâmetros do Modelo e Features
EXOG_COLS_GLOBAL = [
    'client_id', 'avg_temperature', 'avg_humidity', 'temp_humid_interaction',
    'day_counter', 'consumption_lag_1', 'consumption_lag_2', 'consumption_lag_3',
    'consumption_lag_4', 'consumption_lag_7','consumption_lag_15', 
    'tipo_feriado', 'day_of_week_name', 'month',
    'rolling_mean_3', 'rolling_std_3', 'rolling_mean_7', 'rolling_std_7', 'diff_lag_1'
]

CATEG_COLS_GLOBAL = ['client_id', 'tipo_feriado', 'day_of_week_name', 'month']

# Parâmetros da Validação Walk-Forward
INITIAL_TRAIN_SIZE = '2 months' 
VALIDATION_SIZE = '1 month'    
TEST_SIZE = '1 month'          

# Nome do experimento MLflow
MLFLOW_EXPERIMENT_NAME = "Walk-Forward Validation"
MLFLOW_MAIN_RUN_NAME = "Validação Walk-Forward" 

def run_pipeline(
    run_full_training: bool = True, 
    client_id_to_plot: Optional[str] = 'C0088' 
):
    """
    Orquestra todas as etapas do pipeline de Machine Learning.

    Args:
        run_full_training (bool): Se True, executa as etapas de ingestão, FE e treinamento.
                                  Se False, tenta carregar os artefatos de um run MLflow existente.
        client_id_to_plot (Optional[str]): ID do cliente para gerar plots de previsão.
    """
    print("--- INICIANDO O PIPELINE COMPLETO ---")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow Tracking URI configurado para: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow Experiment definido para: '{MLFLOW_EXPERIMENT_NAME}'")


    model_global = None
    df_train_final = None
    df_val_final = None
    df_test_final = None
    exog_cols_loaded = None
    categ_cols_loaded = None
    df_global_original = None 

    with mlflow.start_run(run_name="Full ML Pipeline Run"):
        mlflow.log_param("pipeline_mode", "training" if run_full_training else "inference_only")

        if run_full_training:
            con_db = duckdb.connect(database=':memory:', read_only=False)
            print("✅ Conexão com DuckDB estabelecida.")
            
            try:
                con_db = ingest_and_treat_data(con_db, PATH_CLIENTES, PATH_CLIMA, PATH_CONSUMO)
                
                df_global_original = feature_engineering(con_db, DUCKDB_OUTPUT_DIR) 
                
                con_db.close()
                print("✅ Conexão DuckDB fechada.")

                model_global, df_train_final, df_val_final, df_test_final = train_lgbm_model(
                    df=df_global_original, 
                    mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
                    exog_cols=EXOG_COLS_GLOBAL,
                    categ_cols=CATEG_COLS_GLOBAL,
                    initial_train_size=INITIAL_TRAIN_SIZE,
                    validation_size=VALIDATION_SIZE,
                    test_size=TEST_SIZE,
                    mlflow_tracking_uri=MLFLOW_TRACKING_URI
                )
                exog_cols_loaded = EXOG_COLS_GLOBAL 
                categ_cols_loaded = CATEG_COLS_GLOBAL 

                if model_global is None:
                    print("\n❌ Pipeline encerrada devido à falha no treinamento.")
                    mlflow.log_param("pipeline_status", "failed_training")
                    return
                mlflow.log_param("pipeline_status", "training_completed")
                
            except Exception as e:
                print(f"\n❌ Erro crítico durante as etapas de ingestão/FE/treinamento: {e}")
                mlflow.log_param("pipeline_status", f"failed_critical_error: {str(e)}")
                return
        else:
            print("\n--- Carregando artefatos de um run MLflow existente ---")
            model_global, df_train_final, df_val_final, df_test_final, exog_cols_loaded, categ_cols_loaded = \
                load_all_artifacts_and_params(
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    run_name=MLFLOW_MAIN_RUN_NAME,
                    mlflow_tracking_uri=MLFLOW_TRACKING_URI
                )
            
            if model_global is None or df_test_final is None or exog_cols_loaded is None or categ_cols_loaded is None:
                print("\n❌ Falha ao carregar artefatos do MLflow. Verifique os logs acima.")
                mlflow.log_param("pipeline_status", "failed_loading_artifacts")
                return
            mlflow.log_param("pipeline_status", "artifacts_loaded")

            # MODIFICAÇÃO CRÍTICA: Carrega o DataFrame global completo para plotagem
            try:
                df_global_original = pd.read_parquet(os.path.join(DUCKDB_OUTPUT_DIR, 'dados_lgbm_global.parquet'))
                df_global_original['date'] = pd.to_datetime(df_global_original['date'])
                df_global_original = df_global_original.set_index('date').sort_index()
                print(f"✅ DataFrame global completo carregado de '{os.path.join(DUCKDB_OUTPUT_DIR, 'dados_lgbm_global.parquet')}' para plotagem.")
            except FileNotFoundError:
                print(f"❌ Erro: 'dados_lgbm_global.parquet' não encontrado em '{DUCKDB_OUTPUT_DIR}'. Certifique-se de ter executado a pipeline com '--train' pelo menos uma vez para gerar este arquivo.")
                df_global_original = None # Garante que a plotagem não tente usar um DF inexistente
            except Exception as e:
                print(f"❌ Erro ao carregar 'dados_lgbm_global.parquet' para plotagem: {e}")
                df_global_original = None


        if model_global and df_test_final is not None and exog_cols_loaded is not None and categ_cols_loaded is not None:
            print("\n--- Realizando Inferência ---")
            df_predictions = make_predictions(
                model=model_global,
                df_to_predict=df_test_final.copy(), 
                exog_cols=exog_cols_loaded,
                categ_cols=categ_cols_loaded
            )
            print("\n✅ Previsões geradas com sucesso. Exemplo:")
            print(df_predictions[['client_id', 'total_consumption', 'prediction']].head())
            mlflow.log_param("inference_status", "completed")

            if client_id_to_plot and not df_predictions.empty:
                import matplotlib.pyplot as plt
                from src.utils import create_offset 

                def plot_full_series_with_predictions(
                    model_global: lgb.LGBMRegressor,
                    df_full: pd.DataFrame, # Este DF agora é o df_global_original
                    exog_cols: List[str],
                    categ_cols: List[str],
                    client_id_to_plot: str,
                    test_size: str
                ):
                    df_client = df_full[df_full['client_id'] == client_id_to_plot].copy()
                    if df_client.empty:
                        print(f"❌ Cliente {client_id_to_plot} não encontrado no DataFrame completo para plotagem.")
                        return

                    # O df_full já deve ter o índice de data e estar ordenado
                    if not isinstance(df_client.index, pd.DatetimeIndex):
                        df_client['date'] = pd.to_datetime(df_client['date'])
                        df_client = df_client.set_index('date').sort_index()


                    test_start_date = df_client.index.max() - create_offset(test_size)
                    
                    # A série histórica agora é tudo ANTES do test_start_date
                    df_client_history = df_client[df_client.index < test_start_date].copy()
                    df_client_test_plot = df_client[df_client.index >= test_start_date].copy()

                    if df_client_test_plot.empty:
                        print(f"❌ Não há dados suficientes para o período de teste para o cliente {client_id_to_plot} para plotagem.")
                        return
                    if df_client_history.empty:
                        print(f"❌ Não há dados históricos suficientes para o cliente {client_id_to_plot} para plotagem.")
                        # Isso pode acontecer se o dataset for muito pequeno para o tamanho do test_size
                        # ou se o cliente não tiver histórico antes do teste.
                        print("Considerando todo o DataFrame do cliente como histórico para plotagem.")
                        df_client_history = df_client_test_plot.copy() # Fallback para plotar o que tiver
                        
                    X_test_client_plot = df_client_test_plot[exog_cols].copy()
                    y_true_client_plot = df_client_test_plot['total_consumption']

                    for col in categ_cols:
                        if col in X_test_client_plot.columns:
                            X_test_client_plot[col] = X_test_client_plot[col].astype('category')
                    
                    numeric_cols_test_plot = X_test_client_plot.select_dtypes(include=np.number).columns
                    X_test_client_plot[numeric_cols_test_plot] = X_test_client_plot[numeric_cols_test_plot].fillna(0)

                    y_pred_client_plot = model_global.predict(X_test_client_plot)

                    plt.figure(figsize=(18, 8))
                    plt.plot(df_client_history.index, df_client_history['total_consumption'], 
                             label='Série Histórica (Consumo Real)', color='b')
                    plt.plot(df_client_test_plot.index, y_true_client_plot, 
                             label='Série de Teste (Consumo Real)', color='g')
                    plt.plot(y_true_client_plot.index, y_pred_client_plot, 
                             label='Previsão do Modelo', color='r', linestyle='--')
                    
                    plt.title(f'Consumo Real vs. Previsões para o Cliente: {client_id_to_plot}', fontsize=18)
                    plt.xlabel('Data', fontsize=12)
                    plt.ylabel('Consumo Total (kWh)', fontsize=12)
                    plt.legend()
                    plt.grid(True)
                    plt.show()

                # MODIFICAÇÃO: Passa df_global_original diretamente para a função de plotagem
                # que agora contém o histórico completo.
                if df_global_original is not None:
                    print(f"\n--- Gerando Plot para o Cliente {client_id_to_plot} ---")
                    plot_full_series_with_predictions(
                        model_global=model_global,
                        df_full=df_global_original, # Passa o DF com todo o histórico
                        exog_cols=exog_cols_loaded,
                        categ_cols=categ_cols_loaded,
                        client_id_to_plot=client_id_to_plot,
                        test_size=TEST_SIZE
                    )
                    mlflow.log_param("plot_generated", "True")
                else:
                    print("❌ Não foi possível gerar o plot: DataFrame completo para plotagem não disponível (verifique se 'dados_lgbm_global.parquet' existe).")
                    mlflow.log_param("plot_generated", "False")
            else:
                mlflow.log_param("plot_generated", "False")
        else:
            mlflow.log_param("inference_status", "skipped_no_model_or_data")

    print("\n--- PIPELINE CONCLUÍDO ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute a pipeline de ML para previsão de consumo de energia.")
    
    parser.add_argument(
        '--train', 
        action='store_true', 
        help="Executa o treinamento completo da pipeline (ingestão, FE, treino, inferência). Por padrão, apenas carrega o modelo para inferência."
    )
    
    parser.add_argument(
        '--client_id', 
        type=str, 
        default='C0088', 
        help="ID do cliente para gerar os gráficos de previsão. Padrão: 'C0088'."
    )

    args = parser.parse_args()

    run_pipeline(
        run_full_training=args.train, 
        client_id_to_plot=args.client_id
    )
