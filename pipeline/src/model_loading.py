import mlflow
import pandas as pd
import lightgbm as lgb
import os
import ast 
from typing import Optional, Tuple, List, Dict, Any, Union

from src.utils import create_offset 

def load_all_artifacts_and_params(
    experiment_name: str,
    run_name: str,
    mlflow_tracking_uri: str, # MODIFICAÇÃO: Recebe o URI de tracking do MLflow
    model_artifact_prefix: str = "lgbm_model_fold_", 
    df_artifact_names: Tuple[str, str, str] = ("df_train_final.csv", "df_val_final.csv", "df_test_final.csv")
) -> Tuple[Optional[lgb.LGBMRegressor], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]], Optional[List[str]]]:
    """
    Busca um run do MLflow pelo nome, carrega o modelo, os DataFrames de dados e os parâmetros.

    Args:
        experiment_name (str): O nome do experimento do MLflow.
        run_name (str): O nome do run principal (Walk-Forward Validation) que contém os artefatos.
        mlflow_tracking_uri (str): O URI onde o MLflow está salvando os logs e artefatos.
        model_artifact_prefix (str): O prefixo do nome do artefato do modelo (ex: "lgbm_model_fold_").
                                     Será combinado com o número do fold para carregar o modelo do último fold.
        df_artifact_names (Tuple[str, str, str]): Uma tupla com os nomes dos arquivos dos DataFrames.

    Returns:
        Tuple[Optional[lgb.LGBMRegressor], pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]: 
        Retorna o modelo do último fold, df_train_final, df_val_final, df_test_final, 
        a lista de 'exog_cols' e a lista de 'categ_cols', ou uma tupla de None em caso de falha.
    """
    print(f"--- Tentando carregar todos os artefatos e parâmetros do run '{run_name}' ---")
    
    model, df_train, df_val, df_test, exog_cols_loaded, categ_cols_loaded = None, None, None, None, None, None
    
    try:
        # MODIFICAÇÃO: Configura o URI de tracking do MLflow aqui
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experimento '{experiment_name}' não encontrado.")
            return None, None, None, None, None, None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            print(f"❌ Run com nome '{run_name}' não encontrado no experimento '{experiment_name}'.")
            return None, None, None, None, None, None
            
        main_run_id = runs.iloc[0].run_id
        print(f"✅ Run ID principal encontrado: {main_run_id}")

        print("\n⏳ Carregando os parâmetros do run principal...")
        client_mlflow = mlflow.tracking.MlflowClient() 
        main_run_info = client_mlflow.get_run(main_run_id)
        
        exog_cols_str = main_run_info.data.params.get("exog_cols")
        if exog_cols_str:
            exog_cols_loaded = ast.literal_eval(exog_cols_str)
            print(f"✅ Parâmetro 'exog_cols' carregado: {exog_cols_loaded}")
        else:
            print(f"❌ Parâmetro 'exog_cols' não encontrado no run principal.")

        print("\n⏳ Carregando o modelo LGBM do último fold e seus parâmetros...")
        nested_runs_df = mlflow.search_runs( 
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{main_run_id}'",
            order_by=["start_time DESC"]
        )
        
        if not nested_runs_df.empty:
            last_fold_run_id = nested_runs_df.iloc[0].run_id
            last_fold_run_info = client_mlflow.get_run(last_fold_run_id)
            model_params_last_fold: Dict[str, Any] = last_fold_run_info.data.params
            
            categ_cols_str = model_params_last_fold.get("categorical_feature")
            if categ_cols_str:
                if isinstance(categ_cols_str, str):
                    categ_cols_loaded = ast.literal_eval(categ_cols_str)
                else: 
                    categ_cols_loaded = categ_cols_str
                print(f"✅ Parâmetro 'categ_cols' carregado: {categ_cols_loaded}")
            else:
                print(f"❌ Parâmetro 'categorical_feature' (categ_cols) não encontrado no último fold.")

            last_fold_run_name_tag = last_fold_run_info.data.tags.get('mlflow.runName')
            if last_fold_run_name_tag:
                last_fold_number = int(last_fold_run_name_tag.split('_')[-1])
            else:
                print("❌ Não foi possível extrair o número do fold do nome do run aninhado.")
                last_fold_number = 1 
            
            loaded_model_uri = f"runs:/{last_fold_run_id}/{model_artifact_prefix}{last_fold_number}"
            model = mlflow.lightgbm.load_model(loaded_model_uri)
            print(f"✅ Modelo LGBM do último fold ({last_fold_number}) carregado com sucesso!")
        else:
            print("❌ Não foi possível carregar o modelo: nenhum fold aninhado encontrado.")
            return None, None, None, None, exog_cols_loaded, categ_cols_loaded
        
        print("\n⏳ Baixando e carregando os Dataframes finais...")
        
        download_dir = "./mlflow_downloaded_artifacts_temp" # Diretório temporário para download
        os.makedirs(download_dir, exist_ok=True)
        
        df_train_path = os.path.join(download_dir, df_artifact_names[0])
        df_val_path = os.path.join(download_dir, df_artifact_names[1])
        df_test_path = os.path.join(download_dir, df_artifact_names[2])
        
        client_mlflow.download_artifacts(run_id=main_run_id, path=df_artifact_names[0], dst_path=download_dir)
        client_mlflow.download_artifacts(run_id=main_run_id, path=df_artifact_names[1], dst_path=download_dir)
        client_mlflow.download_artifacts(run_id=main_run_id, path=df_artifact_names[2], dst_path=download_dir)
        
        df_train = pd.read_csv(df_train_path, index_col=0, parse_dates=True)
        df_val = pd.read_csv(df_val_path, index_col=0, parse_dates=True)
        df_test = pd.read_csv(df_test_path, index_col=0, parse_dates=True)
        
        print("✅ DataFrames carregados com sucesso!")

        # Limpa o diretório temporário de download
        for f in os.listdir(download_dir):
            os.remove(os.path.join(download_dir, f))
        os.rmdir(download_dir)
        print(f"✅ Diretório temporário '{download_dir}' removido.")

        return model, df_train, df_val, df_test, exog_cols_loaded, categ_cols_loaded
        
    except Exception as e:
        print(f"❌ Falha inesperada ao carregar artefatos/parâmetros: {e}")
        return None, None, None, None, None, None

