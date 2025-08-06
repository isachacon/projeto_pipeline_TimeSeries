import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import warnings
from io import StringIO
from contextlib import redirect_stderr
from typing import List, Tuple, Optional

# Importa funções auxiliares do módulo utils
from src.utils import calculate_metrics, create_offset

def train_lgbm_model(
    df: pd.DataFrame,
    mlflow_experiment_name: str,
    exog_cols: List[str],
    categ_cols: List[str],
    initial_train_size: str,
    validation_size: str,
    test_size: str,
    mlflow_tracking_uri: str 
) -> Tuple[Optional[lgb.LGBMRegressor], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Encapsula o pipeline completo de preparação de dados e treinamento de um
    modelo LightGBM global usando validação walk-forward, registrando os
    resultados no MLflow.

    Args:
        df (pd.DataFrame): O DataFrame completo com todas as features e o target.
        mlflow_experiment_name (str): O nome do experimento no MLflow.
        exog_cols (list): Lista de colunas a serem usadas como features.
        categ_cols (list): Lista de colunas que devem ser tratadas como categóricas.
        initial_train_size (str): Tamanho inicial da janela de treino (ex: '1 year').
        validation_size (str): Tamanho da janela de validação em cada fold (ex: '30 days').
        test_size (str): Tamanho final da janela de teste (ex: '30 days').
        mlflow_tracking_uri (str): O URI onde o MLflow deve salvar os logs e artefatos.
    
    Returns:
        tuple: (lgb.LGBMRegressor, pd.DataFrame, pd.DataFrame, pd.DataFrame)
                O modelo treinado no último fold e os DataFrames de treino, validação e teste
                do último fold, ou (None, None, None, None) em caso de falha.
    """
    print("--- 3. TREINAMENTO DO MODELO ---")
    
    # --- Transformações no Pandas ---
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    df[categ_cols] = df[categ_cols].astype('category')
    
    warnings.filterwarnings("ignore")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name) # Já está definido no main_pipeline.py, mas não custa garantir.

    print(f"\nMLflow está salvando os logs no URI: {mlflow.get_tracking_uri()}\n")
    
    model_global = None
    
    try:
        # MODIFICAÇÃO CRÍTICA: Adiciona nested=True para fazer deste um run aninhado
        with mlflow.start_run(run_name="Validação Walk-Forward", nested=True): 

            test_offset = create_offset(test_size)
            initial_train_offset = create_offset(initial_train_size)
            validation_offset = create_offset(validation_size)

            end_date_train_val = df.index.max() - test_offset
            df_test_global = df[df.index >= end_date_train_val]
            df_train_val = df[df.index < end_date_train_val]
            
            train_end_date = df_train_val.index[0] + initial_train_offset
            
            print(f'Train end date: {train_end_date}')
            print('Val size offset: ', validation_offset)
            print('Test init date: ', end_date_train_val)
            
            if df_train_val.empty:
                print("❌ Falha na validação walk-forward: df_train_val está vazio após separar o conjunto de teste.")
                mlflow.log_param("status", "falha_df_train_val_vazio")
                return None, None, None, None
                
            if train_end_date >= df_train_val.index[-1]:
                print(f"❌ Falha na validação walk-forward: O tamanho inicial de treino '{initial_train_size}' é muito grande para o dataset disponível. Ajuste `initial_train_size`.")
                mlflow.log_param("status", "falha_initial_train_size_grande")
                return None, None, None, None
                
            if train_end_date + validation_offset > df_train_val.index[-1]:
                print(f"❌ Falha na validação walk-forward: A primeira janela de validação '{validation_size}' excede os dados disponíveis após o treino inicial. Ajuste `validation_size` ou `initial_train_size`.")
                mlflow.log_param("status", "falha_val_window_excede")
                return None, None, None, None

            metrics_per_fold = []
            last_trained_model = None
            
            mlflow.log_param("walk_forward_initial_train_size", initial_train_size)
            mlflow.log_param("walk_forward_validation_size", validation_size)
            mlflow.log_param("test_size", test_size) 
            mlflow.log_param("exog_cols", exog_cols) 
            
            fold = 1
            while train_end_date + validation_offset <= df_train_val.index[-1]:
                print(f"💻 Processando Fold {fold}...")

                df_train_fold = df_train_val[df_train_val.index < train_end_date]
                df_val_fold = df_train_val[(df_train_val.index >= train_end_date) & (df_train_val.index < train_end_date + validation_offset)]
                
                X_train = df_train_fold[exog_cols]
                y_train = df_train_fold['total_consumption']
                X_val = df_val_fold[exog_cols]
                y_val = df_val_fold['total_consumption']
                
                if df_train_fold.empty or df_val_fold.empty:
                    print(f"❌ Fold {fold} pulado: janelas de dados vazias.")
                    break
                    
                with mlflow.start_run(run_name=f"Fold_{fold}", nested=True):
                    params = {
                        "random_state": 42,
                        "verbose": -1,
                        "categorical_feature": categ_cols 
                    }
                    mlflow.log_params(params)
                    mlflow.log_param("fold_train_start_date", df_train_fold.index.min())
                    mlflow.log_param("fold_train_end_date", df_train_fold.index.max())
                    mlflow.log_param("fold_val_start_date", df_val_fold.index.min())
                    mlflow.log_param("fold_val_end_date", df_val_fold.index.max())
                    
                    model = lgb.LGBMRegressor(**params)
                    with StringIO() as buf, redirect_stderr(buf):
                        model.fit(X_train, y_train,
                                  eval_set=[(X_val, y_val)],
                                  callbacks=[lgb.early_stopping(10, verbose=False)])

                    y_pred_val = model.predict(X_val)
                    metrics_val = calculate_metrics(y_val, y_pred_val, y_train=y_train)
                    for metric_name, metric_value in metrics_val.items():
                        mlflow.log_metric(f"{metric_name}", metric_value)
                    
                    metrics_per_fold.append(metrics_val)
                    print(f"✅ Fold {fold} concluído. MAE: {metrics_val['mae']:.4f}")
                    
                    mlflow.lightgbm.log_model(model, artifact_path=f"lgbm_model_fold_{fold}")

                train_end_date += validation_offset
                last_trained_model = model
                fold += 1

            if metrics_per_fold:
                avg_metrics = {
                    f'avg_{metric}': np.mean([m[metric] for m in metrics_per_fold])
                    for metric in metrics_per_fold[0].keys()
                }
                print("\n🏁 Validação walk-forward concluída.")
                print("Métricas Médias Finais:")
                for metric, value in avg_metrics.items():
                    print(f"- {metric.upper()}: {value:.4f}")
                    mlflow.log_metric(metric, value)
                
                if last_trained_model: 
                    mlflow.log_param("final_train_start_date", df_train_fold.index.min())
                    mlflow.log_param("final_train_end_date", df_train_fold.index.max())
                    mlflow.log_param("final_val_start_date", df_val_fold.index.min())
                    mlflow.log_param("final_val_end_date", df_val_fold.index.max())
                    
                    df_train_fold.to_csv("df_train_final.csv", index=True)
                    df_val_fold.to_csv("df_val_final.csv", index=True)
                    df_test_global.to_csv("df_test_final.csv", index=True)

                    mlflow.log_artifact("df_train_final.csv")
                    mlflow.log_artifact("df_val_final.csv")
                    mlflow.log_artifact("df_test_final.csv")

                    import os
                    os.remove("df_train_final.csv")
                    os.remove("df_val_final.csv")
                    os.remove("df_test_final.csv")
                    print("✅ DataFrames finais salvos como artefatos.")
                
            else:
                print("\n❌ Nenhuma fold processada com sucesso. Verifique os tamanhos das janelas.")
                mlflow.log_param("status", "falha_sem_folds")
                return None, None, None, None
                
    except Exception as e:
        print(f"❌ Falha no treinamento do modelo global: {e}")
        mlflow.log_param("status", "falha_geral")
        return None, None, None, None

    return last_trained_model, df_train_fold, df_val_fold, df_test_global
