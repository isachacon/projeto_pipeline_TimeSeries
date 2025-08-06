import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List

def make_predictions(
    model: lgb.LGBMRegressor,
    df_to_predict: pd.DataFrame,
    exog_cols: List[str],
    categ_cols: List[str]
) -> pd.DataFrame:
    """
    Faz previsões em novos dados usando um modelo LightGBM treinado.

    Esta função replica as etapas de pré-processamento aplicadas no treino
    para garantir consistência.

    Args:
        model (lgb.LGBMRegressor): O modelo LightGBM treinado.
        df_to_predict (pd.DataFrame): O DataFrame com novos dados.
        exog_cols (List[str]): Lista de colunas a serem usadas como features.
        categ_cols (List[str]): Lista de colunas categóricas.

    Returns:
        pd.DataFrame: DataFrame original com a coluna 'prediction' adicionada.
    """
    print("\n--- 4. INFERÊNCIA DO MODELO ---")
    df_infer = df_to_predict.copy()

    if 'date' in df_infer.columns:
        df_infer['date'] = pd.to_datetime(df_infer['date'])
        df_infer = df_infer.set_index('date').sort_index()
    elif not isinstance(df_infer.index, pd.DatetimeIndex):
        print("⚠️ Aviso: DataFrame de inferência não tem índice DatetimeIndex ou coluna 'date'.")

    numeric_cols = df_infer.select_dtypes(include=np.number).columns
    df_infer[numeric_cols] = df_infer[numeric_cols].fillna(0)
    
    for col in categ_cols:
        if col in df_infer.columns:
            df_infer[col] = df_infer[col].astype('category')
        else:
            print(f"⚠️ Aviso: Coluna categórica '{col}' não encontrada no DataFrame de inferência.")

    X_predict = df_infer[exog_cols]
    
    df_infer['prediction'] = model.predict(X_predict)
    print("✅ Previsões realizadas com sucesso.")
    
    return df_infer

