import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
import re
from typing import Union, Dict

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series) -> Dict[str, float]:
    """
    Calcula MAE, RMSE, R², MAPE e MASE entre os valores reais e as previsões.
    
    Args:
        y_true (pd.Series): Valores reais.
        y_pred (pd.Series): Valores previstos.
        y_train (pd.Series): Valores de treino (usado para MASE).

    Returns:
        Dict[str, float]: Dicionário contendo as métricas calculadas.
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Garante que y_true_safe não tenha zeros para evitar divisão por zero no MAPE
    y_true_safe = np.where(y_true == 0, 1e-10, y_true) 
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # O MASE compara o erro do seu modelo com o erro de um modelo ingênuo
    metrics['mase'] = mean_absolute_scaled_error(y_true, y_pred, y_train=y_train)

    return metrics

def create_offset(size_str: str) -> Union[pd.DateOffset, pd.Timedelta]:
    """
    Cria um objeto de deslocamento de tempo (DateOffset ou Timedelta)
    a partir de uma string.
    
    Args:
        size_str (str): String com o tamanho do período (ex: '1 year', '30 days', '4 months').
    
    Returns:
        pd.DateOffset or pd.Timedelta: O objeto de deslocamento de tempo.
    """
    parts = size_str.lower().split()
    if len(parts) != 2:
        raise ValueError(f"Formato de string inválido para offset: '{size_str}'. Esperado 'valor unidade' (ex: '1 year').")

    value = int(parts[0])
    unit = parts[1]
    
    if unit in ['months', 'month']:
        return pd.DateOffset(months=value)
    elif unit in ['years', 'year']:
        return pd.DateOffset(years=value)
    else: # days, weeks, hours, etc.
        return pd.Timedelta(size_str)

