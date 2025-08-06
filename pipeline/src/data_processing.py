import pandas as pd
import duckdb
import warnings
import os
from typing import Tuple

# Importa SettingWithCopyWarning de pandas.errors
from pandas.errors import SettingWithCopyWarning 

def ingest_and_treat_data(
    con: duckdb.DuckDBPyConnection,
    path_clientes: str, 
    path_clima: str,     
    path_consumo: str    
) -> duckdb.DuckDBPyConnection:
    """
    Ingere os dados brutos de CSVs, trata os valores ausentes no clima
    e materializa as tabelas no DuckDB.

    Args:
        con (duckdb.DuckDBPyConnection): Objeto de conexão com o DuckDB.
        path_clientes (str): Caminho para o arquivo CSV de clientes.
        path_clima (str): Caminho para o arquivo CSV de clima.
        path_consumo (str): Caminho para o arquivo CSV de consumo.

    Returns:
        duckdb.DuckDBPyConnection: O objeto de conexão DuckDB com as tabelas materializadas.
    """
    print("--- 1. INGESTÃO E TRATAMENTO DE DADOS ---")

    try:
        cli_raw = con.execute(f"SELECT * FROM '{path_clientes}'").fetchdf()
        clima_raw = con.execute(f"SELECT * FROM '{path_clima}'").fetchdf()
        con_raw = con.execute(f"SELECT * FROM '{path_consumo}'").fetchdf()
        print("✅ Dados brutos ingeridos com sucesso.")
    except Exception as e:
        print(f"❌ Falha na ingestão de dados: {e}")
        raise

    print("\n--- Tratando valores ausentes no clima ---")
    clima_raw_sorted = clima_raw.sort_values(by=['region', 'date']).reset_index(drop=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SettingWithCopyWarning) 
        clima_raw_sorted['temperature'] = clima_raw_sorted.groupby('region')['temperature'].transform(lambda x: x.interpolate(method='linear'))
        # MODIFICAÇÃO: Usa .ffill() e .bfill() diretamente
        clima_raw_sorted['temperature'] = clima_raw_sorted.groupby('region')['temperature'].transform(lambda x: x.ffill())
        clima_raw_sorted['temperature'] = clima_raw_sorted.groupby('region')['temperature'].transform(lambda x: x.bfill())
    print("✅ Valores ausentes de temperatura tratados com sucesso.")

    con.execute("CREATE OR REPLACE TABLE clientes AS SELECT * FROM cli_raw")
    con.execute("CREATE OR REPLACE TABLE consumo AS SELECT * FROM con_raw")
    con.execute("CREATE OR REPLACE TABLE clima AS SELECT * FROM clima_raw_sorted")
    print("✅ Tabelas 'clientes', 'consumo' e 'clima' materializadas no DuckDB.")

    return con

def feature_engineering(
    con: duckdb.DuckDBPyConnection,
    output_dir: str 
) -> pd.DataFrame:
    """
    Executa todas as queries de feature engineering no DuckDB (cria tabelas de calendário,
    features e a tabela global final) e retorna o DataFrame final pronto para o treinamento.
    As tabelas intermediárias do DuckDB são salvas em 'output_dir'.

    Args:
        con (duckdb.DuckDBPyConnection): Objeto de conexão com o DuckDB.
        output_dir (str): Caminho para o diretório onde as tabelas DuckDB processadas serão salvas.

    Returns:
        pd.DataFrame: DataFrame final com todas as features e o target, pronto para o modelo.
    """
    print("\n--- 2. FEATURE ENGINEERING COM DUCKDB ---")

    os.makedirs(output_dir, exist_ok=True)

    query_calendario_sql = """
    CREATE OR REPLACE TABLE calendario AS
        SELECT
            date,
            MONTH(date) AS month,
            CAST(strftime(date, '%W') AS BIGINT) + 1 AS week_of_year,
            DAYOFWEEK(date) AS day_of_week,
            DAYOFYEAR(date) AS day_of_year,
            CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_final_semana,
            CASE WHEN (MONTH(date) = 12 AND DAY(date) >= 21) OR (MONTH(date) IN (1, 2)) OR (MONTH(date) = 3 AND DAY(date) < 21) THEN 'Verao'
                 WHEN (MONTH(date) = 3 AND DAY(date) >= 21) OR (MONTH(date) IN (4, 5)) OR (MONTH(date) = 6 AND DAY(date) < 21) THEN 'Outono'
                 WHEN (MONTH(date) = 6 AND DAY(date) >= 21) OR (MONTH(date) IN (7, 8)) OR (MONTH(date) = 9 AND DAY(date) < 21) THEN 'Inverno'
                 ELSE 'Primavera' END AS estacao,
            CASE WHEN date IN ('2023-01-01', '2023-04-21', '2023-05-01', '2023-09-07', '2023-10-12', '2023-11-02', '2023-11-15', '2023-12-25') THEN TRUE ELSE FALSE END AS is_feriado,
            CASE WHEN is_feriado = TRUE AND DAYOFWEEK(date) IN (1, 2, 4, 5) THEN TRUE ELSE FALSE END AS is_feriado_prolongado,
            CASE WHEN is_feriado = TRUE THEN 'Feriado'
                 WHEN is_feriado_prolongado = TRUE THEN 'Feriado Prolongado'
                 ELSE 'Nenhum' END AS tipo_feriado,
            CASE WHEN DAYOFWEEK(date) IN (0, 6) OR is_feriado = TRUE THEN FALSE ELSE TRUE END AS is_dia_util,
            DAYOFWEEK(date) as day_of_week_name,
            MONTH(date) as month
        FROM (SELECT DISTINCT date FROM clima)
        ORDER BY date;
    """
    con.execute(query_calendario_sql)
    print("✅ Tabela 'calendario' criada com sucesso.")
    con.execute(f"COPY calendario TO '{os.path.join(output_dir, 'calendario.parquet')}' (FORMAT PARQUET);")
    print(f"✅ Tabela 'calendario' salva em '{os.path.join(output_dir, 'calendario.parquet')}'.")


    query_create_features = """
    CREATE OR REPLACE TABLE features AS
    SELECT
        c.date,
        c.client_id,
        c.consumption_kwh AS total_consumption,
        cl.temperature as avg_temperature,
        cl.humidity as avg_humidity,
        cl.temperature * cl.humidity AS temp_humid_interaction,
        ROW_NUMBER() OVER (PARTITION BY c.client_id ORDER BY c.date) AS day_counter,
        LAG(c.consumption_kwh, 1) OVER (PARTITION BY c.client_id ORDER BY c.date) AS consumption_lag_1,
        LAG(c.consumption_kwh, 2) OVER (PARTITION BY c.client_id ORDER BY c.date) AS consumption_lag_2,
        LAG(c.consumption_kwh, 3) OVER (PARTITION BY c.client_id ORDER BY c.date) AS consumption_lag_3,
        LAG(c.consumption_kwh, 4) OVER (PARTITION BY c.client_id ORDER BY c.date) AS consumption_lag_4,
        LAG(c.consumption_kwh, 7) OVER (PARTITION BY c.client_id ORDER BY c.date) AS consumption_lag_7,
        LAG(c.consumption_kwh, 15) OVER (PARTITION BY c.client_id ORDER BY c.date) AS consumption_lag_15,
        AVG(c.consumption_kwh) OVER (PARTITION BY c.client_id ORDER BY c.date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_mean_3,
        STDDEV(c.consumption_kwh) OVER (PARTITION BY c.client_id ORDER BY c.date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS rolling_std_3,
        AVG(c.consumption_kwh) OVER (PARTITION BY c.client_id ORDER BY c.date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_mean_7,
        STDDEV(c.consumption_kwh) OVER (PARTITION BY c.client_id ORDER BY c.date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_std_7,
        c.consumption_kwh - LAG(c.consumption_kwh, 1) OVER (PARTITION BY c.client_id ORDER BY c.date) AS diff_lag_1
    FROM consumo AS c
    JOIN clientes AS cli ON c.client_id = cli.client_id
    LEFT JOIN clima AS cl ON c.date = cl.date AND cli.region = cl.region
    ORDER BY c.client_id, c.date;
    """
    con.execute(query_create_features)
    print("✅ Tabela 'features' criada com sucesso.")
    con.execute(f"COPY features TO '{os.path.join(output_dir, 'features.parquet')}' (FORMAT PARQUET);")
    print(f"✅ Tabela 'features' salva em '{os.path.join(output_dir, 'features.parquet')}'.")


    query_global = """
    CREATE OR REPLACE TABLE dados_lgbm_global AS
    SELECT
        f.date, f.client_id, f.total_consumption, f.avg_temperature, f.avg_humidity,
        f.temp_humid_interaction, f.day_counter, f.consumption_lag_1,
        f.consumption_lag_2, f.consumption_lag_3, f.consumption_lag_4,
        f.consumption_lag_7, f.consumption_lag_15, 
        f.rolling_mean_3, f.rolling_std_3, f.rolling_mean_7,
        f.rolling_std_7, f.diff_lag_1, 
        cal.day_of_week AS day_of_week_name, cal.month,
        cal.tipo_feriado
    FROM features AS f
    JOIN calendario AS cal ON f.date = cal.date
    ORDER BY f.client_id, f.date;
    """
    con.execute(query_global)
    print("✅ Tabela 'dados_lgbm_global' criada com sucesso.")
    con.execute(f"COPY dados_lgbm_global TO '{os.path.join(output_dir, 'dados_lgbm_global.parquet')}' (FORMAT PARQUET);")
    print(f"✅ Tabela 'dados_lgbm_global' salva em '{os.path.join(output_dir, 'dados_lgbm_global.parquet')}'.")


    df_global = con.execute("SELECT * FROM dados_lgbm_global").fetchdf()
    print("✅ DataFrame final carregado para o Pandas.")
    return df_global

