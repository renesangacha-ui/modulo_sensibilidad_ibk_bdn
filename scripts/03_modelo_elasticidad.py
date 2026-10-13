#################################### Modelo: Sensibilidad ###############################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tModelo: Sensibilidad')
print('-'*100)

#____________________ 1) Paquetes ----

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import time
from pathlib import Path
import re
import warnings
import sys

from statsmodels.formula.api import ols

sys.path.append("..")
import src.graficos.graficos as graficos

#____________________ 2) Direcciones ----

# Dirección del Proyecto:
dir_p = str(Path.cwd().parent)

# Dirección de la carpeta con las bases de datos:
dir_b = os.path.join(dir_p, "data")

# Dirección de la carpeta de resultados:
dir_r = os.path.join(dir_p, "results")

# Dirección de la carpeta de scripts:
dir_s = os.path.join(dir_p, "scripts")

#____________________ 3) Lectura de archivos ----

datos_sensibilidad_mercado=pd.read_csv(dir_b+r'/processed/V01/base_sensibilidad_componente_mercado.csv',sep='|')
datos_sensibilidad_interna=pd.read_csv(dir_b+r'/processed/V01/base_sensibilidad_componente_interna.csv',sep='|')

#____________________ 4) Parametros ----

# En base al proceso de exploración de la información se han decidido los siguientes parametros
# De materialidad, disponibilidad de informacion, asi como de comportamiento observado

PERIODO_INICIO=202504
PERIODO_FIN=202604
CLUSTER_MODELAMIENTO={
     'cluster_1_sow_no_nulo': 'Agrupados',
     'cluster_2_sow_no_nulo': 'Socio en Crecimiento',
     'cluster_3_sow_no_nulo': 'Agrupados',
     'cluster_4_sow_no_nulo': 'Gran Cuenta por Conquistar',
     'cluster_5_sow_no_nulo': 'Agrupados',
     'cluster_6_sow_no_nulo': 'Cliente por Reactivar',     
     'cluster_1_nulo': 'Cliente Sin Deuda',
     'cluster_2_nulo': 'Cliente Sin Deuda',
     'cluster_3_nulo': 'Cliente Sin Deuda'
}

#____________________ 5) Funnel ----

datos_sensibilidad_interna=datos_sensibilidad_interna.query(f'periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}')
datos_sensibilidad_mercado=datos_sensibilidad_mercado.query(f'periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}')

#____________________ 6) Procesamiento ----

#~~~~ 6.1) Identificación: Cluster de Modelamiento ----

datos_sensibilidad_mercado['cluster_modelar']=datos_sensibilidad_mercado['cluster'].map(CLUSTER_MODELAMIENTO)
datos_sensibilidad_interna['cluster_modelar']=datos_sensibilidad_interna['cluster'].map(CLUSTER_MODELAMIENTO)

datos_sensibilidad_mercado.value_counts(['cluster_modelar','cluster']).reset_index()
datos_sensibilidad_interna.value_counts(['cluster_modelar','cluster']).reset_index()

#~~~~ 6.2) Construcción: Data Panel Sensibilidad ----

#---- 6.2.1) Construcción Data Panel: Sensibilidad Mercado ----

data_panel_sensibilidad_mercado = (
    datos_sensibilidad_mercado
    .groupby(['cluster_modelar', 'periodo'], observed=True)
    .agg(
        total_desembolsos=('desembolsos', 'sum'),
        monto_desembolsado=('monto_desembolsado', 'sum'),
        suma_tasa_x_monto_bucket=('suma_tasa_x_monto', 'sum')
    )
    .assign(
        tasa_mercado=lambda df: df['suma_tasa_x_monto_bucket'] / df['monto_desembolsado']
    )
    .reset_index()
)

data_panel_sensibilidad_mercado.value_counts('cluster_modelar')

#---- 6.2.2) Construcción Data Panel: Sensibilidad Interna ----

def promedio_ponderado(valores, pesos):
  mask = valores.notna() & pesos.notna() & (pesos > 0)
  if mask.any():
    return np.average(valores[mask], weights=pesos[mask])
  return np.nan

data_panel_sensibilidad_interna = (
    datos_sensibilidad_interna
    .groupby(["cluster_modelar", "periodo"], observed=True)
    .agg(
        leads=("llave_interna", "size"),
        monto_desembolsado_ibk=("monto_desembolsado", "sum"),
        monto_ofertado=("monto_ofertado_ajustado", "sum"),
        desembolsos=("indicador_exito", "sum"),
        tasa_desembolsada=(
            "tasa_desembolsada",
            lambda x: promedio_ponderado(
                x, datos_sensibilidad_interna.loc[x.index, "monto_desembolsado"]
            ),
        ),
        tasa_ofertada=(
            "tasa_min",
            lambda x: promedio_ponderado(
                x, datos_sensibilidad_interna.loc[x.index, "monto_ofertado_ajustado"]
            ),
        ),
    )
    .assign(
        efectividad=lambda df: np.where(
            df["monto_ofertado"] > 0, df["monto_desembolsado_ibk"] / df["monto_ofertado"], 0
        ),
        conversion=lambda df: np.where(
            df["leads"] > 0, df["desembolsos"] / df["leads"], 0
        ),
    )
    .reset_index()
)

data_panel_sensibilidad_interna.value_counts('cluster_modelar')

#---- 6.2.3) Construcción Data Panel: Sensibilidad ----

data_panel_sensibilidad=pd.merge(data_panel_sensibilidad_mercado,
         data_panel_sensibilidad_interna,
         how='outer',
         on=['cluster_modelar','periodo'])

#~~~~ 6.3) Creación de Indicadores ----

data_panel_sensibilidad['spread']=data_panel_sensibilidad['tasa_desembolsada']-data_panel_sensibilidad['tasa_mercado']

data_panel_sensibilidad.spread.describe()
data_panel_sensibilidad.efectividad.describe()

#____________________ 7) Modelamiento ----

#~~~~ 7.1) Creación: Variables Auxiliares ----

data_panel_sensibilidad[['cluster_modelar','periodo','efectividad','tasa_desembolsada','spread']]

data_panel_sensibilidad["log_efectividad"] = np.log(data_panel_sensibilidad["efectividad"])
data_panel_sensibilidad["log_tasa"] = np.log(data_panel_sensibilidad["tasa_desembolsada"])

#~~~~ 7.2) Modelo: Demeaning Secuencial + OLS por bucket ----

print('\n--- 2A. Demeaning Secuencial ---')

df_seq = data_panel_sensibilidad.copy()

#---- 7.2.1) Demeaning temporal ----

df_seq["log_efectividad_dm"] = df_seq["log_efectividad"] - df_seq.groupby("periodo")["log_efectividad"].transform("mean")
df_seq["log_tasa_dm"] = df_seq["log_tasa"] - df_seq.groupby("periodo")["log_tasa"].transform("mean")
df_seq["spread_dm"] = df_seq["spread"] - df_seq.groupby("periodo")["spread"].transform("mean")

#---- 7.2.2) Demeaning estructural por grupo ----

df_seq["log_efectividad_within"] = df_seq["log_efectividad_dm"] - df_seq.groupby("cluster_modelar")["log_efectividad_dm"].transform("mean")
df_seq["log_tasa_within"] = df_seq["log_tasa_dm"] - df_seq.groupby("cluster_modelar")["log_tasa_dm"].transform("mean")
df_seq["spread_within"] = df_seq["spread_dm"] - df_seq.groupby("cluster_modelar")["spread_dm"].transform("mean")

#---- 7.2.3) Modelamiento ----

#---- 7.2.3) Modelamiento (3 Modelos: Conjunto, Solo Tasa, Solo Spread) ----
resultados_seq = []
for grupo in df_seq["cluster_modelar"].unique():
    dg = df_seq[df_seq["cluster_modelar"] == grupo].copy()
    
    # Modelo 1: Conjunto (Tasa + Spread)
    m_conj = ols("log_efectividad_within ~ log_tasa_within + spread_within - 1", data=dg).fit()
    
    # Modelo 2: Solo Tasa
    m_tasa = ols("log_efectividad_within ~ log_tasa_within - 1", data=dg).fit()
    
    # Modelo 3: Solo Spread
    m_spread = ols("log_efectividad_within ~ spread_within - 1", data=dg).fit()
    
    resultados_seq.append({
        "grupo": grupo,
        # --- Modelo 1: Conjunto ---
        "conj_elast_tasa": m_conj.params["log_tasa_within"],
        "conj_pval_tasa": m_conj.pvalues["log_tasa_within"],
        "conj_sens_spread": m_conj.params["spread_within"],
        "conj_pval_spread": m_conj.pvalues["spread_within"],
        "conj_r2": m_conj.rsquared,
        
        # --- Modelo 2: Solo Tasa ---
        "tasa_elast": m_tasa.params["log_tasa_within"],
        "tasa_pval": m_tasa.pvalues["log_tasa_within"],
        "tasa_r2": m_tasa.rsquared,
        
        # --- Modelo 3: Solo Spread ---
        "spread_sens": m_spread.params["spread_within"],
        "spread_pval": m_spread.pvalues["spread_within"],
        "spread_r2": m_spread.rsquared
    })

df_res_seq = pd.DataFrame(resultados_seq)
print(df_res_seq.to_string(index=False))

#~~~~ 7.3) Modelo: Demeaning Simultáneo: (Two-Way) + OLS POR BUCKET ----

print('\n--- 2B. Demeaning Simultáneo ---')

df_sim = data_panel_sensibilidad.copy()

#---- 7.2.1) Demeaning ----

def demeaning_simultaneo(data, col):
    g_mean = data[col].mean()
    b_mean = data.groupby("cluster_modelar")[col].transform("mean")
    t_mean = data.groupby("periodo")[col].transform("mean")
    return data[col] - b_mean - t_mean + g_mean

df_sim["log_efectividad_dem"] = demeaning_simultaneo(df_sim, "log_efectividad")
df_sim["log_tasa_dem"] = demeaning_simultaneo(df_sim, "log_tasa")
df_sim["spread_dem"] = demeaning_simultaneo(df_sim, "spread")

#---- 7.2.2) Modelamiento ----

resultados_sim = []
for grupo in df_sim["cluster_modelar"].unique():
    dg = df_sim[df_sim["cluster_modelar"] == grupo].copy()
    
    # Modelo 1: Conjunto (Tasa + Spread)
    m_conj = ols("log_efectividad_dem ~ log_tasa_dem + spread_dem - 1", data=dg).fit()
    
    # Modelo 2: Solo Tasa
    m_tasa = ols("log_efectividad_dem ~ log_tasa_dem - 1", data=dg).fit()
    
    # Modelo 3: Solo Spread
    m_spread = ols("log_efectividad_dem ~ spread_dem - 1", data=dg).fit()
    
    resultados_sim.append({
        "grupo": grupo,
        # --- Modelo 1: Conjunto ---
        "conj_elast_tasa": m_conj.params["log_tasa_dem"],
        "conj_pval_tasa": m_conj.pvalues["log_tasa_dem"],
        "conj_sens_spread": m_conj.params["spread_dem"],
        "conj_pval_spread": m_conj.pvalues["spread_dem"],
        "conj_r2": m_conj.rsquared,
        
        # --- Modelo 2: Solo Tasa ---
        "tasa_elast": m_tasa.params["log_tasa_dem"],
        "tasa_pval": m_tasa.pvalues["log_tasa_dem"],
        "tasa_r2": m_tasa.rsquared,
        
        # --- Modelo 3: Solo Spread ---
        "spread_sens": m_spread.params["spread_dem"],
        "spread_pval": m_spread.pvalues["spread_dem"],
        "spread_r2": m_spread.rsquared
    })

df_res_sim = pd.DataFrame(resultados_sim)
print(df_res_sim.to_string(index=False))
