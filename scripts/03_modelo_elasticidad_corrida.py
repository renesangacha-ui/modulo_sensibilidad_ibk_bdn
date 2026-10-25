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

datos_sensibilidad_mercado=pd.read_csv(dir_b+r'/processed/V02/base_sensibilidad_componente_mercado.csv',sep='|')
datos_sensibilidad_interna=pd.read_csv(dir_b+r'/processed/V02/base_sensibilidad_componente_interna.csv',sep='|')

#____________________ 4) Parametros ----

# En base al proceso de exploración de la información se han decidido los siguientes parametros
# De materialidad, disponibilidad de informacion, asi como de comportamiento observado

PERIODO_INICIO=202504
PERIODO_FIN=202512

# CLUSTER_MODELAMIENTO={
#  'Gran Cuenta por Conquistar & Cliente': 'Grandes por fidelizar',
#  'Gran Cuenta por Conquistar & No Cliente': 'Grandes por fidelizar',
#  'Emprendedor Izipay & No Cliente': 'Emprendedor Izipay & No Cliente',
#  'Socio en Crecimiento & No Cliente': 'Fidelizados',
#  'Socio en Crecimiento & Cliente': 'Fidelizados',
#  'Maduro Multibanco & Cliente': 'Medianos por Fidelizar',
#  'Cliente por Reactivar & No Cliente': 'Medianos por Fidelizar',
#  'Cuenta Estratégica & Cliente': 'Grandes por fidelizar',
#  'Cliente Ancla & Cliente': 'Fidelizados',
#  'Emprendedor Izipay & Cliente': 'Sin historial crediticio',
#  'Nuevo en Expansión & Cliente': 'Sin historial crediticio',
#  'Cliente por Reactivar & Cliente':  'Medianos por Fidelizar',
#  'Próximo Recurrente & Cliente': 'Sin historial crediticio'
# }

CLUSTER_MODELAMIENTO={
 'Gran Cuenta por Conquistar & Cliente': 'Grandes por fidelizar & Cliente',
 'Gran Cuenta por Conquistar & No Cliente': 'No Cliente',
 'Emprendedor Izipay & No Cliente': 'No Cliente',
 'Socio en Crecimiento & No Cliente': 'No Cliente',
 'Socio en Crecimiento & Cliente': 'Fidelizado & Cliente',
 'Maduro Multibanco & Cliente': 'Medianos por Fidelizar & Cliente',
 'Cliente por Reactivar & No Cliente': 'No Cliente',
 'Cuenta Estratégica & Cliente': 'Grandes por fidelizar & Cliente',
 'Cliente Ancla & Cliente': 'Fidelizado & Cliente',
 'Emprendedor Izipay & Cliente': 'Sin historial crediticio & Cliente',
 'Nuevo en Expansión & Cliente': 'Sin historial crediticio & Cliente',
 'Cliente por Reactivar & Cliente':  'Medianos por Fidelizar & Cliente',
 'Próximo Recurrente & Cliente': 'Sin historial crediticio & Cliente'
}

MAPEO_TIPO_CLIENTE={
   1: 'Cliente',
   0: 'No Cliente'
}

#____________________ 5) Funnel ----

datos_sensibilidad_interna=datos_sensibilidad_interna.query(f'periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}')
datos_sensibilidad_mercado=datos_sensibilidad_mercado.query(f'periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}')

#____________________ 6) Procesamiento ----

#~~~~ 6.1) Identificación: Cluster de Modelamiento ----

datos_sensibilidad_mercado['cluster_comercial_new']=datos_sensibilidad_mercado['cluster_comercial_aperturado'].map(CLUSTER_MODELAMIENTO)
datos_sensibilidad_interna['cluster_comercial_new']=datos_sensibilidad_interna['cluster_comercial_aperturado'].map(CLUSTER_MODELAMIENTO)

datos_sensibilidad_mercado['tipo_cliente_new'] = datos_sensibilidad_mercado['FLG_CLIENTE'].map(MAPEO_TIPO_CLIENTE).fillna(datos_sensibilidad_mercado['FLG_CLIENTE'])
datos_sensibilidad_interna['tipo_cliente_new'] = datos_sensibilidad_interna['FLG_CLIENTE'].map(MAPEO_TIPO_CLIENTE).fillna(datos_sensibilidad_interna['FLG_CLIENTE'])

datos_sensibilidad_mercado.value_counts(['cluster_comercial_new','cluster']).reset_index()
datos_sensibilidad_interna.value_counts(['cluster_comercial_new','cluster']).reset_index()

datos_sensibilidad_mercado['cluster_modelar']=datos_sensibilidad_mercado['cluster_comercial_new'].astype(str)+' & '+ datos_sensibilidad_mercado['tipo_cliente_new'].astype(str)
datos_sensibilidad_interna['cluster_modelar']=datos_sensibilidad_interna['cluster_comercial_new'].astype(str)+' & '+ datos_sensibilidad_interna['tipo_cliente_new'].astype(str)

datos_sensibilidad_mercado['cluster_modelar']=datos_sensibilidad_mercado['cluster_comercial_new'].astype(str)
datos_sensibilidad_interna['cluster_modelar']=datos_sensibilidad_interna['cluster_comercial_new'].astype(str)

# ~~~~ 6.2) Construcción: Data Panel Sensibilidad ----

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

data_panel_sensibilidad.cluster_modelar.value_counts()

cluster_interes=[
'Fidelizado & Cliente',
'Grandes por fidelizar & Cliente',
'Medianos por Fidelizar & Cliente',
# 'No Cliente',
'Sin historial crediticio & Cliente'
]

data_panel_sensibilidad=data_panel_sensibilidad.query(f'cluster_modelar in {cluster_interes}')

#~~~~ 6.3) Creación de Indicadores ----

data_panel_sensibilidad['spread']=data_panel_sensibilidad['tasa_ofertada']-data_panel_sensibilidad['tasa_mercado']

data_panel_sensibilidad.spread.describe()
data_panel_sensibilidad.efectividad.describe()

#____________________ 7) Modelamiento ----

resultados_seq_global=pd.DataFrame()
resultados_sim_global=pd.DataFrame()

# for mes_cierre in data_panel_sensibilidad.periodo.drop_duplicates().to_list():
for mes_cierre in [202509,202510,202511,202512,202601,202602,202603,202604]:
   
    data_panel_sensibilidad_cierre=data_panel_sensibilidad.query(f'periodo<={mes_cierre}').copy()

    #~~~~ 7.1) Creación: Variables Auxiliares ----

    data_panel_sensibilidad_cierre["log_efectividad"] = np.log(data_panel_sensibilidad_cierre["efectividad"])
    data_panel_sensibilidad_cierre["log_tasa"] = np.log(data_panel_sensibilidad_cierre["tasa_ofertada"])

    #~~~~ 7.2) Modelo: Demeaning Secuencial + OLS por bucket ----

    print('\n--- 2A. Demeaning Secuencial ---')

    df_seq = data_panel_sensibilidad_cierre.copy()

    #---- 7.2.1) Demeaning temporal ----

    df_seq["log_efectividad_dm"] = df_seq["log_efectividad"] - df_seq.groupby("periodo")["log_efectividad"].transform("mean")
    df_seq["log_tasa_dm"] = df_seq["log_tasa"] - df_seq.groupby("periodo")["log_tasa"].transform("mean")
    df_seq["spread_dm"] = df_seq["spread"] - df_seq.groupby("periodo")["spread"].transform("mean")

    #---- 7.2.2) Demeaning estructural por grupo ----

    df_seq["log_efectividad_within"] = df_seq["log_efectividad_dm"] - df_seq.groupby("cluster_modelar")["log_efectividad_dm"].transform("mean")
    df_seq["log_tasa_within"] = df_seq["log_tasa_dm"] - df_seq.groupby("cluster_modelar")["log_tasa_dm"].transform("mean")
    df_seq["spread_within"] = df_seq["spread_dm"] - df_seq.groupby("cluster_modelar")["spread_dm"].transform("mean")

    #---- 7.2.3) Modelamiento ----

    resultados_seq = []
    for grupo in df_seq["cluster_modelar"].unique():
        dg = df_seq[df_seq["cluster_modelar"] == grupo].copy()
        
        # Modelo 1: Conjunto (Tasa + Spread)
        m_conj = ols("log_efectividad_within ~ log_tasa_within + spread_within - 1", data=dg).fit(cov_type="HC1")
        
        # Modelo 2: Solo Tasa
        m_tasa = ols("log_efectividad_within ~ log_tasa_within - 1", data=dg).fit(cov_type="HC1")
        
        # Modelo 3: Solo Spread
        m_spread = ols("log_efectividad_within ~ spread_within - 1", data=dg).fit(cov_type="HC1")
        
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

    df_sim = data_panel_sensibilidad_cierre.copy()

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
        m_conj = ols("log_efectividad_dem ~ log_tasa_dem + spread_dem - 1", data=dg).fit(cov_type="HC1")
        
        # Modelo 2: Solo Tasa
        m_tasa = ols("log_efectividad_dem ~ log_tasa_dem - 1", data=dg).fit(cov_type="HC1")
        
        # Modelo 3: Solo Spread
        m_spread = ols("log_efectividad_dem ~ spread_dem - 1", data=dg).fit(cov_type="HC1")
        
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

    df_res_seq['mes_cierre']=mes_cierre
    df_res_sim['mes_cierre']=mes_cierre
    
    resultados_seq_global=pd.concat([resultados_seq_global, df_res_seq])
    resultados_sim_global=pd.concat([resultados_sim_global, df_res_sim])    


# #____________________ 8) Visualizaciones ----

# data_panel_largo = data_panel_sensibilidad[['cluster_modelar', 'periodo', 'tasa_ofertada', 'tasa_desembolsada', 'tasa_mercado']].melt(
#     id_vars=['cluster_modelar', 'periodo'], 
#     var_name='tasa', 
#     value_name='valor'
# )

# for bucket in data_panel_largo.cluster_modelar.unique():

#     graficos.plot_evolutivo_multiserie_panel(
#         df=data_panel_largo.query(f"cluster_modelar=='{bucket}'"),
#         columna_periodo='periodo',
#         columna_grupo='tasa',
#         metrica='valor',
#         label_metrica='Tasa (%)',
#         titulo=f'Evolutivos de Tasa\n{bucket}'
#     )

resultados_seq_global['flg_significativo']=(resultados_seq_global['tasa_pval']<0.1)*1

graficos.plot_heatmap_cumplimiento(
    df_panel=resultados_seq_global,
    col_cluster='grupo',
    col_periodo='mes_cierre',
    col_flag='flg_significativo',
    titulo='Elasticidades Significativas', 
    subtitulo=f'Umbral mínimo de significancia: < {0.1:.0%}')

resultados_seq_global[['grupo','tasa_elast','tasa_r2']]

graficos.plot_evolutivo_multiserie_panel(
    df=resultados_seq_global,
    columna_periodo='mes_cierre',
    columna_grupo='grupo',
    metrica='tasa_elast',
    multiplicar_100=False,
    label_metrica='Elasticidad',
    titulo=(
            'Evolución: Elasticidad'
        )
)

graficos.plot_evolutivo_multiserie_panel(
    df=resultados_seq_global,
    columna_periodo='mes_cierre',
    columna_grupo='grupo',
    metrica='tasa_r2',
    multiplicar_100=True,
    label_metrica='R2 (%)',
    titulo=(
            'Evolución: R2'
        )
)

graficos.plot_evolutivo_multiserie_panel(
    df=resultados_seq_global,
    columna_periodo='mes_cierre',
    columna_grupo='grupo',
    metrica='tasa_pval',
    multiplicar_100=True,
    label_metrica='pvalue (%)',
    titulo=(
            'Evolución: P Valor'
        )
)

datos_sensibilidad_interna

data_panel_sensibilidad[['cluster_modelar','periodo','efectividad','tasa_ofertada']]



import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from linearmodels.panel import PanelOLS

df = data_panel_sensibilidad[
    [
        'cluster_modelar',
        'periodo',
        'efectividad',
        'tasa_ofertada'
    ]
].copy()

df = df.rename(columns={
    'cluster_modelar': 'bucket'
})

# Periodo YYYYMM numérico -> fecha mensual
df['mes'] = pd.to_datetime(
    df['periodo'].astype('Int64').astype(str),
    format='%Y%m',
    errors='coerce'
)

# Convertir variables a numéricas
for col in ['efectividad', 'tasa_ofertada']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar registros incompletos
df = df.dropna(
    subset=['bucket', 'mes', 'efectividad', 'tasa_ofertada']
).copy()

# Los logaritmos requieren valores estrictamente positivos
df = df.loc[
    (df['efectividad'] > 0) &
    (df['tasa_ofertada'] > 0)
].copy()

# Transformación logarítmica
df['log_efectividad'] = np.log(df['efectividad'])
df['log_tasa'] = np.log(df['tasa_ofertada'])

# Categorías como texto
df['bucket'] = df['bucket'].astype(str)
# df['mes'] = df['mes'].astype(str)

print(f'Observaciones utilizadas: {len(df):,}')
print(f'Clusters: {df["bucket"].nunique()}')
print(f'Meses: {df["mes"].nunique()}')

# ============================================================
# 2. CREAR INTERACCIONES (Seguro para nombres con espacios)
# ============================================================

buckets = sorted(df['bucket'].unique())
nombres_interaccion = {}

for i, bucket in enumerate(buckets):
    # Identificador numérico limpio para evitar errores por espacios
    nombre_columna = f'tasa_inter_{i}'

    df[nombre_columna] = (
        df['log_tasa'] *
        (df['bucket'] == bucket).astype(int)
    )

    nombres_interaccion[bucket] = nombre_columna

# ============================================================
# 3. MÉTODO 2: PanelOLS CON INTERACCIONES
# ============================================================

# Establecer índice MultiIndex (entidad, tiempo) requerido por linearmodels
df_panel = df.set_index(["bucket", "mes"]).sort_index()

columnas_x = list(nombres_interaccion.values())

modelo_panel = PanelOLS(
    dependent=df_panel["log_efectividad"],
    exog=df_panel[columnas_x],
    entity_effects=True,  # Efectos fijos por bucket
    time_effects=True,    # Efectos fijos por mes
    drop_absorbed=True,
)

# Ajuste del modelo con errores estándar agrupados por entidad
resultado_panel = modelo_panel.fit(
    cov_type="clustered", 
    cluster_entity=True
)

print('\n' + '=' * 70)
print('MODELO PANELOLS CON EFECTOS FIJOS E INTERACCIONES')
print('=' * 70)
print(resultado_panel.summary)

# ============================================================
# 4. EXTRAER ELASTICIDADES Y ESTADÍSTICOS POR CLUSTER
# ============================================================

intervalos = resultado_panel.conf_int()
resultados_elasticidad = []

for bucket, nombre_columna in nombres_interaccion.items():
    resultados_elasticidad.append({
        'cluster': bucket,  # Mantiene el nombre real con espacios
        'elasticidad': resultado_panel.params.get(nombre_columna, np.nan),
        'error_std': resultado_panel.std_errors.get(nombre_columna, np.nan),
        'p_value': resultado_panel.pvalues.get(nombre_columna, np.nan),
        'ic_inferior': intervalos.loc[nombre_columna, 'lower'],
        'ic_superior': intervalos.loc[nombre_columna, 'upper']
    })

tabla_elasticidades = pd.DataFrame(resultados_elasticidad)

tabla_elasticidades['significativo_10pct'] = (
    tabla_elasticidades['p_value'] < 0.10
)

tabla_elasticidades = tabla_elasticidades.sort_values(
    'cluster'
).reset_index(drop=True)

print('\n' + '=' * 70)
print('ELASTICIDADES POR CLUSTER (PanelOLS)')
print('=' * 70)
print(tabla_elasticidades.to_string(index=False))

# Diccionario directo con los betas y el nombre real del bucket
betas_panel = {
    bucket: resultado_panel.params[nombre_columna]
    for bucket, nombre_columna in nombres_interaccion.items()
}

# ============================================================
# 5. RESUMEN DE SIGNIFICANCIA
# ============================================================

print('\n' + '=' * 70)
print('RESUMEN DE SIGNIFICANCIA')
print('=' * 70)

for _, fila in tabla_elasticidades.iterrows():
    print(
        f'{fila["cluster"]}: '
        f'elasticidad={fila["elasticidad"]:.4f}, '
        f'p-value={fila["p_value"]:.4g}, '
        f'significativo al 10%={fila["significativo_10pct"]}'
    )