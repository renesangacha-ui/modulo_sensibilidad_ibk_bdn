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

datos=pd.read_csv(dir_b+r'/processed/V03/base_clientes_tratada.csv',sep='|')

#____________________ 4) Parametros ----

# En base al proceso de exploración de la información se han decidido los siguientes parametros
# De materialidad, disponibilidad de informacion, asi como de comportamiento observado

PERIODO_INICIO=202504
PERIODO_FIN=202604

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

#____________________ 6) Procesamiento ----

#~~~~ 6.1) Identificación: Cluster de Modelamiento ----

datos['cluster_modelar']=datos['cluster_comercial_aperturado'].map(CLUSTER_MODELAMIENTO)

datos.value_counts(['cluster_modelar','cluster_comercial_aperturado']).reset_index()

#~~~~ 6.2) Construcción: Data Panel Sensibilidad ----

funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

datos_oferta_relevante = datos.query("flg_exito_4 in @funnel_oferta_relevante").copy()

datos_oferta_relevante = (
    datos_oferta_relevante
    .sort_values(
        by=['llave_interna', 'flg_exito_4'],
        ascending=[True, True]
    )
    .drop_duplicates(subset=['llave_interna'], keep='first')
    .assign(
        indicador_exito=lambda df: (df['flg_exito_4'] == 'exito_desembolso_ibk_ct').astype(int),
    )
)

datos_oferta_relevante.query('tasa_desembolsada.notna() or monto_desembolsado.notna()').value_counts('indicador_exito',dropna=False)
datos_oferta_relevante.indicador_exito.sum()

datos.query("flg_exito_4 in ['exito_desembolso_ibk_ct']")

#---- 6.2.2) Construcción Data Panel: Sensibilidad Interna ----

def promedio_ponderado(valores, pesos):
  mask = valores.notna() & pesos.notna() & (pesos > 0)
  if mask.any():
    return np.average(valores[mask], weights=pesos[mask])
  return np.nan

data_panel_sensibilidad = (
    datos_oferta_relevante
    # .query('indicador_exito==1')
    .groupby(["cluster_modelar", "periodo"], observed=True)
    .agg(
        leads=("llave_interna", "size"),
        monto_desembolsado_ibk=("monto_desembolsado", "sum"),
        monto_ofertado=("monto_ofertado_ajustado", "sum"),
        desembolsos_ibk=("indicador_exito", "sum"),
        tasa_desembolsada=(
            "tasa_desembolsada",
            lambda x: promedio_ponderado(
                x, datos_oferta_relevante.loc[x.index, "monto_desembolsado"]
            ),
        ),
        tasa_ofertada=(
            "tasa_min",
            lambda x: promedio_ponderado(
                x, datos_oferta_relevante.loc[x.index, "monto_ofertado_ajustado"]
            ),
        ),
    )
    .assign(
        efectividad=lambda df: np.where(
            df["monto_ofertado"] > 0, df["monto_desembolsado_ibk"] / df["monto_ofertado"], 0
        ),
        conversion=lambda df: np.where(
            df["leads"] > 0, df["desembolsos_ibk"] / df["leads"], 0
        ),
    )
    .reset_index()
)

#---- 6.2.3) Construcción Data Panel: Sensibilidad ----

data_panel_sensibilidad.cluster_modelar.value_counts()

cluster_interes=[
'Fidelizado & Cliente',
'Grandes por fidelizar & Cliente',
'Medianos por Fidelizar & Cliente',
# 'No Cliente',
'Sin historial crediticio & Cliente'
]

data_panel_sensibilidad=data_panel_sensibilidad.query(f'cluster_modelar in {cluster_interes}')

data_panel_sensibilidad[['cluster_modelar','periodo','efectividad','tasa_ofertada']]


resultados_globales_rolling=pd.DataFrame()
data_panel_sensibilidad=data_panel_sensibilidad.query(f'{PERIODO_FIN}>=periodo>={PERIODO_INICIO}')







# import statsmodels.api as sm

# for periodo in [202509, 202510, 202511, 202512, 202601, 202602, 202603, 202604, 202605]:

#     df = data_panel_sensibilidad[
#         [
#             'cluster_modelar',
#             'periodo',
#             'efectividad',
#             'tasa_ofertada'
#         ]
#     ].copy()

#     df = df.query(f'periodo<={periodo}')
    
#     df = df.rename(columns={
#         'cluster_modelar': 'bucket'
#     })

#     # Periodo YYYYMM numérico -> fecha mensual
#     df['mes'] = pd.to_datetime(
#         df['periodo'].astype('Int64').astype(str),
#         format='%Y%m',
#         errors='coerce'
#     )

#     # Convertir variables a numéricas
#     for col in ['efectividad', 'tasa_ofertada']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # Eliminar registros incompletos
#     df = df.dropna(
#         subset=['bucket', 'mes', 'efectividad', 'tasa_ofertada']
#     ).copy()

#     # Los logaritmos requieren valores estrictamente positivos
#     df = df.loc[
#         (df['efectividad'] > 0) &
#         (df['tasa_ofertada'] > 0)
#     ].copy()

#     # Transformación logarítmica
#     df['log_efectividad'] = np.log(df['efectividad'])
#     df['log_tasa'] = np.log(df['tasa_ofertada'])

#     # Categorías como texto
#     df['bucket'] = df['bucket'].astype(str)

#     print(f'\n--- Procesando ventana hasta periodo: {periodo} ---')
#     print(f'Observaciones utilizadas: {len(df):,}')
#     print(f'Clusters: {df["bucket"].nunique()}')
#     print(f'Meses: {df["mes"].nunique()}')

#     # ============================================================
#     # 2. CREAR INTERACCIONES
#     # ============================================================

#     buckets = sorted(df['bucket'].unique())
#     nombres_interaccion = {}

#     for i, bucket in enumerate(buckets):
#         nombre_columna = f'tasa_inter_{i}'
#         df[nombre_columna] = (
#             df['log_tasa'] *
#             (df['bucket'] == bucket).astype(int)
#         )
#         nombres_interaccion[bucket] = nombre_columna

#     # ============================================================
#     # 3. DEMEANING MANUAL (DOBLE VÍA: BUCKET Y MES) + OLS HC1
#     # ============================================================
#     try:
#         columnas_x = list(nombres_interaccion.values())
#         vars_a_transformar = ['log_efectividad'] + columnas_x

#         # Aplicamos la transformación Within (demeaning) manual de dos vías:
#         # x_demeaned = x - media_bucket - media_mes + media_global
#         df_dm = df.copy()
#         for var in vars_a_transformar:
#             mean_global = df[var].mean()
#             mean_bucket = df.groupby('bucket')[var].transform('mean')
#             mean_mes = df.groupby('mes')[var].transform('mean')
#             df_dm[f'{var}_dm'] = df[var] - mean_bucket - mean_mes + mean_global

#         # Preparamos las matrices Y y X ya demeaned
#         y_dm = df_dm['log_efectividad_dm']
#         X_dm = df_dm[[f'{col}_dm' for col in columnas_x]]

#         # Corremos OLS estándar sobre los datos limpios con errores HC1
#         modelo_ols = sm.OLS(y_dm, X_dm)
#         resultado_ols = modelo_ols.fit(cov_type='HC3')

#         print('\n' + '=' * 70)
#         print(f'MODELO DEMEANING + OLS HC1 EXITOSO (Periodo: {periodo})')
#         print('=' * 70)
#         print(resultado_ols.summary())

#         # ============================================================
#         # 4. EXTRAER ELASTICIDADES Y ESTADÍSTICOS POR CLUSTER
#         # ============================================================

#         intervalos = resultado_ols.conf_int()
#         resultados_elasticidad = []

#         for bucket, nombre_columna in nombres_interaccion.items():
#             col_dm = f'{nombre_columna}_dm'
#             resultados_elasticidad.append({
#                 'cluster': bucket,
#                 'elasticidad': resultado_ols.params.get(col_dm, np.nan),
#                 'error_std': resultado_ols.bse.get(col_dm, np.nan),
#                 'p_value': resultado_ols.pvalues.get(col_dm, np.nan),
#                 'ic_inferior': intervalos.loc[col_dm, 0],
#                 'ic_superior': intervalos.loc[col_dm, 1]
#             })

#         tabla_elasticidades = pd.DataFrame(resultados_elasticidad)
#         tabla_elasticidades['significativo_10pct'] = (
#             tabla_elasticidades['p_value'] < 0.10
#         )
#         tabla_elasticidades = tabla_elasticidades.sort_values(
#             'cluster'
#         ).reset_index(drop=True)
        
#         tabla_elasticidades['periodo'] = periodo

#         # Métricas globales del modelo para este periodo
#         r2_overall = resultado_ols.rsquared
#         r2_adj = resultado_ols.rsquared_adj
#         f_stat = resultado_ols.fvalue
#         f_pval = resultado_ols.f_pvalue

#         tabla_elasticidades['r2_overall'] = r2_overall
#         tabla_elasticidades['r2_adj'] = r2_adj
#         tabla_elasticidades['f_stat'] = f_stat
#         tabla_elasticidades['f_pval'] = f_pval
        
#         resultados_globales_rolling = pd.concat(
#             [resultados_globales_rolling, tabla_elasticidades], 
#             ignore_index=True
#         )

#         # ============================================================
#         # 5. RESUMEN DE SIGNIFICANCIA
#         # ============================================================

#         print('\n' + '=' * 70)
#         print(f'RESUMEN DE SIGNIFICANCIA (Periodo: {periodo})')
#         print('=' * 70)

#         for _, fila in tabla_elasticidades.iterrows():
#             print(
#                 f'{fila["cluster"]}: '
#                 f'elasticidad={fila["elasticidad"]:.4f}, '
#                 f'p-value={fila["p_value"]:.4g}, '
#                 f'significativo al 10%={fila["significativo_10pct"]}'
#             )

#     except Exception as e:
#         print('\n' + '!' * 70)
#         print(f'⚠️ ADVERTENCIA: Falló la estimación para el periodo {periodo}')
#         print(f'Motivo / Error técnico: {e}')
#         print('Saltando este periodo y continuando con el siguiente...')
#         print('!' * 70 + '\n')
#         continue


# resultados_globales_rolling
resultados_globales_rolling=pd.DataFrame()

resultados_sim_global=pd.DataFrame()

# for mes_cierre in data_panel_sensibilidad.periodo.drop_duplicates().to_list():
for mes_cierre in [202512,202601,202602,202603,202604]:
   
    data_panel_sensibilidad_cierre=data_panel_sensibilidad.query(f'periodo<={mes_cierre}').copy()

    #~~~~ 7.1) Creación: Variables Auxiliares ----

    data_panel_sensibilidad_cierre["log_efectividad"] = np.log(data_panel_sensibilidad_cierre["efectividad"])
    data_panel_sensibilidad_cierre["log_tasa"] = np.log(data_panel_sensibilidad_cierre["tasa_ofertada"])

    #~~~~ 7.2) Modelo: Demeaning Secuencial + OLS por bucket ----

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

    #---- 7.2.2) Modelamiento ----

    resultados_sim = []
    for grupo in df_sim["cluster_modelar"].unique():
        dg = df_sim[df_sim["cluster_modelar"] == grupo].copy()
        
        # Modelo 2: Solo Tasa
        m_tasa = ols("log_efectividad_dem ~ log_tasa_dem - 1", data=dg).fit(cov_type="HC1")
        
        resultados_sim.append({
            "grupo": grupo,

            # --- Modelo 2: Solo Tasa ---
            "tasa_elast": m_tasa.params["log_tasa_dem"],
            "tasa_pval": m_tasa.pvalues["log_tasa_dem"],
            "tasa_r2": m_tasa.rsquared,
        })

    df_res_sim = pd.DataFrame(resultados_sim)

    df_res_sim['mes_cierre']=mes_cierre
    
    resultados_sim_global=pd.concat([resultados_sim_global, df_res_sim])    



import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

# Asegúrate de inicializar tu DataFrame acumulado antes del bucle si no lo tienes:
resultados_globales_rolling = pd.DataFrame()

for periodo in [202512, 202601, 202602, 202603,202604]:
# for periodo in data_panel_sensibilidad.periodo.drop_duplicates().to_list():

    df = data_panel_sensibilidad[
        [
            'cluster_modelar',
            'periodo',
            'efectividad',
            'tasa_ofertada'
        ]
    ].copy()

    df = df.query(f'periodo<={periodo}')

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

    print(f'\n--- Procesando ventana hasta periodo: {periodo} ---')
    print(f'Observaciones utilizadas: {len(df):,}')
    print(f'Clusters: {df["bucket"].nunique()}')
    print(f'Meses: {df["mes"].nunique()}')

    # ============================================================
    # 2. CREAR INTERACCIONES
    # ============================================================

    buckets = sorted(df['bucket'].unique())
    nombres_interaccion = {}

    for i, bucket in enumerate(buckets):
        nombre_columna = f'tasa_inter_{i}'
        df[nombre_columna] = (
            df['log_tasa'] *
            (df['bucket'] == bucket).astype(int)
        )
        nombres_interaccion[bucket] = nombre_columna

    # ============================================================
    # 3. MÉTODO 2: PanelOLS CON INTERACCIONES (Protegido)
    # ============================================================
    try:
        df_panel = df.set_index(["bucket", "mes"]).sort_index()
        columnas_x = list(nombres_interaccion.values())

        modelo_panel = PanelOLS(
            dependent=df_panel["log_efectividad"],
            exog=df_panel[columnas_x],
            entity_effects=True,  # Efectos fijos por bucket
            time_effects=True,    # Efectos fijos por mes
            drop_absorbed=True,
        )

        resultado_panel = modelo_panel.fit(
            cov_type="clustered", 
            cluster_entity=True,
            cluster_time=True
        )

        print('\n' + '=' * 70)
        print(f'MODELO PANELOLS EXITOSO (Periodo: {periodo})')
        print('=' * 70)
        print(resultado_panel.summary)

        # ============================================================
        # 4. EXTRAER ELASTICIDADES Y ESTADÍSTICOS POR CLUSTER
        # ============================================================

        intervalos = resultado_panel.conf_int()
        resultados_elasticidad = []

        for bucket, nombre_columna in nombres_interaccion.items():
            resultados_elasticidad.append({
                'cluster': bucket,
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
        
        tabla_elasticidades['periodo'] = periodo

        # Métricas globales del modelo para este periodo
        r2_overall = resultado_panel.rsquared
        r2_within = resultado_panel.rsquared_within
        r2_between = resultado_panel.rsquared_between
        n_obs = resultado_panel.nobs
        f_stat = resultado_panel.f_statistic.stat
        f_pval = resultado_panel.f_statistic.pval

        # Puedes asignarlas a cada fila de tu DataFrame de resultados:
        tabla_elasticidades['r2_overall'] = r2_overall
        tabla_elasticidades['r2_within'] = r2_within
        tabla_elasticidades['r2_between'] = r2_between
        tabla_elasticidades['f_stat'] = f_stat
        tabla_elasticidades['f_pval'] = f_pval
        
        resultados_globales_rolling = pd.concat(
            [resultados_globales_rolling, tabla_elasticidades], 
            ignore_index=True
        )

        # ============================================================
        # 5. RESUMEN DE SIGNIFICANCIA
        # ============================================================

        print('\n' + '=' * 70)
        print(f'RESUMEN DE SIGNIFICANCIA (Periodo: {periodo})')
        print('=' * 70)

        for _, fila in tabla_elasticidades.iterrows():
            print(
                f'{fila["cluster"]}: '
                f'elasticidad={fila["elasticidad"]:.4f}, '
                f'p-value={fila["p_value"]:.4g}, '
                f'significativo al 10%={fila["significativo_10pct"]}'
            )

    except Exception as e:
        print('\n' + '!' * 70)
        print(f'⚠️ ADVERTENCIA: Falló la estimación para el periodo {periodo}')
        print(f'Motivo / Error técnico: {e}')
        print('Saltando este periodo y continuando con el siguiente...')
        print('!' * 70 + '\n')
        continue

resultados_globales_rolling

resultados_globales_rolling['flg_significativo']=(resultados_globales_rolling['significativo_10pct'])*1

graficos.plot_heatmap_cumplimiento(
    df_panel=resultados_globales_rolling,
    col_cluster='cluster',
    col_periodo='periodo',
    col_flag='flg_significativo',
    titulo='Elasticidades Significativas', 
    subtitulo=f'Umbral mínimo de significancia: < {0.1:.0%}')

graficos.plot_evolutivo_multiserie_panel(
    df=resultados_globales_rolling,
    columna_periodo='periodo',
    columna_grupo='cluster',
    metrica='elasticidad',
    multiplicar_100=False,
    label_metrica='Elasticidad',
    titulo=(
            'Evolución: Elasticidad'
        )
)

graficos.plot_evolutivo_multiserie_panel(
    df=resultados_globales_rolling,
    columna_periodo='periodo',
    columna_grupo='cluster',
    metrica='p_value',
    multiplicar_100=True,
    label_metrica='pvalue (%)',
    titulo=(
            'Evolución: P Valor'
        )
)







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PREPARAR RESULTADOS
# ============================================================

df_control = resultados_globales_rolling.copy()

df_control['periodo'] = pd.to_numeric(
    df_control['periodo'], errors='coerce'
)

df_control['elasticidad'] = pd.to_numeric(
    df_control['elasticidad'], errors='coerce'
)

df_control = (
    df_control
    .dropna(subset=['cluster', 'periodo', 'elasticidad'])
    .sort_values(['cluster', 'periodo'])
    .copy()
)

# Constantes clásicas I-MR para rangos móviles de 2 observaciones
D4 = 3.267
A2_MR = 2.66

resumen_control = []

# ============================================================
# 2. GRAFICAR I-MR POR CLUSTER
# ============================================================

for cluster, data in df_control.groupby('cluster'):

    data = data.sort_values('periodo').copy()

    # Se necesitan al menos dos estimaciones para calcular MR
    if len(data) < 2:
        print(f'{cluster}: insuficientes estimaciones para I-MR.')
        continue

    # --------------------------------------------------------
    # Gráfico I: elasticidades individuales
    # --------------------------------------------------------

    data['MR'] = data['elasticidad'].diff().abs()

    promedio = data['elasticidad'].mean()
    mr_promedio = data['MR'].mean()

    # Límites de control del gráfico I
    lsc_i = promedio + A2_MR * mr_promedio
    lic_i = promedio - A2_MR * mr_promedio

    # Límites de control del gráfico MR
    lc_mr = mr_promedio
    lsc_mr = D4 * mr_promedio
    lic_mr = 0.0

    # Alertas exploratorias
    data['alerta_I'] = (
        (data['elasticidad'] > lsc_i) |
        (data['elasticidad'] < lic_i)
    )

    data['alerta_MR'] = data['MR'] > lsc_mr

    # Periodos para el eje X
    etiquetas = (
        pd.to_datetime(
            data['periodo'].astype('Int64').astype(str),
            format='%Y%m',
            errors='coerce'
        ).dt.strftime('%b-%y')
    )

    x = np.arange(len(data))

    # ========================================================
    # GRÁFICO 1: I - ELASTICIDAD
    # ========================================================

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        x,
        data['elasticidad'],
        marker='o',
        linewidth=1.8,
        label='Elasticidad rolling'
    )

    ax.axhline(
        promedio,
        linestyle='-',
        linewidth=1.5,
        label=f'LC = {promedio:.3f}'
    )

    ax.axhline(
        lsc_i,
        linestyle='--',
        linewidth=1.5,
        label=f'LSC = {lsc_i:.3f}'
    )

    ax.axhline(
        lic_i,
        linestyle='--',
        linewidth=1.5,
        label=f'LIC = {lic_i:.3f}'
    )

    # Destacar puntos fuera de los límites
    alertas_i = data['alerta_I'].to_numpy()

    if alertas_i.any():
        ax.scatter(
            x[alertas_i],
            data.loc[data['alerta_I'], 'elasticidad'],
            marker='x',
            s=100,
            linewidths=2,
            label='Alerta I'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=45)
    ax.set_title(f'Gráfico I - Estabilidad de elasticidad | {cluster}')
    ax.set_xlabel('Corte de información')
    ax.set_ylabel('Elasticidad')
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ========================================================
    # GRÁFICO 2: MR - CAMBIO ENTRE VENTANAS
    # ========================================================

    fig, ax = plt.subplots(figsize=(11, 5))

    # El primer corte no tiene MR: no hay ventana anterior
    mask_mr = data['MR'].notna()

    x_mr = x[mask_mr.to_numpy()]
    mr = data.loc[mask_mr, 'MR'].to_numpy()
    alertas_mr = data.loc[mask_mr, 'alerta_MR'].to_numpy()

    ax.plot(
        x_mr,
        mr,
        marker='o',
        linewidth=1.8,
        label='Cambio absoluto (MR)'
    )

    ax.axhline(
        lc_mr,
        linestyle='-',
        linewidth=1.5,
        label=f'LC = {lc_mr:.3f}'
    )

    ax.axhline(
        lsc_mr,
        linestyle='--',
        linewidth=1.5,
        label=f'LSC = {lsc_mr:.3f}'
    )

    ax.axhline(
        lic_mr,
        linestyle='--',
        linewidth=1,
        label=f'LIC = {lic_mr:.3f}'
    )

    if alertas_mr.any():
        ax.scatter(
            x_mr[alertas_mr],
            mr[alertas_mr],
            marker='x',
            s=100,
            linewidths=2,
            label='Alerta MR'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=45)
    ax.set_title(f'Gráfico MR - Cambios entre ventanas | {cluster}')
    ax.set_xlabel('Corte de información')
    ax.set_ylabel('Cambio absoluto de elasticidad')
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ========================================================
    # 3. RESUMEN DE ALERTAS
    # ========================================================

    data['periodo_anterior'] = data['periodo'].shift(1)

    for _, fila in data.iterrows():
        resumen_control.append({
            'cluster': cluster,
            'periodo': fila['periodo'],
            'elasticidad': fila['elasticidad'],
            'MR': fila['MR'],
            'LC_I': promedio,
            'LIC_I': lic_i,
            'LSC_I': lsc_i,
            'LC_MR': lc_mr,
            'LSC_MR': lsc_mr,
            'alerta_I': fila['alerta_I'],
            'alerta_MR': fila['alerta_MR']
        })

# Tabla final de resultados y alertas
tabla_control = pd.DataFrame(resumen_control)

display(
    tabla_control.sort_values(['cluster', 'periodo'])
)

print('\nResumen de alertas:')
if not tabla_control.empty:
    print('Alertas gráfico I:', int(tabla_control['alerta_I'].sum()))
    print('Alertas gráfico MR:', int(tabla_control['alerta_MR'].sum()))



# resultados_sim_global

# resultados_sim_global['flg_significativo']=(resultados_sim_global['tasa_pval']<0.10)*1

# graficos.plot_heatmap_cumplimiento(
#     df_panel=resultados_sim_global,
#     col_cluster='grupo',
#     col_periodo='mes_cierre',
#     col_flag='flg_significativo',
#     titulo='Elasticidades Significativas', 
#     subtitulo=f'Umbral mínimo de significancia: < {0.1:.0%}')

# graficos.plot_evolutivo_multiserie_panel(
#     df=resultados_sim_global,
#     columna_periodo='mes_cierre',
#     columna_grupo='grupo',
#     metrica='tasa_elast',
#     multiplicar_100=False,
#     label_metrica='Elasticidad',
#     titulo=(
#             'Evolución: Elasticidad'
#         )
# )

# graficos.plot_evolutivo_multiserie_panel(
#     df=resultados_sim_global,
#     columna_periodo='mes_cierre',
#     columna_grupo='grupo',
#     metrica='tasa_pval',
#     multiplicar_100=True,
#     label_metrica='pvalue (%)',
#     titulo=(
#             'Evolución: P Valor'
#         )
# )


# data_panel_sensibilidad


# data_panel_largo = data_panel_sensibilidad[['cluster_modelar', 'periodo', 'tasa_ofertada', 'tasa_desembolsada']].melt(
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






# import numpy as np
# import pandas as pd
# import statsmodels.formula.api as smf

# # Inicializar el DataFrame global para acumular los resultados históricos del rolling
# resultados_globales_rolling = pd.DataFrame()

# for periodo in [202512, 202601, 202602, 202603,202604]:
#     # 1. PREPARACIÓN Y LIMPIEZA DE DATOS
#     df = data_panel_sensibilidad[
#         [
#             'cluster_modelar',
#             'periodo',
#             'efectividad',
#             'tasa_ofertada'
#         ]
#     ].copy()

#     df = df.query(f'periodo<={periodo}')

#     df = df.rename(columns={
#         'cluster_modelar': 'bucket'
#     })

#     # Convertir periodo numérico YYYYMM a string plano para evitar problemas de formato en OLS
#     df['mes'] = df['periodo'].astype(int).astype(str)

#     # Convertir variables métricas a numéricas
#     for col in ['efectividad', 'tasa_ofertada']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # Eliminar registros incompletos
#     df = df.dropna(subset=['bucket', 'mes', 'efectividad', 'tasa_ofertada']).copy()

#     # Filtro para asegurar valores estrictamente positivos para los logaritmos
#     df = df.loc[(df['efectividad'] > 0) & (df['tasa_ofertada'] > 0)].copy()

#     # Aplicar transformación logarítmica
#     df['log_efectividad'] = np.log(df['efectividad'])
#     df['log_tasa'] = np.log(df['tasa_ofertada'])

#     # Asegurar tipos categóricos como texto plano
#     df['bucket'] = df['bucket'].astype(str)

#     print(f'\n--- Procesando ventana OLS hasta periodo: {periodo} ---')
#     print(f'Observaciones utilizadas: {len(df):,}')
#     print(f'Clusters (Buckets): {df["bucket"].nunique()}')
#     print(f'Meses: {df["mes"].nunique()}')

#     # ============================================================
#     # 2. CREACIÓN DE INTERACCIONES EXPLICITAS (TASA * BUCKET)
#     # ============================================================
#     buckets = sorted(df['bucket'].unique())
#     nombres_interaccion = []

#     for bucket in buckets:
#         # Reemplazar caracteres especiales para evitar errores en la fórmula OLS
#         nombre_columna = f'tasa_inter_{str(bucket).replace(" ", "_").replace("-", "_")}'
#         df[nombre_columna] = df['log_tasa'] * (df['bucket'] == bucket).astype(int)
#         nombres_interaccion.append(nombre_columna)

#     # ============================================================
#     # 3. CONSTRUCCIÓN DE LA FÓRMULA CON DUMMIES EXPLICITAS
#     # ============================================================
#     # C(bucket) crea dummies para efectos fijos de entidad
#     # C(mes) crea dummies para efectos fijos de tiempo
#     # Se elimina el intercepto implícito (- 1) para evitar multicolinealidad perfecta
#     lista_exogenas = nombres_interaccion + ['C(bucket)', 'C(mes)']
#     formula = f"log_efectividad ~ {' + '.join(lista_exogenas)} - 1"

#     # ============================================================
#     # 4. ESTIMACIÓN DEL MODELO OLS CON ERRORES ESTÁNDAR CLUSTERIZADOS
#     # ============================================================
#     try:
#         # Ajustamos el modelo OLS
#         modelo_ols = smf.ols(formula, data=df)
        
#         # Clusterizamos los errores estándar por bucket (entidad) para corregir autocorrelación
#         resultado_ols = modelo_ols.fit(
#             cov_type='cluster',
#             cov_kwds={'groups': df['bucket']}
#         )

#         print('\n' + '=' * 70)
#         print(f'MODELO OLS CON DUMMIES EXITOSO (Periodo: {periodo})')
#         print('=' * 70)
        
#         # ============================================================
#         # 5. EXTRACCIÓN GLOBAL DE PARÁMETROS Y P-VALORES
#         # ============================================================
#         # Extraemos intervalos de confianza de todas las variables
#         intervalos = resultado_ols.conf_int()
        
#         # Creamos un DataFrame con el reporte completo de la iteración actual
#         df_metricas_iteracion = pd.DataFrame({
#             'variable': resultado_ols.params.index,
#             'coeficiente': resultado_ols.params.values,
#             'error_std': resultado_ols.bse.values,
#             't_stat': resultado_ols.tvalues.values,
#             'p_value': resultado_ols.pvalues.values,
#             'ic_inferior': intervalos[0].values,
#             'ic_superior': intervalos[1].values
#         })

#         # Añadir metadatos de control y bondad de ajuste
#         df_metricas_iteracion['periodo_corte'] = periodo
#         df_metricas_iteracion['n_obs'] = resultado_ols.nobs
#         df_metricas_iteracion['r2_ajustado'] = resultado_ols.rsquared_adj
#         df_metricas_iteracion['f_stat'] = resultado_ols.fvalue
#         df_metricas_iteracion['f_pval'] = resultado_ols.f_pvalue
#         df_metricas_iteracion['significativo_10pct'] = df_metricas_iteracion['p_value'] < 0.10

#         # Identificar el tipo de variable estimada para facilitar el análisis posterior
#         df_metricas_iteracion['tipo_parametro'] = np.where(
#             df_metricas_iteracion['variable'].str.contains('tasa_inter_'), 'Elasticidad_Cluster',
#             np.where(df_metricas_iteracion['variable'].str.contains('C\(bucket\)'), 'Efecto_Fijo_Bucket', 'Efecto_Fijo_Mes')
#         )

#         # Acumular los resultados de esta iteración en el DataFrame maestro
#         resultados_globales_rolling = pd.concat(
#             [resultados_globales_rolling, df_metricas_iteracion], 
#             ignore_index=True
#         )

#         # Imprimir un resumen rápido de las elasticidades calculadas en pantalla
#         print(df_metricas_iteracion[df_metricas_iteracion['tipo_parametro'] == 'Elasticidad_Cluster'][
#             ['variable', 'coeficiente', 'p_value', 'significativo_10pct']
#         ].to_string(index=False))

#     except Exception as e:
#         print('\n' + '!' * 70)
#         print(f'⚠️ ADVERTENCIA: Falló la estimación para el periodo {periodo}')
#         print(f'Motivo técnico: {e}')
#         print('Saltando este periodo y continuando con el siguiente...')
#         print('!' * 70 + '\n')
#         continue








# import numpy as np
# import pandas as pd
# from linearmodels.panel import PanelOLS

# resultados_globales_rolling = pd.DataFrame()

# for periodo in [202601,202602,202603,202604]:

#     df = data_panel_sensibilidad[
#         [
#             'cluster_modelar',
#             'periodo',
#             'efectividad',
#             'tasa_ofertada'
#         ]
#     ].copy()

#     df = df.query(f'periodo<={periodo}')
#     df = df.rename(columns={'cluster_modelar': 'bucket'})

#     # Periodo YYYYMM numérico -> fecha mensual
#     df['mes'] = pd.to_datetime(
#         df['periodo'].astype('Int64').astype(str),
#         format='%Y%m',
#         errors='coerce'
#     )

#     # Convertir variables a numéricas
#     for col in ['efectividad', 'tasa_ofertada']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # Limpieza de nulos y valores no positivos para los logaritmos
#     df = df.dropna(subset=['bucket', 'mes', 'efectividad', 'tasa_ofertada']).copy()
#     df = df.loc[(df['efectividad'] > 0) & (df['tasa_ofertada'] > 0)].copy()

#     # Transformación logarítmica
#     df['log_efectividad'] = np.log(df['efectividad'])
#     df['log_tasa'] = np.log(df['tasa_ofertada'])
#     df['bucket'] = df['bucket'].astype(str)

#     print(f'\n--- Procesando ventana hasta periodo: {periodo} ---')
#     print(f'Observaciones utilizadas: {len(df):,}')
#     print(f'Clusters: {df["bucket"].nunique()}')
#     print(f'Meses: {df["mes"].nunique()}')

#     # ============================================================
#     # CREACIÓN DE INTERACCIONES SEGÚN LA FORMA FUNCIONAL DE LA IMAGEN
#     # ============================================================
#     buckets = sorted(df['bucket'].unique())
#     nombres_interaccion = {}
#     columnas_x = []

#     for i, bucket in enumerate(buckets):
#         # 1. Sumatoria 1: beta_j * ln(T_kt) * 1(k = j) -> Efecto base por bucket
#         col_beta = f'tasa_beta_{i}'
#         df[col_beta] = df['log_tasa'] * (df['bucket'] == bucket).astype(int)
#         columnas_x.append(col_beta)
#         nombres_interaccion[f'{bucket}_beta'] = col_beta

#         # 2. Sumatoria 2: delta_j * ln(T_kt) * 1(k = j) * 1(t = 202604) -> Efecto específico en 202604
#         col_delta = f'tasa_delta_{i}'
#         df[col_delta] = df['log_tasa'] * (df['bucket'] == bucket).astype(int) * (df['periodo'] >= 202601).astype(int)
#         columnas_x.append(col_delta)
#         nombres_interaccion[f'{bucket}_delta'] = col_delta

#     # ============================================================
#     # ESTIMACIÓN PANELOLS
#     # ============================================================
#     try:
#         df_panel = df.set_index(["bucket", "mes"]).sort_index()

#         modelo_panel = PanelOLS(
#             dependent=df_panel["log_efectividad"],
#             exog=df_panel[columnas_x],
#             entity_effects=True,  # Alfa_k (Efectos fijos por bucket)
#             time_effects=True,    # Lambda_t (Efectos fijos por mes)
#             drop_absorbed=True,
#         )

#         resultado_panel = modelo_panel.fit(
#             cov_type="clustered", 
#             cluster_entity=True,
#             cluster_time=True
#         )

#         print('\n' + '=' * 70)
#         print(f'MODELO PANELOLS EXITOSO (Periodo: {periodo})')
#         print('=' * 70)
#         print(resultado_panel.summary)

#         # Extracción de parámetros para betas y deltas
#         intervalos = resultado_panel.conf_int()
#         resultados_elasticidad = []

#         for etiqueta, nombre_columna in nombres_interaccion.items():
#             resultados_elasticidad.append({
#                 'parametro_modelo': etiqueta,
#                 'coeficiente': resultado_panel.params.get(nombre_columna, np.nan),
#                 'error_std': resultado_panel.std_errors.get(nombre_columna, np.nan),
#                 'p_value': resultado_panel.pvalues.get(nombre_columna, np.nan),
#                 'ic_inferior': intervalos.loc[nombre_columna, 'lower'] if nombre_columna in intervalos.index else np.nan,
#                 'ic_superior': intervalos.loc[nombre_columna, 'upper'] if nombre_columna in intervalos.index else np.nan
#             })

#         tabla_elasticidades = pd.DataFrame(resultados_elasticidad)
#         tabla_elasticidades['significativo_10pct'] = tabla_elasticidades['p_value'] < 0.10
#         tabla_elasticidades['periodo'] = periodo
        
#         # Métricas globales
#         tabla_elasticidades['r2_overall'] = resultado_panel.rsquared
#         tabla_elasticidades['r2_within'] = resultado_panel.rsquared_within
#         tabla_elasticidades['f_stat'] = resultado_panel.f_statistic.stat
#         tabla_elasticidades['f_pval'] = resultado_panel.f_statistic.pval

#         resultados_globales_rolling = pd.concat(
#             [resultados_globales_rolling, tabla_elasticidades], 
#             ignore_index=True
#         )

#     except Exception as e:
#         print('\n' + '!' * 70)
#         print(f'⚠️ ADVERTENCIA: Falló la estimación para el periodo {periodo}')
#         print(f'Motivo / Error técnico: {e}')
#         print('!' * 70 + '\n')
#         continue