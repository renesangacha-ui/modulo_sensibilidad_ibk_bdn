################################## Modelo de Elasticidad ############################

#####################################################################################

print('-' * 100)
print('\t\t\t\t\tModelo de Elasticidad')
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

from linearmodels.panel import PanelOLS

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

datos=pd.read_csv(dir_b+r'/raw/V04/data_actualizada_mar25_set26.csv',sep=',')

#____________________ 4) Procesamiento ----

#~~~~ 4.1) Categorizaciones ----

#---- 4.1.1) Cluster Comercial ----

mapeo_clusters = {
     'cluster_1_sow_no_nulo': 'Cliente Ancla',
     'cluster_2_sow_no_nulo': 'Socio en Crecimiento',
     'cluster_3_sow_no_nulo': 'Cuenta Estratégica',
     'cluster_4_sow_no_nulo': 'Maduro Multibanco',
     'cluster_5_sow_no_nulo': 'Gran Cuenta por Conquistar',
     'cluster_6_sow_no_nulo': 'Cliente por Reactivar',     
     'cluster_1_nulo': 'Emprendedor Izipay',
     'cluster_2_nulo': 'Nuevo en Expansión',
     'cluster_3_nulo': 'Próximo Recurrente',
}

datos['cluster_comercial'] = datos['cluster'].map(mapeo_clusters).fillna(datos['cluster'])
datos[['cluster_comercial','cluster']].value_counts().reset_index()

#---- 4.1.2) Macroperfil ----

CLUSTER_MODELAMIENTO={
     'cluster_1_sow_no_nulo': 'Fidelizados',
     'cluster_2_sow_no_nulo': 'Fidelizados',
     'cluster_3_sow_no_nulo': 'Grandes por fidelizar',
     'cluster_4_sow_no_nulo': 'Medianos por fidelizar',
     'cluster_5_sow_no_nulo': 'Grandes por fidelizar',
     'cluster_6_sow_no_nulo': 'Medianos por fidelizar',     
     'cluster_1_nulo': 'Sin historial crediticio',
     'cluster_2_nulo': 'Sin historial crediticio',
     'cluster_3_nulo': 'Sin historial crediticio'
}

datos['cluster_modelar']=datos['cluster'].map(CLUSTER_MODELAMIENTO)

datos.value_counts(['cluster_modelar','cluster']).reset_index()

#---- 4.1.4) Bloque Cluster ----

datos['bloque_cluster']=datos['cluster_comercial'].isin(['Emprendedor Izipay','Nuevo en Expansión','Próximo Recurrente']).map({True:'Sin Deuda', False:'Con Deuda'})
datos.value_counts(['bloque_cluster','cluster_comercial']).reset_index()

#~~~~ 4.2) Tratamientos ----

datos['tasa_desembolsada']=np.where(
    datos['flg_exito_4'].isin(['exito_desembolso_ibk_ct','fracaso_desembolso_ibk_no_ct']),
    datos['tasa_2022']/100,
    np.nan
)

datos['monto_desembolsado']=np.where(
    datos['flg_exito_4'].isin(['exito_desembolso_ibk_ct','fracaso_desembolso_ibk_no_ct']),
    datos['monto_desembolsado'],
    np.nan
)

datos['monto_ofertado_ajustado'] = np.where(
    ~datos['flg_exito_4'].isin(['exito_desembolso_ibk_ct','fracaso_desembolso_ibk_no_ct']),
    datos['monto_min'],
    np.where(
        datos['monto_desembolsado'] > datos['monto_min'],
        datos['monto_desembolsado'],
        datos['monto_min']
    )
)

datos.query('tasa_desembolsada.notna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.notna()').flg_exito_4.value_counts(dropna=False)

datos.query('tasa_desembolsada.isna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.isna()').flg_exito_4.value_counts(dropna=False)

#____________________ 5) Funnel Poblacional ----

PERIODO_INICIO=202504
PERIODO_FIN=202609

funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

datos_oferta_relevante = datos.query(f"flg_exito_4 in {funnel_oferta_relevante} and periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}").copy()

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
datos.query(f"flg_exito_4 in ['exito_desembolso_ibk_ct'] and periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}").shape

#____________________ 6) Data Panel de Sensibilidad ----

#~~~~ 6.1) Construcción ----

def promedio_ponderado(valores, pesos):
  mask = valores.notna() & pesos.notna() & (pesos > 0)
  if mask.any():
    return np.average(valores[mask], weights=pesos[mask])
  return np.nan

data_panel_sensibilidad = (
    datos_oferta_relevante
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

data_panel_sensibilidad.periodo.value_counts()
data_panel_sensibilidad.cluster_modelar.value_counts()

#~~~~ 6.2) Descriptivos ----

#---- 6.2.1) Materialidad y Mix ----

aux_materialidad_mensual=(
    data_panel_sensibilidad
    .groupby('periodo')
    .agg(desembolsos_ibk=('desembolsos_ibk','sum'),
         monto_desembolsado_ibk=('monto_desembolsado_ibk','sum'))
    .reset_index()
)

graficos.plot_materialidad_por_mes(
      df=aux_materialidad_mensual,
      col_periodo='periodo',
      col_valor='desembolsos_ibk',
      titulo='Evolución de Desembolsos',
      ylabel='Desembolsos',
      color_barras='#1D4290',
      es_monto=False
  )

graficos.plot_materialidad_por_mes(
      df=aux_materialidad_mensual,
      col_periodo='periodo',
      col_valor='monto_desembolsado_ibk',
      titulo='Evolución de Monto Desembolsado',
      ylabel='Monto Desmbolsado (MM)',
      color_barras='#1D4290',
      es_monto=True
  )

graficos.plot_participacion_panel(
    df=data_panel_sensibilidad,
    columna_periodo='periodo',
    columna_grupo='cluster_modelar',
    metrica_materialidad='desembolsos_ibk',
    titulo='Mix de Monto Desembolsado por Macroperfil'
)

graficos.plot_participacion_panel(
    df=data_panel_sensibilidad,
    columna_periodo='periodo',
    columna_grupo='cluster_modelar',
    metrica_materialidad='monto_desembolsado_ibk',
    titulo='Mix de Monto Desembolsado por Macroperfil'
)

#---- 6.2.2) Evolutivo: Tasa Ofertada vs Efectividad ----

data_panel_sensibilidad.groupby('cluster_modelar')[['efectividad']].describe().T
data_panel_sensibilidad.groupby('cluster_modelar')[['tasa_ofertada']].describe().T

for bucket in data_panel_sensibilidad['cluster_modelar'].unique():

    df_filtrado = data_panel_sensibilidad.query("cluster_modelar==@bucket")

    graficos.plot_evolutivo_demanda_precio_barras(
        df=df_filtrado,
        columna_demanda='efectividad',
        columna_precio='tasa_ofertada',
        titulo=f"Evolución de Tasa Ofertada y Efectividad\n{bucket}",
        flg_etiqueta_demanda=True,
        flg_etiqueta_precio=True
    )

#---- 6.2.3) Bubble Plots: Tasa vs Efectividad ----

meses = {
    1: 'Ene',
    2: 'Feb',
    3: 'Mar',
    4: 'Abr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Ago',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dic',
}

for periodo in sorted(data_panel_sensibilidad.query('periodo>=202602')['periodo'].unique()):
  anio = int(periodo) // 100
  mes = int(periodo) % 100
  periodo_label = f"{meses[mes]}' {str(anio)[-2:]}"

  graficos.plot_nube_puntos_precio_demanda_interna(
      data_panel_sensibilidad.query(f'periodo=={periodo}'),
      columna_precio='tasa_ofertada',
      columna_demanda='efectividad',
      columna_tamanio='monto_desembolsado_ibk',
      columna_cluster='cluster_modelar',
      titulo=(
          'Relación entre Tasa Ofertada y'
          f' Efectividad\nPeriodo: {periodo_label}'
      ),
  )

#____________________ 7) Modelamiento Rolling Window: Tasa Ofertada vs Efectividad ----

# Se estiman elasticidades por bucket mediante un modelo de panel con efectos fijos de entidad y tiempo,
# utilizando ventanas acumulativas hasta cada periodo de corte (dic-25 a abr-26)

resultados_globales_rolling = pd.DataFrame()

for periodo in [202512, 202601, 202602, 202603,202604, 202605,202606,202607,202608,202609]:

    #~~~~ 7.1) Información para modelamiento ----

    df = data_panel_sensibilidad[['cluster_modelar','periodo','efectividad','tasa_ofertada']].copy()

    df = df.query(f'periodo<={periodo}')

    df = df.rename(columns={
        'cluster_modelar': 'bucket'
    })

    df['mes'] = pd.to_datetime(
        df['periodo'].astype('Int64').astype(str),
        format='%Y%m',
        errors='coerce'
    )

    for col in ['efectividad', 'tasa_ofertada']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(
        subset=['bucket', 'mes', 'efectividad', 'tasa_ofertada']
    ).copy()

    df = df.loc[
        (df['efectividad'] > 0) &
        (df['tasa_ofertada'] > 0)
    ].copy()

    df['log_efectividad'] = np.log(df['efectividad'])
    df['log_tasa'] = np.log(df['tasa_ofertada'])

    df['bucket'] = df['bucket'].astype(str)

    buckets = sorted(df['bucket'].unique())
    nombres_interaccion = {}

    # Creación de interacciones
    for i, bucket in enumerate(buckets):
        nombre_columna = f'tasa_inter_{i}'
        df[nombre_columna] = (
            df['log_tasa'] *
            (df['bucket'] == bucket).astype(int)
        )
        nombres_interaccion[bucket] = nombre_columna

    print(f'\n--- Procesando ventana hasta periodo: {periodo} ---')
    print(f'Observaciones utilizadas: {len(df):,}')
    print(f'Clusters: {df["bucket"].nunique()}')
    print(f'Meses: {df["mes"].nunique()}')

    #~~~~ 7.2) Estimación: Modelo de Panel con Efectos Fijos e Interacciones ----
  
    try:
        df_panel = df.set_index(["bucket", "mes"]).sort_index()
        columnas_x = list(nombres_interaccion.values())

        modelo_panel = PanelOLS(
            dependent=df_panel["log_efectividad"],
            exog=df_panel[columnas_x],
            entity_effects=True,
            time_effects=True,
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

        intervalos = resultado_panel.conf_int(level=0.95)
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

        r2_overall = resultado_panel.rsquared
        r2_within = resultado_panel.rsquared_within
        r2_between = resultado_panel.rsquared_between
        n_obs = resultado_panel.nobs
        f_stat = resultado_panel.f_statistic.stat
        f_pval = resultado_panel.f_statistic.pval

        tabla_elasticidades['r2_overall'] = r2_overall
        tabla_elasticidades['r2_within'] = r2_within
        tabla_elasticidades['r2_between'] = r2_between
        tabla_elasticidades['f_stat'] = f_stat
        tabla_elasticidades['f_pval'] = f_pval
        
        resultados_globales_rolling = pd.concat(
            [resultados_globales_rolling, tabla_elasticidades], 
            ignore_index=True
        )

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

#____________________ 8) Validaciones ----

# resultados_globales_rolling.query("periodo==202604").to_csv('resultados_elasticidad.csv',sep='|', index=False)

resultados_globales_rolling['flg_significativo']=(resultados_globales_rolling['significativo_10pct'])*1

#~~~~ 8.1) Significancia estadística de las elasticidades ----

graficos.plot_heatmap_cumplimiento(
    df_panel=resultados_globales_rolling,
    col_cluster='cluster',
    col_periodo='periodo',
    col_flag='flg_significativo',
    titulo='Elasticidades Significativas', 
    subtitulo=f'Umbral mínimo de significancia: < {0.1:.0%}')

#~~~~ 8.2) Evolución de las elasticidades ----

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

#~~~~ 8.3) Monitoreo de elasticidades ----

# Se monitorea la estabilidad de las elasticidades por macroperfil a través de ventanas
# de estimación sucesivas, identificando cambios relevantes en su nivel y volatilidad.
# Se utilizan límites de control I-MR, basados en la variabilidad histórica, para detectar
# posibles desviaciones que puedan comprometer la estabilidad de los parámetros de pricing.

# Basado en la metodología de Control Estadístico de Procesos (SPC) de Shewhart, mediante gráficos I-MR. 

#---- 8.3.1) Funciones ----

def calcular_limites_imr(data, variable='elasticidad'):

    data = data.sort_values('periodo').copy()

    if len(data) < 2:
        return None

    # Rango móvil
    data['MR'] = data[variable].diff().abs()

    # Constantes I-MR
    D4 = 3.267
    A2_MR = 2.66

    # Parámetros de control
    media = data[variable].mean()
    mr_promedio = data['MR'].mean()

    # Límites de individuales
    data['LC_I'] = media
    data['LSC_I'] = media + A2_MR * mr_promedio
    data['LIC_I'] = media - A2_MR * mr_promedio

    # Límites de rangos móviles
    data['LC_MR'] = mr_promedio
    data['LSC_MR'] = D4 * mr_promedio
    data['LIC_MR'] = 0.0

    # Alertas
    data['alerta_I'] = (
        (data[variable] > data['LSC_I']) |
        (data[variable] < data['LIC_I'])
    )

    data['alerta_MR'] = data['MR'] > data['LSC_MR']

    return data

#---- 8.3.2) Monitoreo ----

df_control = resultados_globales_rolling.copy()

df_control['periodo'] = pd.to_numeric(df_control['periodo'], errors='coerce')

df_control['elasticidad'] = pd.to_numeric(df_control['elasticidad'], errors='coerce')

df_control = (
    df_control
    .dropna(subset=['cluster', 'periodo', 'elasticidad'])
    .sort_values(['cluster', 'periodo'])
)

resultados_control = []

for cluster, data in df_control.groupby('cluster'):

    data_control = calcular_limites_imr(data)

    if data_control is None:
        print(f'{cluster}: observaciones insuficientes.')
        continue

    resultados_control.append(data_control)

    graficos.graficar_tendencia(data_control, cluster)

    graficos.graficar_volatilidad(data_control, cluster)

df_resultados_control = pd.concat(
    resultados_control,
    ignore_index=True
)

#____________________ 9) Exportación: Resultados ----

data_panel_sensibilidad.to_csv(dir_b+r'/processed/V04/data_panel_sensibilidad_macroperfil.csv',index=False, sep='|')
resultados_globales_rolling.to_csv(dir_b+r'/processed/V04/elasticidad_macroperfil.csv',index=False, sep='|')
df_resultados_control.to_csv(dir_b+r'/processed/V04/monitoreo_elasticidad_macroperfil.csv',index=False, sep='|')
