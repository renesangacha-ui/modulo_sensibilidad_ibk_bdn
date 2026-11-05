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

datos=pd.read_excel(dir_b+r'/raw/V05/df_fin_con_tramo.xlsx')
data_panel_sensibilidad_macroperfil=pd.read_csv(dir_b+r'/processed/V04/data_panel_sensibilidad_macroperfil.csv', sep='|')
elasticidades_macroperfil=pd.read_csv(dir_b+r'/processed/V04/elasticidad_macroperfil.csv', sep='|')

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

#---- 4.1.3) Bloque Cluster ----

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

#____________________ 6) Data Panel de Sensibilidad para Optimizacion ----

#~~~~ 6.1) Construcción ----

def promedio_ponderado(valores, pesos):
  mask = valores.notna() & pesos.notna() & (pesos > 0)
  if mask.any():
    return np.average(valores[mask], weights=pesos[mask])
  return np.nan

data_panel_sensibilidad_optimizacion = (
    datos_oferta_relevante
    .groupby(["cluster_modelar","cluster_comercial","apertura", "periodo"], observed=True)
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

data_panel_sensibilidad_optimizacion.periodo.value_counts()
data_panel_sensibilidad_optimizacion.cluster_modelar.value_counts()
data_panel_sensibilidad_optimizacion.apertura.value_counts()

#~~~~ 6.2) Descriptivos ----

#---- 6.2.1) Materialidad y Mix ----

aux_materialidad_mensual=(
    data_panel_sensibilidad_optimizacion
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
    df=data_panel_sensibilidad_optimizacion,
    columna_periodo='periodo',
    columna_grupo='apertura',
    metrica_materialidad='desembolsos_ibk',
    titulo='Mix de Monto Desembolsado por Macroperfil'
)

graficos.plot_participacion_panel(
    df=data_panel_sensibilidad_optimizacion,
    columna_periodo='periodo',
    columna_grupo='apertura',
    metrica_materialidad='monto_desembolsado_ibk',
    titulo='Mix de Monto Desembolsado por Macroperfil'
)

#---- 6.2.2) Evolutivo: Tasa Ofertada vs Efectividad ----

data_panel_sensibilidad_optimizacion.groupby('apertura')[['efectividad']].describe().T
data_panel_sensibilidad_optimizacion.groupby('apertura')[['tasa_ofertada']].describe().T

for bucket in data_panel_sensibilidad_optimizacion['apertura'].unique():

    df_filtrado = data_panel_sensibilidad_optimizacion.query("apertura==@bucket")

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

for periodo in sorted(data_panel_sensibilidad_optimizacion.query('periodo>=202602')['periodo'].unique()):
  anio = int(periodo) // 100
  mes = int(periodo) % 100
  periodo_label = f"{meses[mes]}' {str(anio)[-2:]}"

  graficos.plot_nube_puntos_precio_demanda_interna(
      data_panel_sensibilidad_optimizacion.query(f'periodo=={periodo}'),
      columna_precio='tasa_ofertada',
      columna_demanda='efectividad',
      columna_tamanio='monto_desembolsado_ibk',
      columna_cluster='apertura',
      titulo=(
          'Relación entre Tasa Ofertada y'
          f' Efectividad\nPeriodo: {periodo_label}'
      ),
  )

#____________________ 7) Construción de escenarios: Tasa Ofertada vs Efectividad ----

#~~~~ 7.1) Parámetros ----

MES_CORTE=202609

#~~~~ 7.2) Preparación de información para escenarios ----

elasticidades_referencia=(
    elasticidades_macroperfil
    .query(f"periodo=={MES_CORTE} and significativo_10pct==True ")
    [['cluster','elasticidad','ic_inferior','ic_superior']]
    .rename(columns={'ic_inferior':'elasticidad_pesimista', 'ic_superior':'elasticidad_optimista', 'cluster':'cluster_modelo'})
)

rangos_referencia=(
    data_panel_sensibilidad_macroperfil
    .query(f'periodo <= {MES_CORTE}')
    .groupby('cluster_modelar')
    .agg(
        tea_max=('tasa_ofertada', 'max'),
        tea_min=('tasa_ofertada', 'min')
    )
    .reset_index()
    .assign(rango=lambda x: x['tea_max'] - x['tea_min'])
    .rename(columns={'cluster_modelar':'cluster_modelo'})
)

datos_escenarios=(
   data_panel_sensibilidad_optimizacion
   .query(f"periodo=={MES_CORTE}")
   .rename(columns={'apertura':'grupo','cluster_modelar':'cluster_modelo','tasa_ofertada':'tasa_as_is', 'monto_ofertado':'monto_ofertado_as_is', 'efectividad':'efectividad_as_is'})
   .assign(monto_desembolsado_as_is=lambda x: x['monto_ofertado_as_is']*x['efectividad_as_is'],
           ingreso_financiero_as_is=lambda x: x['monto_desembolsado_as_is']*x['tasa_as_is'])
    [['grupo','cluster_comercial','cluster_modelo','periodo','tasa_as_is','monto_ofertado_as_is','efectividad_as_is','monto_desembolsado_as_is','ingreso_financiero_as_is']]
)

datos_modelo_elasticidad=pd.merge(
   elasticidades_referencia,
   rangos_referencia,
   how='inner',
   on='cluster_modelo'
)

datos_escenarios_consolidado=pd.merge(
   datos_escenarios,
   datos_modelo_elasticidad,
   how='left',
   on='cluster_modelo',
   indicator='match_elasticidad'
)

datos_escenarios_consolidado.value_counts('match_elasticidad')

datos_escenarios_consolidado=datos_escenarios_consolidado.query('match_elasticidad!="left_only"')

datos_escenarios_consolidado.isnull().mean()

#~~~~ 7.3) Simulación de escenarios ----

#---- 7.3.1) Optimización: Elasticidad Penalizada ----

#____  7.3.1.1) Funciones ----

def nueva_efectividad(efectividad_as_is, beta_elasticidad, tasa_as_is, variacion_tasa):
    return efectividad_as_is * (
        ((tasa_as_is + variacion_tasa) / tasa_as_is) ** beta_elasticidad
    )

def simular_escenarios_tarifarios_if(
    df,
    rango_deltas_pb=(-0.025, 0.0251, 0.005),
    col_grupo="grupo",
    col_tasa_as_is="tasa_as_is",
    col_tea_max="tea_max",
    col_tea_min="tea_min",
    col_rango="rango",
    col_elasticidad="elasticidad",
    col_el_pesimista="elasticidad_pesimista",
    col_el_optimista="elasticidad_optimista",
    col_efectividad_as_is="efectividad_as_is",
    col_monto_ofertado="monto_ofertado_as_is",
    col_monto_desembolsado="monto_desembolsado_as_is",
    col_ingreso_as_is="ingreso_financiero_as_is",
):
    """Ejecuta la simulación de escenarios de tasa con penalización por límites,

    elasticidad asimétrica y cálculo de métricas TO-BE y sus deltas.
    """
    # Hacemos una copia para evitar modificar el DataFrame original en el scope exterior
    datos = df.copy()
    deltas_pb = np.arange(*rango_deltas_pb)
    res = []

    for d in deltas_pb:
        # Cálculo de distancia fuera de límites
        distancia_fuera = np.maximum(
            0, (datos[col_tasa_as_is] + d) - datos[col_tea_max]
        ) + np.maximum(0, datos[col_tea_min] - (datos[col_tasa_as_is] + d))

        peso = np.minimum(1, distancia_fuera / datos[col_rango])

        # Selección de elasticidad según el escenario (positivo o negativo)
        beta_escenario = np.where(
            d >= 0, datos[col_el_pesimista], datos[col_el_optimista]
        )

        beta_efectiva = (
            datos[col_elasticidad]
            + (beta_escenario - datos[col_elasticidad]) * peso
        )

        delta_test = d

        # Cálculos TO-BE
        efectividad_to_be = nueva_efectividad(
            efectividad_as_is=datos[col_efectividad_as_is],
            beta_elasticidad=beta_efectiva,
            tasa_as_is=datos[col_tasa_as_is],
            variacion_tasa=delta_test,
        )

        monto_ofertado_to_be = datos[
            col_monto_ofertado
        ]  # Oferta base constante
        monto_desembolsado_to_be = monto_ofertado_to_be * efectividad_to_be
        tasa_to_be = datos[col_tasa_as_is] + delta_test
        ingreso_financiero_to_be = (
            monto_ofertado_to_be * efectividad_to_be * tasa_to_be
        )

        # Cálculo de Deltas (To-Be menos As-Is)
        delta_monto_ofertado = monto_ofertado_to_be - datos[col_monto_ofertado]
        delta_monto_desembolsado = (
            monto_desembolsado_to_be - datos[col_monto_desembolsado]
        )
        delta_efectividad = efectividad_to_be - datos[col_efectividad_as_is]
        delta_tasa = tasa_to_be - datos[col_tasa_as_is]
        delta_ingreso_financiero = (
            ingreso_financiero_to_be - datos[col_ingreso_as_is]
        )

        # Construcción del dataframe temporal por escenario
        tmp = pd.DataFrame(
            {
                "grupo": datos[col_grupo],
                "ajuste_pb": int(round(d * 10000)),
                "delta_test": delta_test,
                # Métricas AS-IS
                "monto_ofertado_as_is": datos[col_monto_ofertado],
                "monto_desembolsado_as_is": datos[col_monto_desembolsado],
                "efectividad_as_is": datos[col_efectividad_as_is],
                "tasa_as_is": datos[col_tasa_as_is],
                "ingreso_financiero_as_is": datos[col_ingreso_as_is],
                # Métricas TO-BE
                "monto_ofertado_to_be": monto_ofertado_to_be,
                "monto_desembolsado_to_be": monto_desembolsado_to_be,
                "efectividad_to_be": efectividad_to_be,
                "tasa_to_be": tasa_to_be,
                "ingreso_financiero_to_be": ingreso_financiero_to_be,
                # Deltas (To-Be - As-Is)
                "delta_monto_ofertado": delta_monto_ofertado,
                "delta_monto_desembolsado": delta_monto_desembolsado,
                "delta_efectividad": delta_efectividad,
                "delta_tasa": delta_tasa,
                "delta_ingreso_financiero": delta_ingreso_financiero,
                # Variables de control e intermedias
                "beta_efectiva": beta_efectiva,
                "distancia_fuera": distancia_fuera,
                "peso": peso,
                "elasticidad_pesimista": datos[col_el_pesimista],
                "elasticidad_optimista": datos[col_el_optimista],
            }
        )

        res.append(tmp)

    df_escenarios = pd.concat(res, ignore_index=True)
    return df_escenarios

#---- 7.3.1.2) Aplicación ----

df_escenarios=simular_escenarios_tarifarios_if(
    datos_escenarios_consolidado,
    rango_deltas_pb=(-0.030, 0.0301, 0.001),
    col_grupo="grupo",
    col_tasa_as_is="tasa_as_is",
    col_tea_max="tea_max",
    col_tea_min="tea_min",
    col_rango="rango",
    col_elasticidad="elasticidad",
    col_el_pesimista="elasticidad_pesimista",
    col_el_optimista="elasticidad_optimista",
    col_efectividad_as_is="efectividad_as_is",
    col_monto_ofertado="monto_ofertado_as_is",
    col_monto_desembolsado="monto_desembolsado_as_is",
    col_ingreso_as_is="ingreso_financiero_as_is",
)

df_escenarios=df_escenarios.query('0<efectividad_to_be<1')

#---- 7.3.2) Selección: Escenarios óptimos ----

# Seleccionamos los escenarios que máximizen el delta de ingreso financiero (if to be - if as is)
escenarios_optimos = (
    df_escenarios
    .sort_values(["grupo", "delta_ingreso_financiero"])
    .groupby("grupo", as_index=False)
    .last()
)

escenarios_optimos=escenarios_optimos.query('delta_ingreso_financiero>0')
escenarios_optimos.ajuste_pb.describe()
escenarios_optimos.ajuste_pb.value_counts()

escenarios_optimos

#---- 7.3.3) Visualizaciones ----

for g in df_escenarios["grupo"].unique():

    df_g = df_escenarios[df_escenarios["grupo"] == g]

    # Delta de tasa vs delta de ingreso financiero
    graficos.plot_curva_optimizacion_bucket(
        df_grupo=df_g,
        columna_x="ajuste_pb",
        columna_y="delta_ingreso_financiero",
        label_x="Ajuste de Tasa (Puntos Base / bps)",
        label_y="Δ Ingreso Financiero",
        titulo=f"Optimización de Ingreso Financiero\n{g}",
        flg_destacar_optimo=True, 
         parametros_as_is=[
            "tasa",
            "efectividad",
            "monto_ofertado",
            "ingreso_financiero",
        ],
    )

    # Delta de tasa vs delta de ingreso financiero
    graficos.plot_curva_optimizacion_bucket(
        df_grupo=df_g,
        columna_x="ajuste_pb",
        columna_y="efectividad_to_be",
        label_x="Ajuste de Tasa (Puntos Base / bps)",
        label_y="Efectividad(%)",
        titulo=f"Escenarios de Efectividad\n{g}",
        flg_destacar_optimo=False,
        parametros_as_is=[
            "tasa",
            "efectividad",
        ],
    )

    # Delta de tasa vs delta de monto desembolsado
    graficos.plot_curva_optimizacion_bucket(
        df_grupo=df_g,
        columna_x="ajuste_pb",
        columna_y="monto_desembolsado_to_be",
        label_x="Ajuste de Tasa (Puntos Base / bps)",
        label_y="Monto Desembolsado",
        titulo=f"Escenarios del Monto Desembolsado\n{g}",
        flg_destacar_optimo=False,
        parametros_as_is=[
            "tasa",
            "monto_desembolsado",
        ],
    )

    # Delta de tasa vs delta de monto desembolsado
    graficos.plot_curva_optimizacion_bucket(
        df_grupo=df_g,
        columna_x="ajuste_pb",
        columna_y="ingreso_financiero_to_be",
        label_x="Ajuste de Tasa (Puntos Base / bps)",
        label_y="Ingreso Financiero",
        titulo=f"Escenarios de Ingreso Financiero\n{g}",
        flg_destacar_optimo=False,
        parametros_as_is=[
            "tasa",
            "monto_desembolsado",
            "ingreso_financiero"
        ],
    )