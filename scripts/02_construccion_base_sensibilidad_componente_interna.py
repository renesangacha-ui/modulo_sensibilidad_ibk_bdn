################### Construcción Base Sensibilidad: Componente Interna ##################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tConstrucción Base Sensibilidad: Componente Interna')
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

datos=pd.read_csv(dir_b+r'/raw/V01/data_sensibilidad.csv',sep=',')

datos.head().T
datos.info()
datos.isnull().mean().sort_values(ascending=False)
datos.describe()
datos.periodo.min(), datos.periodo.max()

#____________________ 4) Funnel ----

datos.flg_exito_4.value_counts()

# Nos quedamos con los exitos y los fracasos de desembolso no IBK con monto similar con gestión,
#  que son los que nos interesan para la construcción de la base de sensibilidad.

funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

datos = datos.query("flg_exito_4 in @funnel_oferta_relevante").copy()


# Conservamos las entidades que concentran la mayor parte de las inferencias de tasa:
# 1: BCP
# 6: BBVA
# 4: SCOTIA BANK
# 123: MIBANCO
# 2: IBK

cod_entidades_inferencia=[1.0,2.0,6.0,4.0,123.0]

datos=datos.query(f"cod_instit_financiera in @cod_entidades_inferencia")

#____________________ 5) Validaciones ----

# Columnas clave
# id: llave_interna <- key_value + codmes
# tasa ofertada: tasa_min
# monto ofertado: monto_min
# plazo ofertado: plazo_min
# periodo: periodo o codmes?
# banco: cod_instit_financiera / cod_entidad?
# cluster: cluster no caracterizado
# monto desembolsado IBK: monto_desembolsado
# tasa desembolsada IBK: tasa_2022
# monto desembolsado VOI: delta_deuda_final (SoS de la oferta relevante)
# flag de oferta relevante: flg_exito_4
# monto desembolsado Riesgos: saldo_desembolso
# tasa inferida: tasa_ultima

#~~~~ 5.1) Duplicidad: llave_interna ----

datos.value_counts(['llave_interna']).reset_index().value_counts('count', normalize=True)

registros_duplicados=datos.value_counts(['llave_interna']).reset_index().query('count>1').llave_interna.to_list()

validacion_duplicidad=(
    datos
    .query("llave_interna in @registros_duplicados")
    .groupby(['llave_interna'], observed=True)
    .agg(tasa_min=('tasa_min', 'nunique'),
         monto_min=('monto_min', 'nunique'),
         cluster=('cluster', 'nunique'),
         cod_instit_financiera=('cod_instit_financiera', 'nunique'),
         flg_exito_4=('flg_exito_4', 'nunique'))

)

validacion_duplicidad

validacion_duplicidad.query('tasa_min>1 or monto_min>1 or cluster>1')

validacion_duplicidad.query('cod_instit_financiera>1 or flg_exito_4>1')

validacion_duplicidad.query('flg_exito_4==1 and cod_instit_financiera==1')

# El 5.5% de los registros presenta duplicidad por desembolsos en más de un banco.
# A diferencia del Share of Sales, que considera todos los registros del cliente, el análisis
# de sensibilidad evalúa la aceptación de la oferta de IBK, por lo que requiere un único
# registro por cliente-periodo. Se considera éxito si el cliente desembolsa en IBK y fracaso
# en caso contrario.

#~~~~ 5.2) Completitud: Tasas y Monto ----

#--- 5.2.1) Leads: Oferta Relevante con Gestión ----

datos.tasa_min.isna().mean() 
datos.tasa_min.describe()

datos.monto_min.isna().mean()
datos.monto_min.describe()

#--- 5.2.2) Desembolsos ----

datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").tasa_2022.isna().mean() 
datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").tasa_2022.describe()

datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").tasa_min.isna().mean() 
datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").tasa_min.describe()


datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").monto_desembolsado.isna().mean() 
datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").monto_desembolsado.describe()

datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").monto_min.isna().mean() 
datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").monto_min.describe()

#~~~~ 5.3) Desembolso vs Oferta: Monto ----

datos.query("monto_desembolsado.notna() and monto_desembolsado > monto_min")

(
    datos.query("monto_desembolsado.notna() and monto_desembolsado > monto_min")
    .assign(diferencia=lambda x: x["monto_desembolsado"] - x["monto_min"])
    ["diferencia"]
    .describe(percentiles=[i / 100 for i in range(1, 100)])
)

# Se identifican 2,232 registros con un monto desembolsado superior al ofertado por IBK.
# El 90% presenta una diferencia menor a S/ 25, mientras que el 10% restante supera los S/ 10 mil.
# Para evitar efectividades superiores al 100% en el modelamiento de sensibilidad, se iguala
# el monto ofertado al desembolsado en estos casos.

#____________________ 6) Procesamiento ----

#~~~~ 6.1) Recodificaciones ----

mapeo_clusters = {
     'cluster_1_sow_no_nulo': 'Cliente Ancla',
     'cluster_2_sow_no_nulo': 'Socio en Crecimiento',
     'cluster_3_sow_no_nulo': 'Cuenta Estratégica',
     'cluster_4_sow_no_nulo': 'Gran Cuenta por Conquistar',
     'cluster_5_sow_no_nulo': 'Maduro Multibanco',
     'cluster_6_sow_no_nulo': 'Cliente por Reactivar',     
     'cluster_1_nulo': 'Emprendedor Izipay',
     'cluster_2_nulo': 'Nuevo en Expansión',
     'cluster_3_nulo': 'Próximo Recurrente',
}

datos['cluster_comercial'] = datos['cluster'].map(mapeo_clusters).fillna(datos['cluster'])
datos[['cluster_comercial','cluster']].value_counts().reset_index()

#~~~~ 6.2) Reescalamiento ----

datos['tasa_desembolsada'] = datos['tasa_2022']/100

#~~~~ 6.3) Tratamiento: Duplicidad ----

datos_oferta_relevante = (
    datos
    .sort_values(
        by=['llave_interna', 'flg_exito_4'],
        ascending=[True, True]
    )
    .drop_duplicates(subset=['llave_interna'], keep='first')
    .assign(
        indicador_exito=lambda df: (df['flg_exito_4'] == 'exito_desembolso_ibk_ct').astype(int),
    )
)

#~~~~ 6.4) Tratamiento: Monto ofertado ----

datos_oferta_relevante['monto_ofertado_ajustado'] = np.where(
    datos_oferta_relevante['monto_desembolsado'].isna(),
    datos_oferta_relevante['monto_min'],
    np.where(
        datos_oferta_relevante['monto_desembolsado'] > datos_oferta_relevante['monto_min'],
        datos_oferta_relevante['monto_desembolsado'],
        datos_oferta_relevante['monto_min']
    )
)

#~~~~ 6.5) Base final: Componente Sensibilidad Interna ----

columnas_sensibilidad_interna = [
    'periodo',
    'llave_interna',
    'cluster',
    'cluster_comercial',
    'indicador_exito',
    'flg_exito_4',
    'monto_min',
    'monto_ofertado_ajustado',
    'tasa_min',
    'plazo_min',
    'monto_desembolsado',
    'tasa_desembolsada',
]   

datos_oferta_relevante=datos_oferta_relevante[columnas_sensibilidad_interna].copy()

#____________________ 7) EDA ----

#~~~~ 7.1) Construcción: Data Panel ----

def promedio_ponderado(valores, pesos):
  mask = valores.notna() & pesos.notna() & (pesos > 0)
  if mask.any():
    return np.average(valores[mask], weights=pesos[mask])
  return np.nan

data_panel = (
    datos_oferta_relevante
    .groupby(["cluster_comercial", "periodo"], observed=True)
    .agg(
        leads=("llave_interna", "size"),
        monto_desembolsado_ibk=("monto_desembolsado", "sum"),
        monto_ofertado=("monto_ofertado_ajustado", "sum"),
        desembolsos=("indicador_exito", "sum"),
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
            df["leads"] > 0, df["desembolsos"] / df["leads"], 0
        ),
    )
    .reset_index()
)

#~~~~ 7.2) Efectividad Promedio----

aux_tasa_efectividad_global = (
    data_panel.groupby(['cluster_comercial'], observed=True)
    .agg(
        leads=('leads', 'sum'),
        monto_desembolsado_ibk=('monto_desembolsado_ibk', 'sum'),
        monto_ofertado=('monto_ofertado', 'sum'),
        desembolsos=('desembolsos', 'sum'),
        tasa_desembolsada=(
            'tasa_desembolsada',
            lambda x: promedio_ponderado(
                x,
                data_panel.loc[
                    x.index, 'monto_desembolsado_ibk'
                ],
            ),
        ),
        tasa_ofertada=(
            'tasa_ofertada',
            lambda x: promedio_ponderado(
                x, data_panel.loc[x.index, 'monto_ofertado']
            ),
        ),
    )
    .assign(
        efectividad=lambda df: np.where(
            df['monto_ofertado'] > 0,
            df['monto_desembolsado_ibk'] / df['monto_ofertado'],
            0,
        ),
        conversion=lambda df: np.where(
            df['leads'] > 0, df['desembolsos'] / df['leads'], 0
        ),
    )
    .reset_index()
)

aux_tasa_efectividad_global['bloque_cluster']=aux_tasa_efectividad_global['cluster_comercial'].isin(['Emprendedor Izipay','Nuevo en Expansión','Próximo Recurrente']).map({True:'Sin Deuda', False:'Con Deuda'})

#---- 7.2.1) VS: Tasa Desembolsada ----

graficos.plot_efectividad_y_tasa_por_bloques(
    df=aux_tasa_efectividad_global,
    columna_cluster='cluster_comercial',
    columna_bloque='bloque_cluster',
    metrica_efectividad='efectividad',
    columna_tasa='tasa_desembolsada',
    titulo="Efectividad y Tasa por Clúster Comercial\nHorizonte: Mar' 25 - May' 26",
)

#---- 7.2.2) VS: Materialidad ----

graficos.plot_efectividad_por_bloques_y_materialidad(
    df=aux_tasa_efectividad_global,
    columna_cluster='cluster_comercial',
    columna_bloque='bloque_cluster',
    metrica_efectividad='efectividad',
    columna_materialidad='desembolsos',
    titulo="Efectividad por Clúster Comercial\nHorizonte: Mar' 25 - May' 26\nBarras de fondo: materialidad del clúster (Desembolsos)",
)

#~~~~ 7.3) Materilidad ----

#---- 7.3.1) Global ----

aux_materialidad_global = (
    data_panel
    .groupby('cluster_comercial', observed=True)
    .agg(
        leads=('leads', 'sum'),
        monto_ofertado=('monto_ofertado', 'sum'),
        desembolsos=('desembolsos', 'sum'),
        monto_desembolsado_ibk=('monto_desembolsado_ibk', 'sum')
    )
    .assign(
        participacion_leads=lambda df: df['leads'] / df['leads'].sum(),
        participacion_monto_ofertado=lambda df: df['monto_ofertado'] / df['monto_ofertado'].sum(),
        participacion_desembolsos=lambda df: df['desembolsos'] / df['desembolsos'].sum(),
        participacion_monto_desembolsado=lambda df: df['monto_desembolsado_ibk'] / df['monto_desembolsado_ibk'].sum()
    )
    .reset_index()
)

graficos.plot_participacion_segmento(
    df=aux_materialidad_global,
    col_segmento='cluster_comercial',
    col_participacion='participacion_leads',
    titulo="Materialidad: Leads\nPor Segmento Comercial\nHorizonte: Mar' 25 - May' 26",
    ylabel='Participación (%)')

graficos.plot_participacion_segmento(
    df=aux_materialidad_global,
    col_segmento='cluster_comercial',
    col_participacion='participacion_monto_ofertado',
    titulo="Materialidad: Monto Ofertado\nPor Segmento Comercial\nHorizonte: Mar' 25 - May' 26",
    ylabel='Participación (%)')

graficos.plot_participacion_segmento(
    df=aux_materialidad_global,
    col_segmento='cluster_comercial',
    col_participacion='participacion_desembolsos',
    titulo="Materialidad: Desembolsos\nPor Segmento Comercial\nHorizonte: Mar' 25 - May' 26",
    ylabel='Participación (%)')

graficos.plot_participacion_segmento(
    df=aux_materialidad_global,
    col_segmento='cluster_comercial',
    col_participacion='participacion_monto_desembolsado',
    titulo="Materialidad: Monto Desembolsado\nPor Segmento Comercial\nHorizonte: Mar' 25 - May' 26",
    ylabel='Participación (%)')

#---- 7.3.2) Mensual ----

aux_materialidad_mensual = (
    data_panel.groupby(['periodo', 'cluster_comercial'], observed=True)
    .agg(
        leads=('leads', 'sum'),
        monto_ofertado=('monto_ofertado', 'sum'),
        desembolsos=('desembolsos', 'sum'),
        monto_desembolsado_ibk=('monto_desembolsado_ibk', 'sum'),
    )
    .assign(
        participacion_leads=lambda df: df.groupby(level='periodo')[
            'leads'
        ].transform(lambda x: x / x.sum()),
        participacion_monto_ofertado=lambda df: df.groupby(level='periodo')[
            'monto_ofertado'
        ].transform(lambda x: x / x.sum()),
        participacion_desembolsos=lambda df: df.groupby(level='periodo')[
            'desembolsos'
        ].transform(lambda x: x / x.sum()),
        participacion_monto_desembolsado=lambda df: df.groupby(level='periodo')[
            'monto_desembolsado_ibk'
        ].transform(lambda x: x / x.sum()),
    )
    .reset_index()
)

graficos.plot_participacion_segmento_mensual_grid(
    df_panel=aux_materialidad_mensual,  
    col_segmento='cluster_comercial',
    col_participacion='participacion_leads',
    titulo_base='Materialidad: Leads',
    ylabel='Participación (%)',
)

graficos.plot_participacion_segmento_mensual_grid(
    df_panel=aux_materialidad_mensual,  
    col_segmento='cluster_comercial',
    col_participacion='participacion_monto_ofertado',
    titulo_base='Materialidad: Monto Ofertado',
    ylabel='Participación (%)',
)

graficos.plot_participacion_segmento_mensual_grid(
    df_panel=aux_materialidad_mensual,  
    col_segmento='cluster_comercial',
    col_participacion='participacion_desembolsos',
    titulo_base='Materialidad: Desembolsos',
    ylabel='Participación (%)',
)

graficos.plot_participacion_segmento_mensual_grid(
    df_panel=aux_materialidad_mensual,  
    col_segmento='cluster_comercial',
    col_participacion='participacion_monto_desembolsado',
    titulo_base='Materialidad: Monto Desembolsado',
    ylabel='Participación (%)',
)

#---- 7.3.3) Promedio Mensual ----

total_periodos = data_panel['periodo'].nunique()

aux_materialidad_promedio = (
    aux_materialidad_mensual.set_index(['periodo', 'cluster_comercial'])
    .unstack(level='cluster_comercial')
    .fillna(0)
    .sum(axis=0)
    .unstack(level=0) / total_periodos  
).reset_index()

graficos.plot_participacion_segmento(
    df=aux_materialidad_promedio,
    col_segmento='cluster_comercial',
    col_participacion='participacion_leads',
    titulo='Materialidad Promedio Mensual: Leads\nPor Segmento Comercial',
    ylabel='Participación (%)')

graficos.plot_participacion_segmento(
    df=aux_materialidad_promedio,
    col_segmento='cluster_comercial',
    col_participacion='participacion_monto_ofertado',
    titulo='Materialidad Promedio Mensual: Monto Ofertado\nPor Segmento Comercial',
    ylabel='Participación (%)')

graficos.plot_participacion_segmento(
    df=aux_materialidad_promedio,
    col_segmento='cluster_comercial',
    col_participacion='participacion_desembolsos',
    titulo='Materialidad Promedio Mensual: Desembolsos\nPor Segmento Comercial',
    ylabel='Participación (%)')

graficos.plot_participacion_segmento(
    df=aux_materialidad_promedio,
    col_segmento='cluster_comercial',
    col_participacion='participacion_monto_desembolsado',
    titulo='Materialidad Promedio Mensual: Monto Desembolsado\nPor Segmento Comercial',
    ylabel='Participación (%)')

#---- 7.3.4) Reporte: Cumplimiento Materialidad ----

UMBRAL_MATERIALIDAD=0.05

aux_materialidad_mensual['flg_cumplimiento_mat_des_monto']=np.where(
    aux_materialidad_mensual['participacion_monto_desembolsado']>UMBRAL_MATERIALIDAD,1,0
)

aux_materialidad_mensual['flg_cumplimiento_mat_des_n']=np.where(
    aux_materialidad_mensual['participacion_desembolsos']>UMBRAL_MATERIALIDAD,1,0
)

graficos.plot_heatmap_cumplimiento(
    df_panel=aux_materialidad_mensual.query('desembolsos!=0'),
    col_cluster='cluster_comercial',
    col_periodo='periodo',
    col_flag='flg_cumplimiento_mat_des_monto',
    titulo='Cumplimiento de Materialidad: Monto Desembolsado', 
    subtitulo=f'Umbral mínimo de participación: > {UMBRAL_MATERIALIDAD:.0%}'
)

graficos.plot_heatmap_cumplimiento(
    df_panel=aux_materialidad_mensual.query('desembolsos!=0'),
    col_cluster='cluster_comercial',
    col_periodo='periodo',
    col_flag='flg_cumplimiento_mat_des_n',
    titulo='Cumplimiento de Materialidad: Desembolsos', 
    subtitulo=f'Umbral mínimo de participación: > {UMBRAL_MATERIALIDAD:.0%}')

#~~~~ 7.4) Evolutivo: Precio y Demanda ----

for bucket in data_panel['cluster_comercial'].unique():

    df_filtrado = data_panel.query("cluster_comercial==@bucket")

    graficos.plot_evolutivo_precio_demanda(
        df=df_filtrado,
        columna_demanda='efectividad',
        columna_precio='tasa_desembolsada',
        columna_materialidad='desembolsos',
        titulo=f"Evolución de Tasa Desembolsada y Efectividad\n{bucket}",
        flg_materialidad=True
    )

#~~~~ 7.5) Comparativos: Precio y Demanda ----

graficos.plot_evolutivo_multiserie_panel(
    df=data_panel,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica='efectividad',
    label_metrica='Efectividad (%)',
    titulo='Evolución de Efectividad por Cluster Comercial'
)

graficos.plot_evolutivo_multiserie_panel(
    df=data_panel,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica='tasa_desembolsada',
    label_metrica='Tasa Desembolsada (%)',
    titulo='Evolución de Tasa Desembolsada por Cluster Comercial'
)

graficos.plot_participacion_panel(
    df=data_panel,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica_materialidad='desembolsos',
    titulo='Mix de Desembolsos por Cluster Comercial'
)

graficos.plot_participacion_panel(
    df=data_panel,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica_materialidad='monto_desembolsado_ibk',
    titulo='Mix de Monto Desembolsado por Cluster Comercial'
)

#~~~~ 7.6) Nube Puntos: Precio - Demanda - Materialidad ----

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

for periodo in sorted(data_panel['periodo'].unique()):
  anio = int(periodo) // 100
  mes = int(periodo) % 100
  periodo_label = f"{meses[mes]}' {str(anio)[-2:]}"

  graficos.plot_nube_puntos_precio_demanda_interna(
      data_panel.query(f'periodo=={periodo}'),
      columna_precio='tasa_desembolsada',
      columna_demanda='efectividad',
      columna_tamanio='monto_desembolsado_ibk',
      columna_cluster='cluster_comercial',
      titulo=(
          'Relación entre Tasa Desembolsada y'
          f' Efectividad\nPeriodo: {periodo_label}'
      ),
  )

#____________________ 8) Almacenamiento ----

datos_oferta_relevante.to_csv(dir_b+r'/processed/V01/base_sensibilidad_componente_interna.csv',sep='|', index=False)
