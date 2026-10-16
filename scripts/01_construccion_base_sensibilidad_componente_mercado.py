################### Construcción Base Sensibilidad: Componente Mercado ##################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tConstrucción Base Sensibilidad: Componente Mercado')
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

#____________________ 4) Funnel: Oferta Relevante ----

datos.flg_exito_4.value_counts()

# Nos quedamos con los exitos y los fracasos de desembolso no IBK con monto similar con gestión,
# que son los que nos interesan para la construcción de la base de sensibilidad.

funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

datos = datos.query("flg_exito_4 in @funnel_oferta_relevante").copy()

#____________________ 5) Procesamiento Inicial ----

#~~~~ 5.1) Recodificaciones ----

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

#~~~~ 5.2) Imputaciones ----

datos.pasivosprom_u12m.isna().sum()
datos.deuda_directa_vig_total_sf_u12m.isna().sum()

# No se necesitan imputaciones a las variables clave

#~~~~ 5.3) Categorizaciones ----

#---- 5.3.1) Proximo Recurrente y Nuevo en Expansión ----

bins_monto_min=[0, 170_000,250_000,np.inf]
labels_monto_min=['Hasta S/170K', 'Hasta S/250K','Más de S/ 250K']

pd.cut(datos['monto_min'],bins=bins_monto_min, labels=labels_monto_min)

datos['subsegmento_monto_ofertado_sin_deuda']=np.where(datos['cluster_comercial'].isin(["Próximo Recurrente","Nuevo en Expansión"]),
         pd.cut(datos['monto_min'],bins=bins_monto_min, labels=labels_monto_min, right=False),
         np.nan)

datos.query("subsegmento_monto_ofertado_sin_deuda.notna()").cluster_comercial.value_counts(dropna=False)
datos.query("subsegmento_monto_ofertado_sin_deuda.isna()").cluster_comercial.value_counts(dropna=False)

#---- 5.3.2) Emprendendor Izipay ----

bins_pasivos=[0, 1_000,20_000,np.inf]
labels_pasivos=['Hasta S/ 1K', 'Hasta S/ 20K','Más de S/ 20K']
datos.query('pasivosprom_u12m==0')

pd.cut(datos['pasivosprom_u12m'],bins=bins_pasivos, labels=labels_pasivos,right=False).value_counts(dropna=False)

datos['subsegmento_pasivos_sin_deuda']=np.where(datos['cluster_comercial'].isin(["Emprendedor Izipay"]),
        pd.cut(datos['pasivosprom_u12m'],bins=bins_pasivos, labels=labels_pasivos,right=False),
         np.nan)

datos.query("subsegmento_pasivos_sin_deuda.notna()").cluster_comercial.value_counts(dropna=False)
datos.query("subsegmento_pasivos_sin_deuda.isna()").cluster_comercial.value_counts(dropna=False)

#---- 5.3.3) Cliente Ancla & Cuenta Estrategica & Maduro Multibanco ----

bins_monto_min=[0, 170_000, 250_000, 350_000,np.inf]
labels_monto_min=['Hasta S/170K', 'Hasta S/250K','Hasta S/350K','Más de S/ 350K']

pd.cut(datos['monto_min'],bins=bins_monto_min, labels=labels_monto_min)

datos['subsegmento_monto_ofertado_con_deuda']=np.where(datos['cluster_comercial'].isin(["Cliente Ancla","Cuenta Estratégica","Maduro Multibanco"]),
         pd.cut(datos['monto_min'],bins=bins_monto_min, labels=labels_monto_min, right=False),
         np.nan)

datos.query("subsegmento_monto_ofertado_con_deuda.notna()").cluster_comercial.value_counts(dropna=False)
datos.query("subsegmento_monto_ofertado_con_deuda.isna()").cluster_comercial.value_counts(dropna=False)

#---- 5.3.4) Socio en Crecimiento y Gran Cuenta por Conquistar ----

bins_pasivos=[0, 1_000,5_000,np.inf]
labels_pasivos=['Hasta S/ 1K', 'Hasta S/ 5K','Más de S/ 5K']

datos['subsegmento_pasivos_con_deuda']=np.where(datos['cluster_comercial'].isin(["Socio en Crecimiento","Cliente por Reactivar"]),
        pd.cut(datos['pasivosprom_u12m'],bins=bins_pasivos, labels=labels_pasivos,right=False),
         np.nan)

datos.query("subsegmento_pasivos_con_deuda.notna()").cluster_comercial.value_counts(dropna=False)
datos.query("subsegmento_pasivos_con_deuda.isna()").cluster_comercial.value_counts(dropna=False)

#---- 5.3.5) Gran Cuenta por Conquistar ----

bins_deuda_sf=[0, 1_500_000,3_800_000,np.inf]
labels_deuda_sf=['Hasta S/1.5 MM', 'Hasta S/3.8 MM','Más de S/3.8 MM']

pd.cut(datos['deuda_directa_vig_ibk_sf_u12m'],bins=bins_deuda_sf, labels=labels_deuda_sf,right=False)

datos['subsegmento_deuda_sf_con_deuda']=np.where(datos['cluster_comercial'].isin(["Gran Cuenta por Conquistar"]),
        pd.cut(datos['deuda_directa_vig_ibk_sf_u12m'],bins=bins_deuda_sf, labels=labels_deuda_sf,right=False),
        np.nan)

datos.query("subsegmento_deuda_sf_con_deuda.notna()").cluster_comercial.value_counts(dropna=False)
datos.query("subsegmento_deuda_sf_con_deuda.isna()").cluster_comercial.value_counts(dropna=False)

#---- 5.3.6) Subsegmento ----

columnas_subsegmentos=['subsegmento_monto_ofertado_sin_deuda','subsegmento_pasivos_sin_deuda','subsegmento_monto_ofertado_con_deuda','subsegmento_pasivos_con_deuda','subsegmento_deuda_sf_con_deuda']

datos['subsegmento']=datos[columnas_subsegmentos].bfill(axis=1).iloc[:, 0]

#---- 5.3.7) Recodificación del Segmento ----

datos['cluster_comercial']=datos['cluster_comercial'].astype(str)+ ' & ' + datos['subsegmento'].astype(str)

#____________________ 6) Validaciones ----

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

#~~~~ 6.1) Duplicidad: llave_interna ----

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
# Debemos llegar a una tasa de mercado única para cada bucket

#~~~~ 6.2) Análisis Completitud: Tasa de Mercado en el SF ----

# Analizamos la completitud de la inferencia de Tasa de Mercado sin hacer filtros
# por entidades financieras, de interés o de mayor cobertura/share of sales.

#---- 6.2.1) Descriptivos Generales ----

datos.query('tasa_ultima.notna()').shape[0]/datos.shape[0]

# Inferencia en IBK
datos.query("flg_exito_4 in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()")
datos.query("flg_exito_4 in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()").cod_instit_financiera.value_counts(dropna=False)

# Inferencia en el SF
datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()")
datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()").cod_instit_financiera.value_counts(dropna=False)

(
    datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()").shape[0]/
    datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] ").shape[0]
)

#---- 6.2.2) Visualizaciones ----

#____ By Cluster Comercial ----
aux_completitud_inferencia_fracasos_by_cluster = (
    datos.query("flg_exito_4 != 'exito_desembolso_ibk_ct'")
    .groupby('cluster_comercial')
    .agg(
        completitud=('tasa_ultima', lambda x: x.notna().mean()),
        registros=('llave_interna', 'size'),
    )
    .reset_index()
)

aux_completitud_inferencia_fracasos_by_cluster['bloque_cluster']=aux_completitud_inferencia_fracasos_by_cluster['cluster_comercial'].isin(['Emprendedor Izipay','Nuevo en Expansión','Próximo Recurrente']).map({True:'Sin Deuda', False:'Con Deuda'})

graficos.plot_completitud_por_bloques_y_materialidad(
    df=aux_completitud_inferencia_fracasos_by_cluster,
    columna_cluster='cluster_comercial',
    columna_bloque='bloque_cluster',
    metrica_completitud='completitud',
    columna_materialidad='registros',
    titulo=(
        "Completitud de la Tasa de Mercado (Sin Filtro por Entidad"
        " Financiera)\nHorizonte: Mar' 25 - May' 26"
    )
)

#____ By Periodo ----

aux_completitud_inferencia_fracasos_by_periodo = (
    datos.query("flg_exito_4 != 'exito_desembolso_ibk_ct'")
    .groupby('periodo')
    .agg(
        completitud=('tasa_ultima', lambda x: x.notna().mean()),
        registros=('llave_interna', 'size'),
    )
    .reset_index()
)

graficos.plot_completitud_por_periodo_y_materialidad(
    df=aux_completitud_inferencia_fracasos_by_periodo,
    columna_periodo='periodo',
    metrica_completitud='completitud',
    columna_materialidad='registros',
    titulo=(
        'Evolución: Completitud de la Tasa de Mercado (Sin Filtro por Entidad'
        ' Financiera)\nCartera Total'
    )
)

#____ By Cluster y Periodo ----

aux_completitud_inferencia_fracasos_by_periodo_cluster = (
    datos.query("flg_exito_4 != 'exito_desembolso_ibk_ct'")
    .groupby(['periodo','cluster_comercial'])
    .agg(
        completitud=('tasa_ultima', lambda x: x.notna().mean()),
        registros=('llave_interna', 'size'),
    )
    .reset_index()
)

graficos.plot_evolutivo_multiserie_panel(
    df=aux_completitud_inferencia_fracasos_by_periodo_cluster,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica='completitud',
    label_metrica='Completitud (%)',
    titulo=(
            'Evolución: Completitud de la Tasa de Mercado (Sin Filtro por'
            ' Entidad Financiera)\nPor Clúster Comercial'
        )
)

for cluster in aux_completitud_inferencia_fracasos_by_periodo_cluster.cluster_comercial.unique():
    
    graficos.plot_completitud_por_periodo_y_materialidad(
    df=aux_completitud_inferencia_fracasos_by_periodo_cluster.query(f'cluster_comercial=="{cluster}"'),
    columna_periodo='periodo',
    metrica_completitud='completitud',
    columna_materialidad='registros',
    titulo=(
        "Evolución: Completitud de la Tasa de Mercado (Sin Filtro por Entidad"
        f" Financiera)\n{cluster}"
    )
    )

#~~~~ 6.3) Pareto: Entidades con inferencia ----

datos['bloque_cluster']=datos['cluster_comercial'].isin(['Emprendedor Izipay','Nuevo en Expansión','Próximo Recurrente']).map({True:'Sin Deuda', False:'Con Deuda'})

#---- 6.3.1) Pareto General ----

aux_pareto=(
    datos.query(
        "flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()"
    )
    .groupby('cod_instit_financiera')
    .size()
    .reset_index(name='registros')
    .sort_values(by='registros', ascending=False)
    .assign(
        proporcion=lambda df: df['registros'] / df['registros'].sum(),
        cumsum_proporcion=lambda df: df['proporcion'].cumsum(),
    )
    .reset_index(drop=True)
)

graficos.plot_pareto_instituciones(
    aux_pareto,
    columna_institucion='cod_instit_financiera',
    columna_registros='registros',
    titulo=(
        'Diagrama de Pareto: Inferencia de Tasa de Mercado '
        " por Entidad Financiera\nCartera Total\nHorizonte: Mar' 25 -"
        " May' 26"
    )
)

#---- 6.3.2) Pareto: Sin/Con Deuda ----

aux_pareto_by_bloque_cluster = (
    datos.query(
        "flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()"
    )
    .groupby(['bloque_cluster', 'cod_instit_financiera'])
    .size()
    .reset_index(name='registros')
    .sort_values(
        by=['bloque_cluster', 'registros'], ascending=[True, False]
    )
    .assign(
        proporcion=lambda df: df.groupby('bloque_cluster')[
            'registros'
        ].transform(lambda x: x / x.sum()),
        cumsum_proporcion=lambda df: df.groupby('bloque_cluster')[
            'proporcion'
        ].cumsum(),
    )
    .reset_index(drop=True)
)

for bloque_cluster in aux_pareto_by_bloque_cluster.bloque_cluster.unique():
    
    graficos.plot_pareto_instituciones(
        aux_pareto_by_bloque_cluster.query(f"bloque_cluster=='{bloque_cluster}'"),
        columna_institucion='cod_instit_financiera',
        columna_registros='registros',
        titulo=(
            'Diagrama de Pareto: Inferencia de Tasa de Mercado '
            f" por Entidad Financiera\n{bloque_cluster}\nHorizonte: Mar' 25 -"
            " May' 26"
        )
    )

#---- 6.3.3) Pareto: Cluster ----

aux_pareto_by_cluster=(
    datos.query(
        "flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()"
    )
    .groupby(['cluster_comercial', 'cod_instit_financiera'])
    .size()
    .reset_index(name='registros')
    .sort_values(
        by=['cluster_comercial', 'registros'], ascending=[True, False]
    )
    .assign(
        proporcion=lambda df: df.groupby('cluster_comercial')[
            'registros'
        ].transform(lambda x: x / x.sum()),
        cumsum_proporcion=lambda df: df.groupby('cluster_comercial')[
            'proporcion'
        ].cumsum(),
    )
    .reset_index(drop=True)
)

for cluster in aux_pareto_by_cluster.cluster_comercial.unique():
    
    graficos.plot_pareto_instituciones(
        aux_pareto_by_cluster.query(f"cluster_comercial=='{cluster}'"),
        columna_institucion='cod_instit_financiera',
        columna_registros='registros',
        titulo=(
            'Diagrama de Pareto: Inferencia de Tasa de Mercado '
            f" por Entidad Financiera\n{cluster}\nHorizonte: Mar' 25 -"
            " May' 26"
        )
    )

#---- 6.3.4) Pareto: Periodo ----

aux_pareto_by_periodo=(
    datos.query(
        "flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()"
    )
    .groupby(['cod_instit_financiera', 'periodo'])
    .size()
    .reset_index(name='registros')
    .sort_values(
        by=['cod_instit_financiera', 'registros'], ascending=[True, False]
    )
    .assign(
        proporcion=lambda df: df.groupby('periodo')[
            'registros'
        ].transform(lambda x: x / x.sum()),
        cumsum_proporcion=lambda df: df.groupby('periodo')[
            'proporcion'
        ].cumsum(),
    )
    .reset_index(drop=True)
)

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

for periodo in aux_pareto_by_periodo.periodo.unique():

    anio = int(periodo) // 100
    mes = int(periodo) % 100
    periodo_label = f"{meses[mes]}' {str(anio)[-2:]}"

    graficos.plot_pareto_instituciones(
        aux_pareto_by_periodo.query(f"periodo=={periodo}"),
        columna_institucion='cod_instit_financiera',
        columna_registros='registros',
        titulo=(
            'Diagrama de Pareto: Inferencia de Tasa de Mercado '
            f" por Entidad Financiera\nCartera Total\nPeriodo: {periodo_label}"
        )
    )

#____________________ 7) Funnel: Entidades Financieras ----

# Conservamos las entidades que concentran la mayor parte de las inferencias de tasa:
# 1: BCP
# 6: BBVA
# 4: SCOTIA BANK
# 123: MIBANCO

cod_entidades_inferencia=[1.0,6.0,4.0,123.0]

cod_entidades_inferencia=[1.0,
 6.0,
 4.0,
 103.0,
 123.0,
 22.0,
 106.0,
 148.0,
 107.0,
 114.0,
 228.0,
 104.0,
 108.0,
 55.0,
 224.0,
 185.0,
 112.0,
 231.0,
 184.0,
 109.0,
 227.0,
 102.0,
 171.0,
 170.0,
 172.0,
 169.0]

datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()")
datos_tasa_mercado=datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna() and cod_instit_financiera in @cod_entidades_inferencia")

datos_tasa_mercado['tasa_ultima'].describe(percentiles=[0.01,0.05,0.10, 0.25, 0.50, 0.75, 0.90,0.95,0.99])

#____________________ 8) Procesamiento ----

#~~~~ 8.1) Winsorizacion: Inferencia de Tasa ----

UMBRAL_WINSORIZACION=0.025

limites_winsor = datos_tasa_mercado.groupby('cluster_comercial')['tasa_ultima'].agg(
    limite_inferior=lambda x: x.quantile(UMBRAL_WINSORIZACION),
    limite_superior=lambda x: x.quantile(1 - UMBRAL_WINSORIZACION)
).reset_index()


datos_tasa_mercado

datos_tasa_mercado = datos_tasa_mercado.merge(limites_winsor, on='cluster_comercial', how='left')

datos_tasa_mercado['tasa_mercado_winsorizada'] = datos_tasa_mercado['tasa_ultima'].clip(
    lower=datos_tasa_mercado['limite_inferior'], 
    upper=datos_tasa_mercado['limite_superior']
)

datos_tasa_mercado.query('tasa_mercado_winsorizada!=tasa_ultima')

# datos_tasa_mercado=datos_tasa_mercado.query('tasa_mercado_winsorizada==tasa_ultima')

#~~~~ 8.2) Data panel: Por Entidad ----

datos_tasa_mercado['nombre_ifi']=datos_tasa_mercado['cod_instit_financiera'].map({1.0:'BCP', 6.0: 'BBVA', 4.0:'SCOTIA', 123.0:'MI BANCO'}).fillna('OTROS')

datos_tasa_mercado.value_counts(['nombre_ifi','cod_instit_financiera']).reset_index()

# Promedio ponderado seguro
prom_pond = lambda v, w: (
    np.average(v[v.notna() & w.notna() & (w > 0)], weights=w[v.notna() & w.notna() & (w > 0)])
    if (v.notna() & w.notna() & (w > 0)).any()
    else np.nan
)

data_panel_mercado_by_entidad = (
    datos_tasa_mercado
    .assign(
        tasa_x_monto=lambda df: df['tasa_mercado_winsorizada'] * df['delta_deuda_final']
    )
    .groupby(['cluster','cluster_comercial', 'nombre_ifi', 'periodo'], observed=True)
    .agg(
        desembolsos=('llave_interna', 'size'),
        monto_desembolsado=('delta_deuda_final', 'sum'),
        suma_tasa_x_monto=('tasa_x_monto', 'sum'),
        tasa_mercado=(
            'tasa_mercado_winsorizada',
            lambda x: prom_pond(
                x, 
                datos_tasa_mercado.loc[x.index, 'delta_deuda_final']
            )
        )
    )
    .reset_index()
)    

#~~~~ 8.3) Data panel: Mercado ----

data_panel_mercado = (
    data_panel_mercado_by_entidad
    .groupby(['cluster_comercial', 'periodo'], observed=True)
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

#____________________ 9) EDA ----

#~~~~ 9.1) Evolutivos: Tasa Mercado ----

# Tasa de Mercado por segmento
graficos.plot_evolutivo_multiserie_panel(
    df=data_panel_mercado,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica='tasa_mercado',
    label_metrica='Tasa Mercado (%)',
    titulo=(
            'Evolución: Tasa de Mercado (4 Principales Bancos)\nPor Clúster Comercial'
        )
)

# Tasa de cada institución y su aporte, por segmento
for cluster in data_panel_mercado_by_entidad.cluster_comercial.unique():

    graficos.plot_evolutivo_multiserie_con_barras_apiladas(
        df=data_panel_mercado_by_entidad.query(f"cluster_comercial=='{cluster}'"),
        columna_periodo='periodo',
        columna_grupo='nombre_ifi',
        label_grupo='Institución Financiera',
        metrica='tasa_mercado',
        label_metrica='Tasa de Mercado (%)',
        columna_materialidad='monto_desembolsado',
        modo_materialidad='porcentaje',
        titulo=(
                f'Evolución: Tasa de Mercado (4 Principales Bancos)\n{cluster}'
            )
    )

#____________________ 10) Almacenamiento ----

data_panel_mercado_by_entidad.to_csv(dir_b+r'/processed/V01/base_sensibilidad_componente_mercado.csv',sep='|', index=False)
