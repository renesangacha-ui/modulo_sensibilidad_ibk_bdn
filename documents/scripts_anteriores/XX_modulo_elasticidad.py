################################## Procesamiento Informacion ############################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tProcesamiento Informacion')
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


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
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

datos=pd.read_csv(dir_b+r'/raw/V02/data_fin_no_clientes.csv',sep=',')

datos.head().T
datos.info()
datos.isnull().mean().sort_values(ascending=False)
datos.describe()
datos.periodo.min(), datos.periodo.max()

#____________________ 4) Validaciones ----

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

#~~~~ 5.1) Completitud ----

datos.isnull().mean().sort_values(ascending=False)

#---- 5.1.1) Validaciones: Idoneidad Completitud ----

# Nulidad: Información de desembolsos
datos.query('tasa_2022.isna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.isna()').flg_exito_4.value_counts(dropna=False)

datos.query('tasa_2022.notna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.notna()').flg_exito_4.value_counts(dropna=False)
datos.query('tasa_2022.notna() or monto_desembolsado.notna()').flg_exito_4.value_counts(dropna=False)

# Se observan fracasos con infotmación de desembolsos, tasa y monto, en teoria no deberian tener información
# Dentro de los que tienen información se encuentran los fracasos del tipo desembolso ibk pero no ct, pero estan fracasos
# en otras entidades, posiblemenete son leads que desembolsaron en mas de una entidad y los desembolsso en otros lados se ha pegado la
# tasa información de IBK, en ese caso dado que hay dos tipos de desemsbolso en ibk de interes (ct) y los que no (no ct), algo importante
# en ese caso, cual tomar en caso de que un cliente tenga mas de uno, el de ct, porque acepto el que me interesaba, para la info de desembolsos
# en este caso no se tiene eso, cada lead tiene un unico desembolsso en el mes de un solo producto, por lo que no es necesario, en caso de tener se puede aplicar este caso
# en este caso, al cliente se le oferto un producto pero desembolso en otro, si en el caso de que desembolso enmnas de uno y de ellos sea el de interes, se toma la info de desembolsos
# del de interes,  cc, no se necesita tomar una info de desembolsso de interes porque no es producyo dfe interes

(
    datos
    .query(f'flg_exito_4 in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"]')
    .llave_interna
    .value_counts(dropna=False)
    .reset_index()
    .value_counts('count',dropna=False)    
)

datos.query(f'flg_exito_4 in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"]').shape
datos.value_counts(['flg_exito_4','flg_desembolso_ibk_all_products'],dropna=False).reset_index()

# Validamos la hipotesis, de que los leads desembolsados en otro lado y tienen info de desembolsos, es oprqu qdemas tienen un desembolsos en ibk, es decir desembolsaron en ibk  y en otra entidad
desembolsos_competencia_validacion=(
    datos
    .query('(monto_desembolsado.notna() or tasa_2022.notna()) and flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"]')
    .llave_interna
    .drop_duplicates()
)

casos_no_validados=pd.merge(
    desembolsos_competencia_validacion,
    datos.query(f'flg_exito_4 in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"]')[['llave_interna']],
    how='left',
    on=['llave_interna'],
    indicator='match'
).query('match!="both"')

casos_no_validados

datos_casos_no_validados=pd.merge(
    datos,
    casos_no_validados,
    how='inner',
    on='llave_interna' 
)

datos_casos_no_validados.FLG_CLIENTE.value_counts(dropna=False)
datos_casos_no_validados.flg_exito_4.value_counts(dropna=False)
datos_casos_no_validados.cluster.value_counts(dropna=False)
datos_casos_no_validados.cod_instit_financiera.value_counts(dropna=False)
datos_casos_no_validados.periodo.value_counts(dropna=False)

# hay 835 de 911 que si tienen un desembolso en ibk, pero hay otros 76 que no tienen info en esta base de desembolso en ibk, pueden ser por que son del flujo regular, por lo tanto, no tienen información de lead, esto por su magnitud, es la hipotesis
# habria que validar o son desembolsos que no aparecen en mis exitos, pero si en la base anterior, algo que se debe validar

# Conclusion de la validacion componente interna: Hay registros no desembolsados en ibk que tienen información de desembolso, esto se debe en su mayoria, 91.7%, porque la persona que desembolso en otro lado, tambien lo ha hecho en IBK y se ha puesto la info
# de desembolso del de IBk, esto se debe limpiar, donde para los restantes en donde no se puede corrobar lo anterior, con los desembolsos de ibnk actuale, 76, cabe validar el motivo del porque tienen info de desembolso su lead, es porque se han filtrado los desembolsos 
# IBK tal vez? consultar, al ser 76 se pueden eliminar, no hay un comportamiento o caravetristica que comparten estos clientes, en el tiempo o por su perfil, por lo cual se debe ahondar más porque estos clientes tienen información de desembolso, o que se agrupen en un mes especifico

#~~~~ 5.3) Idoneidad de informacion ----

datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").tasa_2022.describe()
datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").monto_desembolsado.describe()


datos.monto_min.isna().mean()
datos.monto_min.describe()
datos.query('monto_min==0').T

datos.tasa_min.isna().mean() 
datos.tasa_min.describe()
datos.query('tasa_min==0').T

#~~~~ 5.3) Desembolso vs Oferta: Monto ----

datos.query("monto_desembolsado.notna() and monto_desembolsado > monto_min")

(
    datos.query("monto_desembolsado.notna() and monto_desembolsado > monto_min")
    .assign(diferencia=lambda x: x["monto_desembolsado"] - x["monto_min"])
    ["diferencia"]
    .describe(percentiles=[i / 100 for i in range(1, 100)])
)

# El 80% tieen diferencias menores a 20 soles, el 20% tieene diferencia superfiores a 1897 soles

# Se conluye de la informacion informacion interna, de leads y desembolsos, que 
# la tasa debe reescalrse en unidades para trabajr en los mimsom niveles que tasa ofertada, que todo cliente desembolsado tiene monto y tasas,
# QUe todo cliente tiene informaicon de tasa y monto ofertado, hay dos clientes con tasa o monto cero, no hace sentido, pero son clientes d
# fracaso_desembolso_ibk_no_ct, que no entran propiamente al analisis de competencia

# hay desembolsos en la competencia que tienene tasas desembolsada y monto desembolsada en IBK, lo cual de primera manera no es posible, esto se debee l 90% de las veces porque ese cliente
# ademas desembolso en IBK y esa es la tasa que se ha pagado, en esre caso como hay dos tipos de desembolsos en IBK (ct, de interes,  y no ct, de no interes), al lead se debe dar en caso de más de un desembolso, en el mes, se debe pegar el de CT,
# porque el cliente acepto el que yo le ofereci de interes, si tienen solo los no ct, no se debe pegar la info de desembolso al lead de este portque no sopn desembolsos a a anlizar,seria un fracaso, esto no sucede aca, pero a futuro se debe considerar,
# hay 76 creditos en la competencia que no cumplen que 
# tengan un desembolsos en IBK para que tengan info de IBK desembolso, hay que entender mas el porque esto sucede, son 76 creditos que en teoria no deberian tener info y se escapan de la hipotesis aux_completitud_inferencia_fracasos_by_periodo_cluster#


# para la base de oferta relevante, se haran los tratamientos respectivos en monto ofertado<monto desembolsado para los exitos IBK, se limpiara la tasa y monto desembolsado de los fracasos ya que no deben tener, con fin de evitar inconsistencia a futuro, eso en la base uno a uno
# de leads/desembolso, asi como las recodifciaciones, categorizaciones y reescalamientos,  para la base uno a uno que se use en los demas analisis 

#~~~~ Validacion Mercado ----

# Los desembolsos en IBK solo tienen una entidad, 
datos.value_counts(['cod_entidad','flg_desembolso_ibk_all_products'],dropna=False).reset_index()

datos.query('cod_entidad!=2')

datos.query('cod_entidad!=2')

datos.llave_interna.value_counts(dropna=False).reset_index().value_counts('count',dropna=False, normalize=True)
# El 95.62% de los leads aparece una vez, los demás tienen más de un desembolso en el SSFF, 
# es un comportamiento esperado, para el tema de efectividad, el lead solo ingresa una unica vez, y responde si desembolsa o no, a diferencia del Shgare of sales
# donde se anlizan todos los desembolsos en la competencia, en caso de que acepte ibk y otros bancos, para mi ya es exuto porque ya desembolso con migo, sino lo hizo, pero en al menos es de interes (oferta relevante)
# basta para considerar como fracaso, porque yo pude ser uno de los bancos donde desembolso pero no lo hizo

datos.query('flg_exito_4 in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"]').tasa_ultima.isnull().mean()
datos.query('flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"]').tasa_ultima.isnull().mean()

datos.query('flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"] and tasa_ultima.notna()').tasa_ultima.describe()
datos.query('flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"] and tasa_ultima.notna()').delta_deuda_final.describe()
datos.query('flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"] and tasa_ultima.notna() and tasa_ultima>=1')
datos.query('flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"] and tasa_ultima.notna()').delta_deuda_final.isnull().mean()


(
    datos.query('flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"]')
    .cod_entidad
    .value_counts(dropna=False, normalize=True)
    .cumsum()
)

(
    datos.query('flg_exito_4 not in ["exito_desembolso_ibk_ct", "fracaso_desembolso_ibk_no_ct"] and tasa_ultima.notna()')
    .cod_entidad
    .value_counts(dropna=False, normalize=True)
    .cumsum()
)


# Conclusion: Nos quedamos con los desembolsos de la competencia, de estos los que tienen inferencia, hay clientes con inferencia de tasa mayor a 1, y muy cercaca a 0, por lo cual se propone winsorizar la tasa de la competencia, acorde a los grupos que se desee modelar, por ejemplo
# por macroperfil, de forma general, ...

#~~~~ 5.2) Duplicidad ----

# llave_interna: id_lead + periodo
datos.value_counts(['llave_interna']).reset_index().value_counts('count', normalize=True)

registros_duplicados=datos.value_counts(['llave_interna']).reset_index().query('count>1').llave_interna.to_list()

validacion_duplicidad=(
    datos
    .query("llave_interna in @registros_duplicados")
    .groupby(['llave_interna'], observed=True)
    .agg(registros=('llave_interna','size'),
        tasa_min=('tasa_min', 'nunique'),
        monto_min=('monto_min', 'nunique'),
        cluster=('cluster', 'nunique'),
        cod_instit_financiera=('cod_instit_financiera', 'nunique'),
        flg_exito_4=('flg_exito_4', 'nunique'),
        tasa_2022=('tasa_2022','nunique'),
        monto_desembolsado=('monto_desembolsado','nunique'),
        FLG_CLIENTE=('FLG_CLIENTE','nunique'))

)

validacion_duplicidad

validacion_duplicidad.query('tasa_min>1 or monto_min>1 or cluster>1 or tasa_2022>1 or monto_desembolsado>1 or FLG_CLIENTE>1')

validacion_duplicidad.query('cod_instit_financiera>1 or flg_exito_4>1')

validacion_duplicidad.query('flg_exito_4==1 and cod_instit_financiera==1')

# Unicidad a nivel cliente: Ningún cliente tiene más de una tasa o monto
# ofertado/desembolsado, ni pertenece a más de un clúster o tipo de cliente.
#
# La duplicidad de registros se debe a que el lead realizó más de un desembolso
# en el SSFF en un mes

#____________________ 5) Procesamiento Inicial ----

#~~~~ 5.1) Recodificaciones ----

#---- 5.1.1) Cluster Comercial ----

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

#---- 5.1.2) Tipo Cliente ----

mapeo_tipo_cliente={
   1: 'Cliente',
   0: 'No Cliente'
}

datos['tipo_cliente'] = datos['FLG_CLIENTE'].map(mapeo_tipo_cliente).fillna(datos['FLG_CLIENTE'])
datos[['tipo_cliente','FLG_CLIENTE']].value_counts().reset_index()

#---- 5.1.3) Cluster Comercial Aperturado ----

datos['cluster_comercial_aperturado']=datos['cluster_comercial'].astype(str)+' & '+ datos['tipo_cliente'].astype(str)

datos['cluster_comercial_aperturado'].value_counts()

#~~~~ 5.2) Tratamiento: Variables Desembolso ----

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

#~~~~ 5.3) Tratamiento: Monto ofertado ----

datos['monto_ofertado_ajustado'] = np.where(
    ~datos['flg_exito_4'].isin(['exito_desembolso_ibk_ct','fracaso_desembolso_ibk_no_ct']),
    datos['monto_min'],
    np.where(
        datos['monto_desembolsado'] > datos['monto_min'],
        datos['monto_desembolsado'],
        datos['monto_min']
    )
)

#~~~~ 5.4) Categorizaciones ----

datos['bloque_cluster']=datos['cluster_comercial'].isin(['Emprendedor Izipay','Nuevo en Expansión','Próximo Recurrente']).map({True:'Sin Deuda', False:'Con Deuda'})
datos.value_counts(['bloque_cluster','cluster_comercial']).reset_index()

datos.query('tasa_desembolsada.notna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.notna()').flg_exito_4.value_counts(dropna=False)

datos.query('tasa_desembolsada.isna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.isna()').flg_exito_4.value_counts(dropna=False)

#____________________ 4) Base Tratada ----

datos.to_csv(dir_b+r'/processed/V03/base_clientes_tratada.csv', index=False, sep='|')

#____________________ 4) Parametros ----

# En base al proceso de exploración de la información se han decidido los siguientes parametros
# De materialidad, disponibilidad de informacion, asi como de comportamiento observado

PERIODO_INICIO=202504
PERIODO_FIN=202604

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

data_panel_sensibilidad=data_panel_sensibilidad.query(f'{PERIODO_FIN}>=periodo>={PERIODO_INICIO}')

recategorizacion_clusters={
   'Fidelizado & Cliente':'Fidelizado',
   'Grandes por fidelizar & Cliente': 'Grandes por fidelizar',
    'Medianos por Fidelizar & Cliente':'Medianos por Fidelizar',
  'Sin historial crediticio & Cliente': 'Sin historial crediticio'}

data_panel_sensibilidad['cluster_modelar']=data_panel_sensibilidad['cluster_modelar'].map(recategorizacion_clusters)

# Asegúrate de inicializar tu DataFrame acumulado antes del bucle si no lo tienes:
resultados_globales_rolling = pd.DataFrame()

for periodo in [202512, 202601, 202602, 202603,202604]:

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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    df_control.dropna(subset=['cluster', 'periodo', 'elasticidad'])
    .sort_values(['cluster', 'periodo'])
    .copy()
)

# Constantes clásicas I-MR para rangos móviles de 2 observaciones
D4 = 3.267
A2_MR = 2.66

# Paleta corporativa de storytelling
COLOR_AZUL = '#1D4290'
COLOR_ROJO = '#E63946'  # Rojo ejecutivo elegante para alertas

# ============================================================
# 2. GRAFICAR I-MR POR CLUSTER (STORYTELLING DE NEGOCIO)
# ============================================================

for cluster, data in df_control.groupby('cluster'):

  data = data.sort_values('periodo').copy()

  if len(data) < 2:
    print(f'{cluster}: insuficientes estimaciones para I-MR.')
    continue

  # Cálculos estadísticos de control
  data['MR'] = data['elasticidad'].diff().abs()

  promedio = data['elasticidad'].mean()
  mr_promedio = data['MR'].mean()

  lsc_i = promedio + A2_MR * mr_promedio
  lic_i = promedio - A2_MR * mr_promedio

  lc_mr = mr_promedio
  lsc_mr = D4 * mr_promedio
  lic_mr = 0.0

  data['alerta_I'] = (data['elasticidad'] > lsc_i) | (
      data['elasticidad'] < lic_i
  )
  data['alerta_MR'] = data['MR'] > lsc_mr

  etiquetas = (
      pd.to_datetime(
          data['periodo'].astype('Int64').astype(str),
          format='%Y%m',
          errors='coerce',
      )
      .dt.strftime('%b-%y')
      .values
  )

  x = np.arange(len(data))

  # ========================================================
  # GRÁFICO 1: I - ELASTICIDAD
  # ========================================================
  fig, ax = plt.subplots(figsize=(10, 5))

  ax.plot(
      x,
      data['elasticidad'],
      color=COLOR_AZUL,
      marker='o',
      markersize=5,
      linewidth=1.8,
      zorder=3,
      label='Elasticidad Real (Tendencia)',
  )

  ax.axhline(
      promedio,
      color='#555555',
      linestyle='-',
      linewidth=1.2,
      alpha=0.8,
      label=f'Línea Base (Media Histórica: {promedio:.3f})',
  )
  ax.axhline(
      lsc_i,
      color=COLOR_ROJO,
      linestyle='--',
      linewidth=1.2,
      alpha=0.8,
      label=f'Límite Superior Control ({lsc_i:.3f})',
  )
  ax.axhline(
      lic_i,
      color=COLOR_ROJO,
      linestyle='--',
      linewidth=1.2,
      alpha=0.8,
      label=f'Límite Inferior Control ({lic_i:.3f})',
  )

  alertas_i = data['alerta_I'].to_numpy()
  if alertas_i.any():
    ax.scatter(
        x[alertas_i],
        data.loc[data['alerta_I'], 'elasticidad'],
        color=COLOR_ROJO,
        marker='X',
        s=90,
        zorder=4,
        label='Desviación Crítica (Fuera de Control)',
    )

  # Tags en TODOS los periodos (sin negrita, con colores dinámicos)
  for xi, val, es_alerta in zip(
      x, data['elasticidad'].values, data['alerta_I'].values
  ):
    color_borde = COLOR_ROJO if es_alerta else COLOR_AZUL
    color_fondo = '#FADBD8' if es_alerta else '#E8F1F5'

    ax.annotate(
        f'{val:.3f}',
        (xi, val),
        xytext=(0, 8),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=7,
        fontweight='normal',
        color='black',
        bbox=dict(
            boxstyle='round,pad=0.2',
            fc=color_fondo,
            ec=color_borde,
            lw=0.8,
            alpha=0.9,
        ),
    )

  # Limpieza visual ejecutiva
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_color('#CCCCCC')

  ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
  ax.set_xticks(x)
  ax.set_xticklabels(etiquetas, fontsize=8, color='black')
  ax.tick_params(axis='y', labelsize=8)

  ax.set_title(
      f'Monitoreo de Estabilidad de Elasticidad | {cluster}',
      fontsize=11,
      fontweight='bold',
      color=COLOR_AZUL,
      pad=15,
  )
  # Leyenda movida a la esquina inferior izquierda
  ax.legend(
      loc='lower left',
      frameon=True,
      facecolor='white',
      edgecolor='lightgray',
      fontsize=7.5,
  )
  plt.tight_layout()
  plt.show()

  # ========================================================
  # GRÁFICO 2: MR - CAMBIO ENTRE VENTANAS
  # ========================================================
  fig, ax = plt.subplots(figsize=(10, 5))

  mask_mr = data['MR'].notna()
  x_mr = x[mask_mr.to_numpy()]
  mr = data.loc[mask_mr, 'MR'].to_numpy()
  alertas_mr = data.loc[mask_mr, 'alerta_MR'].to_numpy()

  ax.plot(
      x_mr,
      mr,
      color=COLOR_AZUL,
      marker='o',
      markersize=5,
      linewidth=1.8,
      zorder=3,
      label='Variación Absoluta (Periodo a Periodo)',
  )

  ax.axhline(
      lc_mr,
      color='#555555',
      linestyle='-',
      linewidth=1.2,
      alpha=0.8,
      label=f'Variación Promedio Esperada ({lc_mr:.3f})',
  )
  ax.axhline(
      lsc_mr,
      color=COLOR_ROJO,
      linestyle='--',
      linewidth=1.2,
      alpha=0.8,
      label=f'Límite Superior de Volatilidad ({lsc_mr:.3f})',
  )
  ax.axhline(
      lic_mr, color='#555555', linestyle=':', linewidth=1, alpha=0.8
  )

  if alertas_mr.any():
    ax.scatter(
        x_mr[alertas_mr],
        mr[alertas_mr],
        color=COLOR_ROJO,
        marker='X',
        s=90,
        zorder=4,
        label='Alerta por Picos de Volatilidad',
    )

  # Tags en TODOS los puntos de MR sin negrita
  for xi, val, es_alerta in zip(x_mr, mr, alertas_mr):
    color_borde = COLOR_ROJO if es_alerta else COLOR_AZUL
    color_fondo = '#FADBD8' if es_alerta else '#E8F1F5'

    ax.annotate(
        f'{val:.3f}',
        (xi, val),
        xytext=(0, 8),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=7,
        fontweight='normal',
        color='black',
        bbox=dict(
            boxstyle='round,pad=0.2',
            fc=color_fondo,
            ec=color_borde,
            lw=0.8,
            alpha=0.9,
        ),
    )

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_color('#CCCCCC')

  ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
  ax.set_xticks(x)
  ax.set_xticklabels(etiquetas, fontsize=8, color='black')
  ax.tick_params(axis='y', labelsize=8)

  ax.set_title(
      f'Monitoreo de Volatilidad y Cambios (MR) | {cluster}',
      fontsize=11,
      fontweight='bold',
      color=COLOR_AZUL,
      pad=15,
  )
  # Leyenda movida a la esquina inferior izquierda
  ax.legend(
      loc='lower left',
      frameon=True,
      facecolor='white',
      edgecolor='lightgray',
      fontsize=7.5,
  )
  plt.tight_layout()
  plt.show()

for bucket in data_panel_sensibilidad['cluster_modelar'].unique():

    df_filtrado = data_panel_sensibilidad.query("cluster_modelar==@bucket")

    graficos.plot_evolutivo_precio_demanda(
        df=df_filtrado.query('periodo>=202602'),
        columna_demanda='efectividad',
        columna_precio='tasa_ofertada',
        columna_materialidad='desembolsos_ibk',
        titulo=f"Evolución de Tasa Ofertada y Efectividad\n{bucket}",
        flg_materialidad=False,
        flg_etiqueta_demanda=True,
        flg_etiqueta_precio=True
    )


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



resultados_globales_rolling.cluster.value_counts()
resultados_globales_rolling.query("cluster=='Fidelizado & Cliente'")

resultados_globales_rolling.query("periodo==202604")


data_panel_sensibilidad.periodo.value_counts()

(
    data_panel_sensibilidad
    .groupby('cluster_modelar')
    .agg(
        monto_desembolsado_ibk=('monto_desembolsado_ibk', 'sum'),
        desembolsos_ibk=('desembolsos_ibk', 'sum')
    )
    .assign(
        materialidad_monto_pct=lambda x: (
            x['monto_desembolsado_ibk']
            / x['monto_desembolsado_ibk'].sum() * 100
        ),
        materialidad_desembolsos_pct=lambda x: (
            x['desembolsos_ibk']
            / x['desembolsos_ibk'].sum() * 100
        )
    )
    .round(2)
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





import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Obtener los datos limpios en logaritmos del último periodo evaluado
# (asumiendo que 'df' es tu DataFrame ya filtrado y con log_efectividad y log_tasa)

# 2. Calcular la transformación de Efectos Fijos (Within Transformation) manualmente para graficar:
# Restar la media por bucket y por mes para ver el efecto neto que captura PanelOLS
df['log_efectividad_demeaned'] = (
    df['log_efectividad'] 
    - df.groupby('bucket')['log_efectividad'].transform('mean')
    - df.groupby('mes')['log_efectividad'].transform('mean')
    + df['log_efectividad'].mean()
)

df['log_tasa_demeaned'] = (
    df['log_tasa'] 
    - df.groupby('bucket')['log_tasa'].transform('mean')
    - df.groupby('mes')['log_tasa'].transform('mean')
    + df['log_tasa'].mean()
)

# 3. Gráfico de dispersión con la pendiente real del modelo (Relación Neta)
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, 
    x='log_tasa_demeaned', 
    y='log_efectividad_demeaned', 
    hue='bucket', 
    palette='tab10', 
    alpha=0.8,
    s=70
)

# Añadir línea de tendencia general o por grupo que refleja la elasticidad
sns.regplot(
    data=df, 
    x='log_tasa_demeaned', 
    y='log_efectividad_demeaned', 
    scatter=False, 
    color='#1D4290', 
    line_kws={'linewidth': 2, 'label': 'Pendiente estimada (Elasticidad TWFE)'}
)

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.title('Relación Neta (Espacio Within) - Lo que ve el PanelOLS', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Log(Tasa Ofertada) - Centrada por Efectos Fijos', fontsize=10)
plt.ylabel('Log(Efectividad) - Centrada por Efectos Fijos', fontsize=10)
plt.legend(title='Macroperfil', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

###################



# import numpy as np
# import pandas as pd
# from linearmodels.panel import PanelOLS

# # ============================================================
# # 1. INICIALIZAR RESULTADOS
# # ============================================================

# resultados_globales_rolling = pd.DataFrame()
# resultados_predicciones_rolling = pd.DataFrame()

# periodos_rolling = [202512, 202601, 202602, 202603, 202604]

# # ============================================================
# # 2. ESTIMACIÓN ROLLING
# # ============================================================

# for periodo in periodos_rolling:

#     print(f'\n{"=" * 80}')
#     print(f'PROCESANDO VENTANA HASTA {periodo}')
#     print(f'{"=" * 80}')

#     try:

#         # --------------------------------------------------------
#         # 2.1. PREPARAR BASE
#         # --------------------------------------------------------

#         df = data_panel_sensibilidad[
#             [
#                 'cluster_modelar',
#                 'periodo',
#                 'efectividad',
#                 'tasa_ofertada'
#             ]
#         ].copy()

#         df = df.query('periodo <= @periodo').copy()

#         df = df.rename(columns={
#             'cluster_modelar': 'bucket'
#         })

#         # Periodo YYYYMM -> fecha mensual
#         df['mes'] = pd.to_datetime(
#             df['periodo'].astype('Int64').astype(str),
#             format='%Y%m',
#             errors='coerce'
#         )

#         # Variables numéricas
#         for col in ['efectividad', 'tasa_ofertada']:
#             df[col] = pd.to_numeric(
#                 df[col],
#                 errors='coerce'
#             )

#         # Eliminar registros incompletos
#         df = df.dropna(
#             subset=[
#                 'bucket',
#                 'mes',
#                 'efectividad',
#                 'tasa_ofertada'
#             ]
#         ).copy()

#         # Los logaritmos requieren valores positivos
#         df = df.loc[
#             (df['efectividad'] > 0) &
#             (df['tasa_ofertada'] > 0)
#         ].copy()

#         # Validar que existan observaciones
#         if df.empty:
#             print('No hay observaciones válidas. Se omite la ventana.')
#             continue

#         # Transformaciones logarítmicas
#         df['log_efectividad'] = np.log(df['efectividad'])
#         df['log_tasa'] = np.log(df['tasa_ofertada'])

#         df['bucket'] = df['bucket'].astype(str)

#         print(f'Observaciones: {len(df):,}')
#         print(f'Buckets: {df["bucket"].nunique()}')
#         print(f'Meses: {df["mes"].nunique()}')

#         # --------------------------------------------------------
#         # 2.2. CREAR INTERACCIONES TASA x BUCKET
#         # --------------------------------------------------------

#         buckets = sorted(df['bucket'].unique())

#         nombres_interaccion = {}

#         for i, bucket in enumerate(buckets):

#             nombre_columna = f'tasa_inter_{i}'

#             df[nombre_columna] = (
#                 df['log_tasa'] *
#                 (df['bucket'] == bucket).astype(int)
#             )

#             nombres_interaccion[bucket] = nombre_columna

#         # --------------------------------------------------------
#         # 2.3. ESTIMAR PANELOLS CON EFECTOS FIJOS
#         # --------------------------------------------------------

#         df_panel = (
#             df.set_index(['bucket', 'mes'])
#               .sort_index()
#         )

#         columnas_x = list(nombres_interaccion.values())

#         modelo_panel = PanelOLS(
#             dependent=df_panel['log_efectividad'],
#             exog=df_panel[columnas_x],
#             entity_effects=True,
#             time_effects=True,
#             drop_absorbed=True
#         )

#         resultado_panel = modelo_panel.fit(
#             cov_type='clustered',
#             cluster_entity=True,
#             cluster_time=True
#         )

#         print('\nRESUMEN DEL MODELO')
#         print(resultado_panel.summary)

#         # --------------------------------------------------------
#         # 2.4. CALCULAR VALORES PREDICHOS
#         # --------------------------------------------------------
#         # Modelo:
#         # log(efectividad) = beta_bucket * log(tasa)
#         #                    + efecto fijo bucket
#         #                    + efecto fijo mes
#         #
#         # Los valores ajustados completos incorporan:
#         # componente explicado por X + efectos fijos.
#         # --------------------------------------------------------

#         valores_explicados = resultado_panel.fitted_values

#         efectos_fijos = resultado_panel.estimated_effects

#         # Alinear ambos componentes por índice
#         log_predicho = (
#             valores_explicados.iloc[:, 0] +
#             efectos_fijos.iloc[:, 0]
#         )

#         # Base de predicciones
#         df_pred = df_panel.reset_index()

#         # Alinear predicciones con las observaciones del modelo
#         df_pred = df_pred.set_index(['bucket', 'mes']).sort_index()

#         df_pred['log_efectividad_predicha'] = log_predicho

#         # Escala original
#         df_pred['efectividad_predicha'] = np.exp(
#             df_pred['log_efectividad_predicha']
#         )

#         # Residuales en escala logarítmica
#         df_pred['residuo_log'] = (
#             df_pred['log_efectividad'] -
#             df_pred['log_efectividad_predicha']
#         )

#         # Residuales en escala original
#         df_pred['residuo_efectividad'] = (
#             df_pred['efectividad'] -
#             df_pred['efectividad_predicha']
#         )

#         # Identificar ventana rolling
#         df_pred = df_pred.reset_index()
#         df_pred['periodo_cierre'] = periodo

#         columnas_prediccion = [
#             'bucket',
#             'mes',
#             'periodo',
#             'periodo_cierre',
#             'efectividad',
#             'efectividad_predicha',
#             'tasa_ofertada',
#             'log_efectividad',
#             'log_tasa',
#             'log_efectividad_predicha',
#             'residuo_log',
#             'residuo_efectividad'
#         ]

#         resultados_predicciones_rolling = pd.concat(
#             [
#                 resultados_predicciones_rolling,
#                 df_pred[columnas_prediccion]
#             ],
#             ignore_index=True
#         )

#         # --------------------------------------------------------
#         # 2.5. EXTRAER ELASTICIDADES E INTERVALOS DE CONFIANZA
#         # --------------------------------------------------------

#         intervalos = resultado_panel.conf_int()

#         resultados_elasticidad = []

#         for bucket, nombre_columna in nombres_interaccion.items():

#             # Puede que alguna variable se haya absorbido
#             # durante la estimación
#             estimada = nombre_columna in resultado_panel.params.index

#             if estimada:
#                 ic_inferior = intervalos.loc[
#                     nombre_columna, 'lower'
#                 ]

#                 ic_superior = intervalos.loc[
#                     nombre_columna, 'upper'
#                 ]

#                 elasticidad = resultado_panel.params[nombre_columna]
#                 error_std = resultado_panel.std_errors[nombre_columna]
#                 p_value = resultado_panel.pvalues[nombre_columna]

#             else:
#                 elasticidad = np.nan
#                 error_std = np.nan
#                 p_value = np.nan
#                 ic_inferior = np.nan
#                 ic_superior = np.nan

#             resultados_elasticidad.append({
#                 'cluster': bucket,
#                 'elasticidad': elasticidad,
#                 'error_std': error_std,
#                 'p_value': p_value,
#                 'ic_inferior': ic_inferior,
#                 'ic_superior': ic_superior
#             })

#         tabla_elasticidades = pd.DataFrame(
#             resultados_elasticidad
#         )

#         tabla_elasticidades['significativo_10pct'] = (
#             tabla_elasticidades['p_value'] < 0.10
#         )

#         tabla_elasticidades = (
#             tabla_elasticidades
#             .sort_values('cluster')
#             .reset_index(drop=True)
#         )

#         tabla_elasticidades['periodo'] = periodo

#         # --------------------------------------------------------
#         # 2.6. MÉTRICAS GLOBALES
#         # --------------------------------------------------------

#         tabla_elasticidades['r2_overall'] = (
#             resultado_panel.rsquared
#         )

#         tabla_elasticidades['r2_within'] = (
#             resultado_panel.rsquared_within
#         )

#         tabla_elasticidades['r2_between'] = (
#             resultado_panel.rsquared_between
#         )

#         tabla_elasticidades['n_obs'] = (
#             resultado_panel.nobs
#         )

#         tabla_elasticidades['f_stat'] = (
#             resultado_panel.f_statistic.stat
#         )

#         tabla_elasticidades['f_pval'] = (
#             resultado_panel.f_statistic.pval
#         )

#         resultados_globales_rolling = pd.concat(
#             [
#                 resultados_globales_rolling,
#                 tabla_elasticidades
#             ],
#             ignore_index=True
#         )

#         # --------------------------------------------------------
#         # 2.7. RESUMEN DE RESULTADOS
#         # --------------------------------------------------------

#         print('\nELASTICIDADES ESTIMADAS')

#         print(
#             tabla_elasticidades[
#                 [
#                     'cluster',
#                     'elasticidad',
#                     'error_std',
#                     'p_value',
#                     'ic_inferior',
#                     'ic_superior',
#                     'significativo_10pct'
#                 ]
#             ].to_string(index=False)
#         )

#         print('\nMÉTRICAS GLOBALES')

#         print(f'R² overall: {resultado_panel.rsquared:.4f}')
#         print(f'R² within:  {resultado_panel.rsquared_within:.4f}')
#         print(f'R² between: {resultado_panel.rsquared_between:.4f}')
#         print(f'N observaciones: {resultado_panel.nobs:,}')

#         print('\nEJEMPLO DE VALORES PREDICHOS')

#         print(
#             df_pred[
#                 [
#                     'bucket',
#                     'mes',
#                     'efectividad',
#                     'efectividad_predicha',
#                     'residuo_efectividad'
#                 ]
#             ].head(10).to_string(index=False)
#         )

#     except Exception as e:

#         print(f'\nERROR EN LA VENTANA {periodo}')
#         print(f'Detalle: {e}')
#         print('Se continúa con la siguiente ventana.')
#         continue


# # ============================================================
# # 3. RESULTADOS FINALES
# # ============================================================

# print('\n' + '=' * 80)
# print('PROCESO ROLLING FINALIZADO')
# print('=' * 80)

# print(
#     f'Filas de elasticidades: '
#     f'{len(resultados_globales_rolling):,}'
# )

# print(
#     f'Filas de predicciones: '
#     f'{len(resultados_predicciones_rolling):,}'
# )

# # Ordenar resultados
# if not resultados_globales_rolling.empty:
#     resultados_globales_rolling = (
#         resultados_globales_rolling
#         .sort_values(['periodo', 'cluster'])
#         .reset_index(drop=True)
#     )

# if not resultados_predicciones_rolling.empty:
#     resultados_predicciones_rolling = (
#         resultados_predicciones_rolling
#         .sort_values(['periodo_cierre', 'bucket', 'mes'])
#         .reset_index(drop=True)
#     )



# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # ============================================================
# # 1. PREPARAR DATOS
# # ============================================================

# df_plot = resultados_predicciones_rolling.copy()

# df_plot['mes'] = pd.to_datetime(df_plot['mes'])

# # Promediar por mes y bucket si existen varias observaciones
# # dentro de un mismo mes y bucket
# df_plot = (
#     df_plot
#     .groupby(
#         ['periodo_cierre', 'bucket', 'mes'],
#         as_index=False
#     )
#     .agg(
#         efectividad_obs=('efectividad', 'mean'),
#         efectividad_predicha=('efectividad_predicha', 'mean')
#     )
# )

# # ============================================================
# # 2. GRAFICAR POR BUCKET
# # ============================================================

# for bucket in sorted(df_plot['bucket'].dropna().unique()):

#     datos_bucket = df_plot[
#         df_plot['bucket'] == bucket
#     ].copy()

#     fig, ax = plt.subplots(figsize=(14, 6))

#     # Valores observados: una sola serie
#     observado = (
#         datos_bucket
#         .sort_values('mes')
#         .drop_duplicates(subset=['mes'])
#     )

#     ax.plot(
#         observado['mes'],
#         observado['efectividad_obs'],
#         marker='o',
#         linewidth=2.5,
#         label='Observada'
#     )

#     # Valores predichos: una serie por ventana rolling
#     for periodo_cierre in sorted(
#         datos_bucket['periodo_cierre'].unique()
#     ):

#         datos_ventana = (
#             datos_bucket[
#                 datos_bucket['periodo_cierre'] == periodo_cierre
#             ]
#             .sort_values('mes')
#         )

#         ax.plot(
#             datos_ventana['mes'],
#             datos_ventana['efectividad_predicha'],
#             marker='.',
#             linewidth=1.5,
#             alpha=0.8,
#             label=f'Predicha hasta {periodo_cierre}'
#         )

#     ax.set_title(
#         f'Bucket {bucket}: efectividad observada vs. predicha',
#         fontsize=14
#     )
#     ax.set_xlabel('Mes')
#     ax.set_ylabel('Efectividad')
#     ax.xaxis.set_major_locator(mdates.MonthLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))

#     ax.grid(True, linestyle='--', alpha=0.3)
#     ax.legend()
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# import numpy as np
# import pandas as pd

# # ============================================================
# # 1. PREPARAR BASE
# # ============================================================

# df_metricas = resultados_predicciones_rolling.copy()

# df_metricas = df_metricas.dropna(
#     subset=['efectividad', 'efectividad_predicha']
# ).copy()

# y = df_metricas['efectividad']
# y_pred = df_metricas['efectividad_predicha']

# df_metricas['error'] = y_pred - y
# df_metricas['error_abs'] = np.abs(df_metricas['error'])
# df_metricas['error_cuadrado'] = df_metricas['error'] ** 2

# # MAPE: se calcula únicamente donde el observado es distinto de cero
# df_metricas['error_pct'] = np.where(
#     y != 0,
#     np.abs((y - y_pred) / y),
#     np.nan
# )

# # ============================================================
# # 2. FUNCIÓN DE MÉTRICAS
# # ============================================================

# def calcular_metricas(g):

#     observado = g['efectividad'].to_numpy()
#     predicho = g['efectividad_predicha'].to_numpy()

#     error = predicho - observado

#     mae = np.mean(np.abs(error))
#     rmse = np.sqrt(np.mean(error ** 2))

#     suma_obs_abs = np.sum(np.abs(observado))

#     wape = (
#         np.sum(np.abs(error)) / suma_obs_abs * 100
#         if suma_obs_abs > 0
#         else np.nan
#     )

#     mape = (
#         np.mean(np.abs(error[observado != 0] /
#                        observado[observado != 0])) * 100
#         if np.any(observado != 0)
#         else np.nan
#     )

#     bias = np.mean(error)

#     ss_res = np.sum(error ** 2)
#     ss_tot = np.sum((observado - np.mean(observado)) ** 2)

#     r2 = (
#         1 - ss_res / ss_tot
#         if ss_tot > 0
#         else np.nan
#     )

#     return pd.Series({
#         'n_obs': len(g),
#         'MAE': mae,
#         'RMSE': rmse,
#         'MAPE_pct': mape,
#         'WAPE_pct': wape,
#         'Bias': bias,
#         'R2': r2
#     })

# # ============================================================
# # 3. MÉTRICAS POR VENTANA Y BUCKET
# # ============================================================

# metricas_bucket = (
#     df_metricas
#     .groupby(['periodo_cierre', 'bucket'])
#     .apply(calcular_metricas, include_groups=False)
#     .reset_index()
# )

# # ============================================================
# # 4. MÉTRICAS GLOBALES POR VENTANA
# # ============================================================

# metricas_globales = (
#     df_metricas
#     .groupby('periodo_cierre')
#     .apply(calcular_metricas, include_groups=False)
#     .reset_index()
# )

# # ============================================================
# # 5. VISUALIZAR RESULTADOS
# # ============================================================

# print('MÉTRICAS POR BUCKET')
# display(
#     metricas_bucket.sort_values(
#         ['bucket', 'periodo_cierre']
#     ).round(4)
# )

# print('MÉTRICAS GLOBALES')
# display(
#     metricas_globales.sort_values(
#         'periodo_cierre'
#     ).round(4)
# )

# metricas_bucket.to_clipboard(index=False)