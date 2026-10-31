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


#____________________ 4) Funnel: Oferta Relevante ----


datos.flg_exito_4.value_counts()

# Nos quedamos con los exitos y los fracasos de desembolso no IBK con monto similar con gestión,
# que son los que nos interesan para la construcción de la base de sensibilidad.

# funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

# datos = datos.query("flg_exito_4 in @funnel_oferta_relevante").copy() 



#____________________ 6) Validaciones ----


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
    .groupby(['cluster_comercial','cluster_comercial_aperturado'])
    .agg(
        completitud=('tasa_ultima', lambda x: x.notna().mean()),
        registros=('llave_interna', 'size'),
    )
    .reset_index()
)

aux_completitud_inferencia_fracasos_by_cluster['bloque_cluster']=aux_completitud_inferencia_fracasos_by_cluster['cluster_comercial'].isin(['Emprendedor Izipay','Nuevo en Expansión','Próximo Recurrente']).map({True:'Sin Deuda', False:'Con Deuda'})

graficos.plot_completitud_por_bloques_y_materialidad(
    df=aux_completitud_inferencia_fracasos_by_cluster,
    columna_cluster='cluster_comercial_aperturado',
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
    .groupby(['periodo','cluster_comercial','cluster_comercial_aperturado'])
    .agg(
        completitud=('tasa_ultima', lambda x: x.notna().mean()),
        registros=('llave_interna', 'size'),
    )
    .reset_index()
)

graficos.plot_evolutivo_multiserie_panel(
    df=aux_completitud_inferencia_fracasos_by_periodo_cluster,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial_aperturado',
    metrica='completitud',
    label_metrica='Completitud (%)',
    titulo=(
            'Evolución: Completitud de la Tasa de Mercado (Sin Filtro por'
            ' Entidad Financiera)\nPor Clúster Comercial'
        )
)

for cluster in aux_completitud_inferencia_fracasos_by_periodo_cluster.cluster_comercial_aperturado.unique():
    
    graficos.plot_completitud_por_periodo_y_materialidad(
    df=aux_completitud_inferencia_fracasos_by_periodo_cluster.query(f'cluster_comercial_aperturado=="{cluster}"'),
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
    .groupby(['cluster_comercial_aperturado', 'cod_instit_financiera'])
    .size()
    .reset_index(name='registros')
    .sort_values(
        by=['cluster_comercial_aperturado', 'registros'], ascending=[True, False]
    )
    .assign(
        proporcion=lambda df: df.groupby('cluster_comercial_aperturado')[
            'registros'
        ].transform(lambda x: x / x.sum()),
        cumsum_proporcion=lambda df: df.groupby('cluster_comercial_aperturado')[
            'proporcion'
        ].cumsum(),
    )
    .reset_index(drop=True)
)

for cluster in aux_pareto_by_cluster.cluster_comercial_aperturado.unique():
    
    graficos.plot_pareto_instituciones(
        aux_pareto_by_cluster.query(f"cluster_comercial_aperturado=='{cluster}'"),
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

datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()")
datos_tasa_mercado=datos.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna()")

datos_tasa_mercado['tasa_ultima'].describe(percentiles=[0.01,0.05,0.10, 0.25, 0.50, 0.75, 0.90,0.95,0.99])

#____________________ 8) Procesamiento ----

#~~~~ 8.1) Winsorizacion: Inferencia de Tasa ----

UMBRAL_WINSORIZACION=0.025

limites_winsor = datos_tasa_mercado.groupby('cluster_comercial_aperturado')['tasa_ultima'].agg(
    limite_inferior=lambda x: x.quantile(UMBRAL_WINSORIZACION),
    limite_superior=lambda x: x.quantile(1 - UMBRAL_WINSORIZACION)
).reset_index()

datos_tasa_mercado

datos_tasa_mercado = datos_tasa_mercado.merge(limites_winsor, on='cluster_comercial_aperturado', how='left')

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
    .groupby(['cluster','cluster_comercial','FLG_CLIENTE','tipo_cliente','cluster_comercial_aperturado', 'nombre_ifi', 'periodo'], observed=True)
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
    .groupby(['cluster_comercial_aperturado', 'periodo'], observed=True)
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
    columna_grupo='cluster_comercial_aperturado',
    metrica='tasa_mercado',
    label_metrica='Tasa Mercado (%)',
    titulo=(
            'Evolución: Tasa de Mercado (4 Principales Bancos)\nPor Clúster Comercial'
        )
)

# Tasa de cada institución y su aporte, por segmento
for cluster in data_panel_mercado_by_entidad.cluster_comercial_aperturado.unique():

    graficos.plot_evolutivo_multiserie_con_barras_apiladas(
        df=data_panel_mercado_by_entidad.query(f"cluster_comercial_aperturado=='{cluster}'"),
        columna_periodo='periodo',
        columna_grupo='nombre_ifi',
        label_grupo='Institución Financiera',
        metrica='tasa_mercado',
        label_metrica='Tasa de Mercado (%)',
        columna_materialidad='monto_desembolsado',
        modo_materialidad='porcentaje',
        titulo=(
                f'Evolución: Tasa de Mercado (Principales Bancos)\n{cluster}'
            )
    )

#____________________ 10) Almacenamiento ----

data_panel_mercado_by_entidad.to_csv(dir_b+r'/processed/V02/base_sensibilidad_componente_mercado.csv',sep='|', index=False)
