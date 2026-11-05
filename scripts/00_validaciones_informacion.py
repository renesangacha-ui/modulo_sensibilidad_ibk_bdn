############################### Validaciones: Informacion ############################

#####################################################################################

print('-' * 100)
print('\t\t\t\t\tValidaciones: Informacion')
print('-'*100)

#____________________ 1) Paquetes ----

import pandas as pd
import numpy as np

import os
from pathlib import Path

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

#~~~~ 5.2) Consistencia ----

# Prueba: Los desembolsos en la competencia no  deben tener información de desembolso en IBK
datos.query('tasa_2022.isna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.isna()').flg_exito_4.value_counts(dropna=False)

datos.query('tasa_2022.notna()').flg_exito_4.value_counts(dropna=False)
datos.query('monto_desembolsado.notna()').flg_exito_4.value_counts(dropna=False)
datos.query('tasa_2022.notna() or monto_desembolsado.notna()').flg_exito_4.value_counts(dropna=False)

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

datos_casos_no_validados.flg_exito_4.value_counts(dropna=False)
datos_casos_no_validados.cluster.value_counts(dropna=False)
datos_casos_no_validados.cod_instit_financiera.value_counts(dropna=False)
datos_casos_no_validados.periodo.value_counts(dropna=False)

# Se identifican fracasos con información de desembolsos en IBK; en la mayoría de los casos,
# esto se explica porque el mismo cliente también registra un desembolso en IBK.
# Se aplicarán los ajustes necesarios para evitar atribuir esta información al fracaso.

#~~~~ 5.3) Idoneidad ----

datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").tasa_2022.describe()
datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").monto_desembolsado.describe()

datos.monto_min.isna().mean()
datos.monto_min.describe()
datos.query('monto_min==0').T

datos.tasa_min.isna().mean() 
datos.tasa_min.describe()
datos.query('tasa_min==0').T

# Monto ofertado vs Desembolsado
datos.query("monto_desembolsado.notna() and monto_desembolsado > monto_min")

(
    datos.query("monto_desembolsado.notna() and monto_desembolsado > monto_min")
    .assign(diferencia=lambda x: x["monto_desembolsado"] - x["monto_min"])
    ["diferencia"]
    .describe(percentiles=[i / 100 for i in range(1, 100)])
)

# El 80% tiene diferencias menores a 20 soles, el 20% tiene diferencia superiores a 1897 soles

#~~~~ 5.4) Duplicidad ----

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
        monto_desembolsado=('monto_desembolsado','nunique'))

)

validacion_duplicidad

validacion_duplicidad.query('tasa_min>1 or monto_min>1 or cluster>1 or tasa_2022>1 or monto_desembolsado>1 ')

validacion_duplicidad.query('cod_instit_financiera>1 or flg_exito_4>1')

validacion_duplicidad.query('flg_exito_4==1 and cod_instit_financiera==1')

# Unicidad a nivel cliente: Ningún cliente tiene más de una tasa o monto
# ofertado/desembolsado, ni pertenece a más de un clúster o tipo de cliente.
#
# La duplicidad de registros se debe a que el lead realizó más de un desembolso
# en el SSFF en un mes
