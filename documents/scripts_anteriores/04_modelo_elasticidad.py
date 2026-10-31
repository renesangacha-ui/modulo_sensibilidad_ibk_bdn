
# datos_sensibilidad_interna

# data_panel_sensibilidad[['cluster_modelar','periodo','efectividad','tasa_ofertada']]



# import numpy as np
# import pandas as pd
# import statsmodels.formula.api as smf

# from linearmodels.panel import PanelOLS

# df = data_panel_sensibilidad[
#     [
#         'cluster_modelar',
#         'periodo',
#         'efectividad',
#         'tasa_ofertada'
#     ]
# ].copy()

# df = df.rename(columns={
#     'cluster_modelar': 'bucket'
# })

# # Periodo YYYYMM numérico -> fecha mensual
# df['mes'] = pd.to_datetime(
#     df['periodo'].astype('Int64').astype(str),
#     format='%Y%m',
#     errors='coerce'
# )

# # Convertir variables a numéricas
# for col in ['efectividad', 'tasa_ofertada']:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Eliminar registros incompletos
# df = df.dropna(
#     subset=['bucket', 'mes', 'efectividad', 'tasa_ofertada']
# ).copy()

# # Los logaritmos requieren valores estrictamente positivos
# df = df.loc[
#     (df['efectividad'] > 0) &
#     (df['tasa_ofertada'] > 0)
# ].copy()

# # Transformación logarítmica
# df['log_efectividad'] = np.log(df['efectividad'])
# df['log_tasa'] = np.log(df['tasa_ofertada'])

# # Categorías como texto
# df['bucket'] = df['bucket'].astype(str)
# # df['mes'] = df['mes'].astype(str)

# print(f'Observaciones utilizadas: {len(df):,}')
# print(f'Clusters: {df["bucket"].nunique()}')
# print(f'Meses: {df["mes"].nunique()}')

# # ============================================================
# # 2. CREAR INTERACCIONES (Seguro para nombres con espacios)
# # ============================================================

# buckets = sorted(df['bucket'].unique())
# nombres_interaccion = {}

# for i, bucket in enumerate(buckets):
#     # Identificador numérico limpio para evitar errores por espacios
#     nombre_columna = f'tasa_inter_{i}'

#     df[nombre_columna] = (
#         df['log_tasa'] *
#         (df['bucket'] == bucket).astype(int)
#     )

#     nombres_interaccion[bucket] = nombre_columna

# # ============================================================
# # 3. MÉTODO 2: PanelOLS CON INTERACCIONES
# # ============================================================

# # Establecer índice MultiIndex (entidad, tiempo) requerido por linearmodels
# df_panel = df.set_index(["bucket", "mes"]).sort_index()

# columnas_x = list(nombres_interaccion.values())

# modelo_panel = PanelOLS(
#     dependent=df_panel["log_efectividad"],
#     exog=df_panel[columnas_x],
#     entity_effects=True,  # Efectos fijos por bucket
#     time_effects=True,    # Efectos fijos por mes
#     drop_absorbed=True,
# )

# # Ajuste del modelo con errores estándar agrupados por entidad
# resultado_panel = modelo_panel.fit(
#     cov_type="clustered", 
#     cluster_entity=True
# )

# print('\n' + '=' * 70)
# print('MODELO PANELOLS CON EFECTOS FIJOS E INTERACCIONES')
# print('=' * 70)
# print(resultado_panel.summary)

# # ============================================================
# # 4. EXTRAER ELASTICIDADES Y ESTADÍSTICOS POR CLUSTER
# # ============================================================

# intervalos = resultado_panel.conf_int()
# resultados_elasticidad = []

# for bucket, nombre_columna in nombres_interaccion.items():
#     resultados_elasticidad.append({
#         'cluster': bucket,  # Mantiene el nombre real con espacios
#         'elasticidad': resultado_panel.params.get(nombre_columna, np.nan),
#         'error_std': resultado_panel.std_errors.get(nombre_columna, np.nan),
#         'p_value': resultado_panel.pvalues.get(nombre_columna, np.nan),
#         'ic_inferior': intervalos.loc[nombre_columna, 'lower'],
#         'ic_superior': intervalos.loc[nombre_columna, 'upper']
#     })

# tabla_elasticidades = pd.DataFrame(resultados_elasticidad)

# tabla_elasticidades['significativo_10pct'] = (
#     tabla_elasticidades['p_value'] < 0.10
# )

# tabla_elasticidades = tabla_elasticidades.sort_values(
#     'cluster'
# ).reset_index(drop=True)

# print('\n' + '=' * 70)
# print('ELASTICIDADES POR CLUSTER (PanelOLS)')
# print('=' * 70)
# print(tabla_elasticidades.to_string(index=False))

# # Diccionario directo con los betas y el nombre real del bucket
# betas_panel = {
#     bucket: resultado_panel.params[nombre_columna]
#     for bucket, nombre_columna in nombres_interaccion.items()
# }

# # ============================================================
# # 5. RESUMEN DE SIGNIFICANCIA
# # ============================================================

# print('\n' + '=' * 70)
# print('RESUMEN DE SIGNIFICANCIA')
# print('=' * 70)

# for _, fila in tabla_elasticidades.iterrows():
#     print(
#         f'{fila["cluster"]}: '
#         f'elasticidad={fila["elasticidad"]:.4f}, '
#         f'p-value={fila["p_value"]:.4g}, '
#         f'significativo al 10%={fila["significativo_10pct"]}'
#     )