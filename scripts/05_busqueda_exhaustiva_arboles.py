############################# Busqueda Exhaustiva: Arboles ##############################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tBusqueda Exhaustiva: Arboles')
print('-'*100)

#____________________ 1) Paquetes ----

import numpy as np
import pandas as pd

import os
import sys
import string
import warnings
from pathlib import Path
import itertools
from itertools import combinations, product

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

import xgboost as xgb
import shap
from tqdm.auto import tqdm
from joblib import Parallel, delayed

sys.path.append("..")
import src.modelamiento.funcionesmodelamiento as mdl
warnings.filterwarnings('ignore')

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

#____________________ 4) Parametros ----

# A partir de la exploración de los datos, se definieron los siguientes parámetros:

PERIODO_INICIO=202504
PERIODO_FIN=202604
SEMILLA_ALEATORIZACION=5646
PORC_PARTICION_DEV=0.3

#____________________ 5) Funnel ----

datos=datos.query(f'periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}')

funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

datos = datos.query("flg_exito_4 in @funnel_oferta_relevante").copy()

#____________________ 6) Procesamiento ----

#~~~~ 6.1) Reescalamiento ----

datos['tasa_desembolsada'] = datos['tasa_2022']/100

#~~~~ 6.2) Recodificaciones ----

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

datos['nombre_ifi']=datos['cod_instit_financiera'].map({1.0:'BCP', 6.0: 'BBVA', 4.0:'SCOTIA', 123.0:'MI BANCO'}).fillna('OTROS')

#____________________ 7) Segmentación: Árbol de decisión ----

#~~~~ 7.1) Construcción: Base Propension ----

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

#~~~~ 7.2) Procesamiento ----

#---- 7.2.1) Tratamiento: Monto ofertado ----

datos_oferta_relevante['monto_ofertado_ajustado'] = np.where(
    datos_oferta_relevante['monto_desembolsado'].isna(),
    datos_oferta_relevante['monto_min'],
    np.where(
        datos_oferta_relevante['monto_desembolsado'] > datos_oferta_relevante['monto_min'],
        datos_oferta_relevante['monto_desembolsado'],
        datos_oferta_relevante['monto_min']
    )
)

datos_oferta_relevante.periodo.value_counts().sort_index()

#~~~~ 7.3) Separación de Bases: Dev/OOT ----

oot=datos_oferta_relevante.query(f'periodo>=202602')
dev=datos_oferta_relevante.query(f'periodo<202602')

train_df, test_df = train_test_split(
    dev, 
    test_size=PORC_PARTICION_DEV, 
    stratify=dev['indicador_exito'], 
    random_state=SEMILLA_ALEATORIZACION
)

train_df = train_df.copy()
test_df = test_df.copy()

train_df['base'] = 'train'
test_df['base'] = 'test'
oot['base']='oot'

base_propension = pd.concat([train_df, test_df,oot], ignore_index=True)
del train_df, test_df,oot

#~~~~ 7.4) Selección de variables óptimas ----

#---- 7.4.1) Identificación de Variables ----

var_dependiente='indicador_exito'

baneadas = []

variables_numericas_aptas=['monto_min',
    'nivel_ventas',
    'pasivosprom_u3m',
    'pasivosprom_u6m',
    'pasivosprom_u12m',
    'flag_izipay_u03m',
    'flag_izipay_u06m',
    'flag_izipay_u12m',
    'suma_uai_u12m',
    'deuda_directa_vig_total_sf_u03m',
    'deuda_directa_vig_total_sf_u06m',
    'deuda_directa_vig_total_sf_u12m',
    ]

#---- 7.4.2) Selección datos ----

X_ini_tr=base_propension.query('base=="train"')[variables_numericas_aptas]
y_ini_tr=base_propension.query('base=="train"')[var_dependiente]

X_ini_te=base_propension.query('base=="test"')[variables_numericas_aptas]
y_ini_te=base_propension.query('base=="test"')[var_dependiente]

#---- 7.4.3) Framework de seleccion de variables por ML ----

sel_num_ini, sel_dummy_ini = mdl.seleccionar_variables_pipeline(
    X_ini_tr, y_ini_tr,
    X_ini_te, y_ini_te,
    features_num=variables_numericas_aptas,
    features_dummy=[],
    target_type='classification',
    n_top=30,
    umbral_corr=0.70,
    vars_baneadas=baneadas
)

variables_arbol_ini = sel_num_ini + sel_dummy_ini

print("Variables a evaluar (con relevancia por ML) en el AD:")
print("\n".join([f" - {v}" for v in variables_arbol_ini]))

#~~~~ 7.5) Busqueda Exhaustiva: Arbol de decisión ----

#---- 7.5.1) funcion: Modelo AD Unitario ----
def _evaluar_modelo_ad_unitario(
    id_modelo, subset_vars, max_leaves, porc_leafs, usar_peso,
    originaciones, X_ini_tr, y_ini_tr, 
    col_peso, umbral_separacion, umbral_psi,
    var_dependiente, period_col, base_col
):
    try:
        X_arbol_tr = X_ini_tr[list(subset_vars)].values
        y_arbol_tr = y_ini_tr.values
        min_samples_leaf = int(porc_leafs * X_arbol_tr.shape[0])
        
        if min_samples_leaf * max_leaves > len(X_arbol_tr):
            return None
            
        pesos = X_ini_tr[col_peso].values if (usar_peso and col_peso in X_ini_tr.columns) else None
        
        arbol = DecisionTreeClassifier(
            max_leaf_nodes=max_leaves,
            criterion='gini',
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        if pesos is not None:
            arbol.fit(X_arbol_tr, y_arbol_tr, sample_weight=pesos)
        else:
            arbol.fit(X_arbol_tr, y_arbol_tr)
            
        importances = arbol.feature_importances_
        n_variables_efectivas = int(np.sum(importances > 0))
        variables_efectivas_nombres = [var for var, imp in zip(subset_vars, importances) if imp > 0]
            
        leaf_ids_tr = arbol.apply(X_arbol_tr)
        temp_tr = X_ini_tr.copy()
        temp_tr['segmento_arbol'] = leaf_ids_tr
        temp_tr[var_dependiente] = y_arbol_tr
        
        tabla_bucket = (
            temp_tr.groupby('segmento_arbol')[var_dependiente]
            .mean()
            .rename('ratio_exito')
            .reset_index()
            .sort_values('ratio_exito')
        )
        
        n_segmentos_reales = len(tabla_bucket)
        
        letras = list(string.ascii_uppercase)
        recodificacion = {
            seg: f"{i+1}. Segmento {letras[i]}"
            for i, seg in enumerate(tabla_bucket["segmento_arbol"])
        }
        
        leaf_ids_total = arbol.apply(originaciones[list(subset_vars)])
        df_feat = originaciones.copy()
        df_feat['segmento_arbol'] = leaf_ids_total
        df_feat['segmento_arbol_new'] = df_feat['segmento_arbol'].map(recodificacion)
        
        if df_feat['segmento_arbol_new'].isnull().any():
            return None
            
        ratio_exito = (
            df_feat.pivot_table(index='segmento_arbol_new', columns=base_col, values=var_dependiente, aggfunc='mean')
            .sort_index()
        )
        
        dict_cumplimiento = mdl.obtener_diccionario_cumplimiento_separacion(ratio_exito, umbral=umbral_separacion)
        _, estadisticos_globales = mdl.calcular_reporte_cruces_por_cosecha(
            df=df_feat, period_col=period_col, segment_col='segmento_arbol_new', target_col=var_dependiente, ascending=True
        )
        _, stats_bases = mdl.calcular_psi_temporal_por_base(df_feat, period_col='periodo', segment_col='segmento_arbol_new', base_col='base', base_ref_nombre='train')
        resultado_psi = mdl.evaluar_estabilidad_por_metrica(stats_bases, umbral=umbral_psi)
        
        fila_datos = {
            'id_modelo': id_modelo,
            'variables_usadas': ", ".join(subset_vars),
            'n_variables_candidatas': len(subset_vars),
            'n_variables_efectivas': n_variables_efectivas,
            'variables_efectivas_nombres': ", ".join(variables_efectivas_nombres),
            'max_leaf_nodes_config': max_leaves,
            'n_segmentos_reales': n_segmentos_reales, 
            'porc_min_leafs': porc_leafs,
            'ponderado_monto': 'Sí' if usar_peso else 'No',
        }
        
        for base, val in dict_cumplimiento.items():
            fila_datos[f'cumplimiento_sep_{base}'] = val
        for stat, val in estadisticos_globales.items():
            fila_datos[f'cruces_{stat}'] = val
        for metrica, val in resultado_psi.items():
            fila_datos[f'psi_estable_{metrica}'] = val
            
        fila_datos['_objeto_arbol'] = arbol
        fila_datos['_recodificacion'] = recodificacion
        fila_datos['_subset_vars'] = list(subset_vars)
        fila_datos['_usar_peso'] = usar_peso
        
        return fila_datos
    except Exception as e:
        return {
            'id_modelo': id_modelo,
            'variables_usadas': ", ".join(subset_vars) if subset_vars else "",
            'n_variables_candidatas': len(subset_vars) if subset_vars else 0,
            'n_variables_efectivas': 0,
            'variables_efectivas_nombres': "",
            'max_leaf_nodes_config': max_leaves,
            'n_segmentos_reales': 0,
            'porc_min_leafs': porc_leafs,
            'ponderado_monto': 'Sí' if usar_peso else 'No',
            'error': str(e)
        }

#---- 7.5.2) funcion:  Grid Search ----
def ejecutar_grid_search_con_segmentos_reales(
    originaciones, 
    X_ini_tr, 
    y_ini_tr, 
    variables_disponibles, 
    min_vars=1, 
    max_vars=6,
    lista_max_leafs=[2, 3, 4],
    lista_porc_leafs=[0.10, 0.15, 0.20],
    lista_ponderacion=[True, False],
    col_peso='monto_desembolsado',
    umbral_separacion=0.03,
    umbral_psi=0.10,
    var_dependiente='indicador_exito',
    period_col='periodo_x',
    base_col='base'
):
    combinaciones_vars = []
    max_v = min(max_vars, len(variables_disponibles))
    min_v = max(min_vars, 1)
    
    for r in range(min_v, max_v + 1):
        for subset in itertools.combinations(variables_disponibles, r):
            combinaciones_vars.append(subset)
            
    tareas = [
        (subset_vars, max_leaves, porc_leafs, usar_peso)
        for subset_vars in combinaciones_vars
        for max_leaves in lista_max_leafs
        for porc_leafs in lista_porc_leafs
        for usar_peso in lista_ponderacion
    ]
    
    total_iteraciones = len(tareas)
    
    print("=" * 80)
    print(f"🚀 INICIANDO GRID SEARCH PARALELIZADO (Con auditoría de variables efectivas y control de errores)")
    print(f"• Total combinaciones de variables: {len(combinaciones_vars)}")
    print(f"• Total iteraciones totales: {total_iteraciones}")
    print("=" * 80 + "\n")
    
    resultados_paralelos = Parallel(n_jobs=-1, verbose=10)(
        delayed(_evaluar_modelo_ad_unitario)(
            id_modelo=i+1,
            subset_vars=tarea[0],
            max_leaves=tarea[1],
            porc_leafs=tarea[2],
            usar_peso=tarea[3],
            originaciones=originaciones,
            X_ini_tr=X_ini_tr,
            y_ini_tr=y_ini_tr,
            col_peso=col_peso,
            umbral_separacion=umbral_separacion,
            umbral_psi=umbral_psi,
            var_dependiente=var_dependiente,
            period_col=period_col,
            base_col=base_col
        )
        for i, tarea in enumerate(tareas)
    )
    
    # 🎯 Incluimos todos los resultados devueltos (no None)
    registros_ejecutivos = [res for res in resultados_paralelos if res is not None]
    df_grid_maestro = pd.DataFrame(registros_ejecutivos)
    
    # Resumen de ejecución
    modelos_exitosos = df_grid_maestro['error'].isna().sum() if 'error' in df_grid_maestro.columns else len(df_grid_maestro)
    modelos_con_error = len(df_grid_maestro) - modelos_exitosos
    
    print("\n" + "=" * 80)
    print(f"✅ ¡Grid Search completado!")
    print(f"• Total modelos procesados: {len(df_grid_maestro)}/{total_iteraciones}")
    print(f"• Modelos evaluados con éxito: {modelos_exitosos}")
    print(f"• Modelos con errores registrados: {modelos_con_error}")
    print("=" * 80 + "\n")
    
    return df_grid_maestro

#---- 7.5.3) Ejecución: Grid Search ----
df_grid_maestro = ejecutar_grid_search_con_segmentos_reales(
    originaciones=base_propension,
    X_ini_tr=X_ini_tr,
    y_ini_tr=y_ini_tr,
    variables_disponibles=variables_arbol_ini,
    min_vars=2,
    max_vars=5,
    lista_max_leafs=[3,4,5],
    lista_porc_leafs=[0.10, 0.15],
    lista_ponderacion=[True,False],
    umbral_separacion=0.03,
    col_peso='monto_ofertado_ajustado',
    period_col='periodo',
    base_col='base',
    var_dependiente='indicador_exito'
)

#---- 7.5.4) Modelos Aprrobados ----
df_grid_maestro['estado_final'] = np.where(
    (df_grid_maestro['psi_estable_Mediana'] == 'Sí') & 
    (df_grid_maestro['cruces_suma'] <= 3) & 
    (df_grid_maestro['cumplimiento_sep_train'] >= 0.90),
    'SÍ SALE (Aprobado)', 
    'NO SALE (Rechazado)'
)

#---- 7.5.5) Analisis ----
modelos_aprobados = df_grid_maestro[df_grid_maestro['estado_final'] == 'SÍ SALE (Aprobado)']

if not modelos_aprobados.empty:
    # Seleccionar el mejor modelo según tus criterios
    mejor_modelo = modelos_aprobados.sort_values(by=['n_segmentos_reales', 'cruces_suma'], ascending=[False, True]).iloc[0]
    
    print("=" * 65)
    print("🏆 ¡MODELO GANADOR IDENTIFICADO Y APTO PARA SALIR! 🚀")
    print("=" * 65)
    print(f"• ID del Modelo: {mejor_modelo['id_modelo']}")
    print(f"• Estado: {mejor_modelo['estado_final']}")
    print(f"• Variables Candidatas: {mejor_modelo['variables_usadas']}")
    print(f"• Variables Efectivas (Usadas): {mejor_modelo['variables_efectivas_nombres']}")
    print(f"• Max Leaf Nodes: {mejor_modelo['max_leaf_nodes_config']}")
    print(f"• Suma de Cruces: {mejor_modelo['cruces_suma']}")
    print(f"• PSI Máximo Estable: {mejor_modelo['psi_estable_Máximo']}")
    print("=" * 65)

    # ==========================================
    # PASO 3.1: VISUALIZAR LOS CORTES DEL ÁRBOL
    # ==========================================
    arbol_ganador = mejor_modelo['_objeto_arbol']
    
    # ⚠️ Usar siempre las variables candidatas originales con las que se entrenó el árbol
    vars_modelo = mejor_modelo['_subset_vars']  
    
    # 1. Ver reglas en formato de texto por consola
    from sklearn.tree import export_text
    print("\n🌳 REGLAS Y CORTES DEL ÁRBOL GANADOR:")
    print(export_text(arbol_ganador, feature_names=list(vars_modelo)))
    
    # 2. Graficar el árbol visualmente
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    
    plt.figure(figsize=(12, 6), dpi=150)
    plot_tree(
        arbol_ganador, 
        feature_names=list(vars_modelo), 
        filled=True, 
        rounded=True,
        precision=4
    )
    plt.title(f"Estructura del Árbol - Modelo ID: {mejor_modelo['id_modelo']}", fontsize=12)
    plt.show()

    # ==========================================
    # PASO 4: REPLICACIÓN AUTOMÁTICA Y PREDICCIÓN DE TRAIN
    # ==========================================
    recod_ganadora = mejor_modelo['_recodificacion']
    
    # Aplicar el árbol usando el subset exacto de variables con el que fue entrenado
    leaf_ids_ganador = arbol_ganador.apply(base_propension[list(vars_modelo)])
    base_propension_ganador = base_propension.copy()
    base_propension_ganador['segmento_arbol'] = leaf_ids_ganador
    base_propension_ganador['segmento_arbol_new'] = base_propension_ganador['segmento_arbol'].map(recod_ganadora)
    
    # 🎯 ADJUNTAR LA PREDICCIÓN/RATIO DE TRAIN A CADA BUCKET EN EL MODELO GENERAL
    train_con_segmentos = X_ini_tr.copy()
    train_con_segmentos['segmento_arbol'] = arbol_ganador.apply(X_ini_tr[list(vars_modelo)])
    train_con_segmentos['segmento_arbol_new'] = train_con_segmentos['segmento_arbol'].map(recod_ganadora)
    train_con_segmentos['indicador_exito'] = y_ini_tr.values
    
    dict_ratio_train = train_con_segmentos.groupby('segmento_arbol_new')['indicador_exito'].mean().to_dict()
    base_propension_ganador['prediccion'] = base_propension_ganador['segmento_arbol_new'].map(dict_ratio_train)
    
    print("📌 ¡Predicción/Ratio de train pegada exitosamente a cada bucket en 'originaciones_ganador' (Columna: 'ratio_train_esperado')!")

    print("\n📊 Generando reportes y gráficos finales con el modelo ganador...\n")
    
    # Ejecutar visualizaciones finales de forma automática
    mdl.plot_escalera_tasas_global(base_propension_ganador, segment_col='segmento_arbol_new', target_col='indicador_exito')
    mdl.plot_escalera_tasas_global(base_propension_ganador.query('base=="train"'), segment_col='segmento_arbol_new', target_col='indicador_exito')
    mdl.plot_barras_comparacion_base(base_propension_ganador, segment_col='segmento_arbol_new', target_col='indicador_exito', base_col='base')
    mdl.plot_todas_series_juntas(base_propension_ganador,  period_col='periodo', segment_col='segmento_arbol_new', target_col='indicador_exito')
    mdl.plot_evolucion_segmentos_grid(base_propension_ganador, period_col='periodo', segment_col='segmento_arbol_new', target_col='indicador_exito', base_col='base', ncols=3)
    mdl.plot_materialidad_global_temporal(base_propension_ganador, period_col='periodo', segment_col='segmento_arbol_new')
    mdl.plot_materialidad_y_precancelacion(
        base_propension_ganador,
        period_col='periodo',
        segment_col='segmento_arbol_new',
        target_col='indicador_exito',
        oot_inicio='202602'
    )
    reporte_mensual, estadisticos_globales = mdl.calcular_reporte_cruces_por_cosecha(
    df=base_propension_ganador,
    period_col='periodo',
    segment_col='segmento_arbol_new',
    target_col='indicador_exito',
    ascending=True
    )

    print("Reporte Mensual:", reporte_mensual)
    print("Resumen Estadístico:", estadisticos_globales)

    df_psi_temporal, stats_bases = mdl.calcular_psi_temporal_por_base(base_propension_ganador, period_col='periodo', segment_col='segmento_arbol_new', base_col='base', base_ref_nombre='train')

    print("--- Estadísticos Resumen del PSI Temporal por Base ---")
    print(stats_bases)

    print("\n--- Muestra de la Serie Temporal por Base ---")
    print(df_psi_temporal)

    mdl.plot_precancelacion_barras_agrupadas(
    base_propension_ganador,
    period_col='periodo',
    segment_col='segmento_arbol_new',
    target_col='indicador_exito',
    oot_inicio='202602',
    mostrar_promedios=False
    )
else:
    print("❌ Ningún modelo cumplió con los criterios estrictos de aprobación. Se sugiere revisar la grilla o relajar los umbrales.")


base_propension_ganador

# #____________________ 7) Busqueda Exhaustiva ----

# #~~~~ 7.1) Variables Candidatas ----

# listado_variables=vars_categorizadas.columns

# #~~~~ 7.2) Función unitaria de modelamiento ----

# def evaluar_familia_variables(
#     combinacion_vars, 
#     datos_base, 
#     p_value_umbral=0.05, 
#     umbral_materialidad=0.03,
#     periodo_inicio=202504,
#     periodo_fin=202604,
#     umbral_winsorizacion=0.025,
#     meses_tol_min_materialidad=9
# ):
#     try:
#         df_iter = datos_base.copy()
#         df_iter['cluster_comercial'] = df_iter[list(combinacion_vars)].astype(str).agg(' & '.join, axis=1)

#         # Procesamiento: mercado
#         datos_tasa_mercado = df_iter.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna() ")
        
#         limites_winsor = datos_tasa_mercado.groupby('cluster_comercial')['tasa_ultima'].agg(
#             limite_inferior=lambda x: x.quantile(umbral_winsorizacion),
#             limite_superior=lambda x: x.quantile(1 - umbral_winsorizacion)
#         ).reset_index()
        
#         datos_tasa_mercado = datos_tasa_mercado.merge(limites_winsor, on='cluster_comercial', how='left')
#         datos_tasa_mercado['tasa_mercado_winsorizada'] = datos_tasa_mercado['tasa_ultima'].clip(
#             lower=datos_tasa_mercado['limite_inferior'], 
#             upper=datos_tasa_mercado['limite_superior']
#         )
        
#         prom_pond = lambda v, w: (
#             np.average(v[v.notna() & w.notna() & (w > 0)], weights=w[v.notna() & w.notna() & (w > 0)])
#             if (v.notna() & w.notna() & (w > 0)).any()
#             else np.nan
#         )
        
#         data_panel_mercado_by_entidad = (
#             datos_tasa_mercado
#             .assign(tasa_x_monto=lambda df: df['tasa_mercado_winsorizada'] * df['delta_deuda_final'])
#             .groupby(['cluster_comercial', 'nombre_ifi', 'periodo'], observed=True)
#             .agg(
#                 desembolsos=('llave_interna', 'size'),
#                 monto_desembolsado=('delta_deuda_final', 'sum'),
#                 suma_tasa_x_monto=('tasa_x_monto', 'sum'),
#                 tasa_mercado=('tasa_mercado_winsorizada', lambda x: prom_pond(x, datos_tasa_mercado.loc[x.index, 'delta_deuda_final']))
#             )
#             .reset_index()
#         )
        
#         data_panel_sensibilidad_mercado = (
#             data_panel_mercado_by_entidad
#             .groupby(['cluster_comercial', 'periodo'], observed=True)
#             .agg(
#                 total_desembolsos_mercado=('desembolsos', 'sum'),
#                 monto_desembolsado_mercado=('monto_desembolsado', 'sum'),
#                 suma_tasa_x_monto_bucket=('suma_tasa_x_monto', 'sum')
#             )
#             .assign(tasa_mercado=lambda df: df['suma_tasa_x_monto_bucket'] / df['monto_desembolsado_mercado'])
#             .reset_index()
#         )
        
#         # Procesamiento: sensibilidad interna
#         datos_oferta_relevante = (
#             df_iter
#             .sort_values(by=['llave_interna', 'flg_exito_4'], ascending=[True, True])
#             .drop_duplicates(subset=['llave_interna'], keep='first')
#             .assign(indicador_exito=lambda df: (df['flg_exito_4'] == 'exito_desembolso_ibk_ct').astype(int))
#         )
        
#         datos_oferta_relevante['monto_ofertado_ajustado'] = np.where(
#             datos_oferta_relevante['monto_desembolsado'].isna(),
#             datos_oferta_relevante['monto_min'],
#             np.where(
#                 datos_oferta_relevante['monto_desembolsado'] > datos_oferta_relevante['monto_min'],
#                 datos_oferta_relevante['monto_desembolsado'],
#                 datos_oferta_relevante['monto_min']
#             )
#         )
        
#         columnas_si = ['periodo', 'llave_interna', 'cluster_comercial', 'indicador_exito', 'flg_exito_4', 'monto_min', 'monto_ofertado_ajustado', 'tasa_min', 'monto_desembolsado', 'tasa_desembolsada']
#         df_oi = datos_oferta_relevante[columnas_si].copy()
        
#         def promedio_ponderado(valores, pesos):
#             mask = valores.notna() & pesos.notna() & (pesos > 0)
#             if mask.any():
#                 return np.average(valores[mask], weights=pesos[mask])
#             return np.nan

#         data_panel_sensibilidad_interna = (
#             df_oi
#             .groupby(["cluster_comercial", "periodo"], observed=True)
#             .agg(
#                 leads=("llave_interna", "size"),
#                 monto_desembolsado_ibk=("monto_desembolsado", "sum"),
#                 monto_ofertado=("monto_ofertado_ajustado", "sum"),
#                 desembolsos=("indicador_exito", "sum"),
#                 tasa_desembolsada=("tasa_desembolsada", lambda x: promedio_ponderado(x, df_oi.loc[x.index, "monto_desembolsado"])),
#                 tasa_ofertada=("tasa_min", lambda x: promedio_ponderado(x, df_oi.loc[x.index, "monto_ofertado_ajustado"])),
#             )
#             .assign(
#                 efectividad=lambda df: np.where(df["monto_ofertado"] > 0, df["monto_desembolsado_ibk"] / df["monto_ofertado"], 0),
#                 conversion=lambda df: np.where(df["leads"] > 0, df["desembolsos"] / df["leads"], 0),
#             )
#             .reset_index()
#         )
        
#         # Validación de Materialidad
#         aux_mat = (
#             data_panel_sensibilidad_interna.groupby(['periodo', 'cluster_comercial'], observed=True)
#             .agg(
#                 leads=('leads', 'sum'),
#                 monto_ofertado=('monto_ofertado', 'sum'),
#                 desembolsos=('desembolsos', 'sum'),
#                 monto_desembolsado_ibk=('monto_desembolsado_ibk', 'sum'),
#             )
#             .assign(
#                 participacion_monto_desembolsado=lambda df: df.groupby(level='periodo')['monto_desembolsado_ibk'].transform(lambda x: x / x.sum()),
#                 participacion_desembolsos=lambda df: df.groupby(level='periodo')['desembolsos'].transform(lambda x: x / x.sum()),
#             )
#             .reset_index()
#         )
        
#         aux_mat['flg_mat_monto'] = np.where(aux_mat['participacion_monto_desembolsado'] > umbral_materialidad, 1, 0)
#         aux_mat['flg_mat_n'] = np.where(aux_mat['participacion_desembolsos'] > umbral_materialidad, 1, 0)
        
#         buckets_val = (
#             aux_mat
#             .query(f'desembolsos != 0 and periodo >= {periodo_inicio} and periodo <= {periodo_fin}')
#             .groupby('cluster_comercial')
#             .agg(conteo=('cluster_comercial', 'size'),
#                  mat_n=('flg_mat_n', 'sum'),
#                  mat_monto=('flg_mat_monto', 'sum'))
#             .reset_index()
#         )
        
#         n_buckets_iniciales = len(buckets_val)
#         buckets_val_filtrados = buckets_val.query(f'conteo == 13 and mat_monto >={meses_tol_min_materialidad} and mat_n >= {meses_tol_min_materialidad}')
#         n_buckets_validos = len(buckets_val_filtrados)
#         listado_val = buckets_val_filtrados.cluster_comercial.to_list()
        
#         if n_buckets_validos == 0:
#             return None
            
#         # Panel de sensibilidad completo
#         dp_sens = pd.merge(data_panel_sensibilidad_mercado, data_panel_sensibilidad_interna, how='outer', on=['cluster_comercial', 'periodo'])
#         dp_sens = dp_sens.query(f"cluster_comercial in {listado_val}")
#         dp_sens['spread'] = dp_sens['tasa_ofertada'] - dp_sens['tasa_mercado']
        
#         dp_sens["log_efectividad"] = np.log(dp_sens["efectividad"].replace(0, np.nan))
#         dp_sens["log_tasa"] = np.log(dp_sens["tasa_ofertada"])
        
#         # Demeaning Simultáneo (Two-Way)
#         def demeaning_sim(data, col):
#             g_mean = data[col].mean()
#             b_mean = data.groupby("cluster_comercial")[col].transform("mean")
#             t_mean = data.groupby("periodo")[col].transform("mean")
#             return data[col] - b_mean - t_mean + g_mean

#         dp_sim = dp_sens.copy()
#         dp_sim["log_efect_dem"] = demeaning_sim(dp_sim, "log_efectividad")
#         dp_sim["log_tasa_dem"] = demeaning_sim(dp_sim, "log_tasa")
#         dp_sim["spread_dem"] = demeaning_sim(dp_sim, "spread")
        
#         # Modelamiento OLS por bucket
#         resultados_sim = []
#         for grupo in dp_sim["cluster_comercial"].unique():
#             dg = dp_sim[dp_sim["cluster_comercial"] == grupo].dropna(subset=["log_efect_dem", "log_tasa_dem", "spread_dem"])
#             if len(dg) < 5: continue
            
#             m_conj = ols("log_efect_dem ~ log_tasa_dem + spread_dem - 1", data=dg).fit()
#             m_tasa = ols("log_efect_dem ~ log_tasa_dem - 1", data=dg).fit()
#             m_spread = ols("log_efect_dem ~ spread_dem - 1", data=dg).fit()
            
#             resultados_sim.append({
#                 "grupo": grupo,
#                 "conj_elast_tasa": m_conj.params["log_tasa_dem"],
#                 "conj_pval_tasa": m_conj.pvalues["log_tasa_dem"],
#                 "conj_sens_spread": m_conj.params["spread_dem"],
#                 "conj_pval_spread": m_conj.pvalues["spread_dem"],
#                 "tasa_elast": m_tasa.params["log_tasa_dem"],
#                 "tasa_pval": m_tasa.pvalues["log_tasa_dem"],
#                 "spread_sens": m_spread.params["spread_dem"],
#                 "spread_pval": m_spread.pvalues["spread_dem"],
#             })
            
#         df_res_sim = pd.DataFrame(resultados_sim)
#         df_res_sim['familia'] = " & ".join(combinacion_vars)
        
#         # Métricas de resumen
#         sig_conj_tasa = (df_res_sim['conj_pval_tasa'] < p_value_umbral).sum()
#         sig_conj_spread = (df_res_sim['conj_pval_spread'] < p_value_umbral).sum()
        
#         sig_conj_ambos = ((df_res_sim['conj_pval_tasa'] < p_value_umbral) & 
#                           (df_res_sim['conj_pval_spread'] < p_value_umbral)).sum()
        
#         sig_tasa_m2 = (df_res_sim['tasa_pval'] < p_value_umbral).sum()
#         sig_spread_m3 = (df_res_sim['spread_pval'] < p_value_umbral).sum()
        
#         resumen_info = {
#             "familia": " & ".join(combinacion_vars),
#             "n_buckets_iniciales": n_buckets_iniciales,
#             "n_buckets_validos": n_buckets_validos,
#             "buckets_inferencia_conjunto": df_res_sim['grupo'].to_list(),
#             "cant_sig_modelo1_tasa": int(sig_conj_tasa),
#             "cant_sig_modelo1_spread": int(sig_conj_spread),
#             "cant_sig_modelo1_ambos": int(sig_conj_ambos),
#             "cant_sig_modelo2_tasa": int(sig_tasa_m2),
#             "cant_sig_modelo3_spread": int(sig_spread_m3)
#         }
        
#         return df_res_sim, resumen_info

#     except Exception as e:
#         print(f"Error evaluando familia {combinacion_vars}: {e}")
#         return None, None

# #~~~~ 7.3) Busqueda ----

# listado_variables = list(vars_categorizadas.columns)

# combinaciones_totales = []
# for r in [1,2,3]:
#     combinaciones_totales.extend(list(itertools.combinations(listado_variables, r)))

# print(f"Evaluando {len(combinaciones_totales)} familias de variables en paralelo...")

# resultados_paralelos = Parallel(n_jobs=-1, verbose=10)(
#     delayed(evaluar_familia_variables)(
#         comb, 
#         datos, 
#         p_value_umbral=0.10, 
#         umbral_materialidad=0.03,
#         periodo_inicio=202504,
#         periodo_fin=202604,
#         umbral_winsorizacion=0.025,
#         meses_tol_min_materialidad=9
#     ) 
#     for comb in combinaciones_totales
# )

# valid_results = [res for res in resultados_paralelos if res is not None and res[0] is not None]

# df_elasticidades_todas = pd.concat(
#     [res[0] for res in valid_results], 
#     ignore_index=True
# )

# lista_diccionarios_resumen = [
#     res[1] for res in valid_results if res[1] is not None
# ]

# df_resumen_familias = pd.DataFrame(lista_diccionarios_resumen)

# if not df_resumen_familias.empty:
#     df_resumen_familias = df_resumen_familias.sort_values(
#         by=['cant_sig_modelo1_ambos', 'n_buckets_validos'], 
#         ascending=False
#     ).reset_index(drop=True)

# print("\n--- Top 5 Familias de Variables ---")
# print(df_resumen_familias.head())
