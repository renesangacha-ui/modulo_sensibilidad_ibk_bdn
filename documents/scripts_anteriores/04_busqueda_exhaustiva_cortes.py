############################# Busqueda Exhaustiva: Cortes ###############################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tBusqueda Exhaustiva: Cortes')
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

import itertools
from joblib import Parallel, delayed

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

datos=pd.read_csv(dir_b+r'/raw/V01/data_sensibilidad.csv',sep=',')

#____________________ 4) Parametros ----

# En base al proceso de exploración de la información se han decidido los siguientes parametros
# De materialidad, disponibilidad de informacion, asi como de comportamiento observado

PERIODO_INICIO=202504
PERIODO_FIN=202604

#____________________ 5) Funnel ----

datos=datos.query(f'periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}')

funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

datos = datos.query("flg_exito_4 in @funnel_oferta_relevante").copy()

#____________________ 6) Procesamiento ----

#~~~~ 6.1) Recategorización ----

datos['nombre_ifi']=datos['cod_instit_financiera'].map({1.0:'BCP', 6.0: 'BBVA', 4.0:'SCOTIA', 123.0:'MI BANCO'}).fillna('OTROS')

#~~~~ 6.2) Reescalamiento ----

datos['tasa_desembolsada'] = datos['tasa_2022']/100

#~~~~ 6.3) Categorización de Variables ----

def categorizar_en_buckets_con_cortes(serie, n_buckets=3, label_na="Sin info"):
    """
    Divide la data válida en N buckets por percentiles.
    - Incluyente por la izquierda: [a, b) y cerrado en el extremo final [a, b].
    - Inteligente con binarias.
    - Maneja correctamente el límite superior para que no bote 'Sin info' por error.
    """
    serie_trabajo = serie.copy()
    serie_limpia = serie_trabajo.dropna()
    
    if len(serie_limpia) == 0:
        return pd.Series([label_na] * len(serie), index=serie.index, name=f"{serie.name}_cat")
    
    # DETECCIÓN DE BINARIAS
    valores_unicos = serie_limpia.unique()
    if len(valores_unicos) <= 2:
        resultado_binario = serie_trabajo.astype(object)
        resultado_binario = resultado_binario.fillna(label_na)
        return pd.Series(resultado_binario, name=f"{serie.name}_cat")
    
    porcentajes = np.linspace(0, 100, n_buckets + 1)
    cortes = np.percentile(serie_limpia, porcentajes)
    cortes = sorted(list(set(cortes)))
    
    if len(cortes) < 2:
        return pd.Series([label_na] * len(serie), index=serie.index, name=f"{serie.name}_cat")

    cortes[-1] = cortes[-1] + 1e-9

    resultado_limpio = pd.cut(
        serie_limpia, 
        bins=cortes, 
        right=False, 
        include_lowest=True
    )
    
    intervalos_unicos = resultado_limpio.cat.categories
    mapa_etiquetas = {
        intervalo: f"[{intervalo.left:.2f} - {intervalo.right:.2f})" 
        for intervalo in intervalos_unicos
    }
    resultado_limpio = resultado_limpio.map(mapa_etiquetas)

    resultado_final = pd.Series(index=serie.index, dtype='object')
    resultado_final.loc[serie_limpia.index] = resultado_limpio.astype(str)
    
    resultado_final = resultado_final.fillna(label_na)
    
    return pd.Series(resultado_final, name=f"{serie.name}_cat")

vars_monto_ofertado=['monto_min']
vars_nivel_ventas=['nivel_ventas']
vars_pasivos=['pasivosprom_u3m','pasivosprom_u6m','pasivosprom_u12m']
vars_salpromedios_creditos=['salprom_creditos_u03m','salprom_creditos_u06m','salprom_creditos_u12m']
vars_izipay=['flag_izipay_u03m','flag_izipay_u06m','flag_izipay_u12m']
vars_uai=['suma_uai_u12m']
vars_deuda_directa_sf=['deuda_directa_vig_total_sf_u03m','deuda_directa_vig_total_sf_u06m','deuda_directa_vig_total_sf_u12m']
vars_sow=['sow_deuda_u12m']

variables_candidatas=vars_monto_ofertado+vars_nivel_ventas+vars_pasivos+vars_salpromedios_creditos+vars_izipay+vars_uai+vars_deuda_directa_sf+vars_sow

vars_categorizadas=pd.concat(
    {f"{var}_cat": categorizar_en_buckets_con_cortes(datos[var], n_buckets=4, label_na='Sin info')
     for var in variables_candidatas}, 
    axis=1
)

datos=pd.concat([datos, vars_categorizadas],axis=1)

#____________________ 7) Busqueda Exhaustiva ----

#~~~~ 7.1) Variables Candidatas ----

listado_variables=vars_categorizadas.columns

#~~~~ 7.2) Función unitaria de modelamiento ----

def evaluar_familia_variables(
    combinacion_vars, 
    datos_base, 
    p_value_umbral=0.05, 
    umbral_materialidad=0.03,
    periodo_inicio=202504,
    periodo_fin=202604,
    umbral_winsorizacion=0.025,
    meses_tol_min_materialidad=9
):
    try:
        df_iter = datos_base.copy()
        df_iter['cluster_comercial'] = df_iter[list(combinacion_vars)].astype(str).agg(' & '.join, axis=1)

        # Procesamiento: mercado
        datos_tasa_mercado = df_iter.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna() ")
        
        limites_winsor = datos_tasa_mercado.groupby('cluster_comercial')['tasa_ultima'].agg(
            limite_inferior=lambda x: x.quantile(umbral_winsorizacion),
            limite_superior=lambda x: x.quantile(1 - umbral_winsorizacion)
        ).reset_index()
        
        datos_tasa_mercado = datos_tasa_mercado.merge(limites_winsor, on='cluster_comercial', how='left')
        datos_tasa_mercado['tasa_mercado_winsorizada'] = datos_tasa_mercado['tasa_ultima'].clip(
            lower=datos_tasa_mercado['limite_inferior'], 
            upper=datos_tasa_mercado['limite_superior']
        )
        
        prom_pond = lambda v, w: (
            np.average(v[v.notna() & w.notna() & (w > 0)], weights=w[v.notna() & w.notna() & (w > 0)])
            if (v.notna() & w.notna() & (w > 0)).any()
            else np.nan
        )
        
        data_panel_mercado_by_entidad = (
            datos_tasa_mercado
            .assign(tasa_x_monto=lambda df: df['tasa_mercado_winsorizada'] * df['delta_deuda_final'])
            .groupby(['cluster_comercial', 'nombre_ifi', 'periodo'], observed=True)
            .agg(
                desembolsos=('llave_interna', 'size'),
                monto_desembolsado=('delta_deuda_final', 'sum'),
                suma_tasa_x_monto=('tasa_x_monto', 'sum'),
                tasa_mercado=('tasa_mercado_winsorizada', lambda x: prom_pond(x, datos_tasa_mercado.loc[x.index, 'delta_deuda_final']))
            )
            .reset_index()
        )
        
        data_panel_sensibilidad_mercado = (
            data_panel_mercado_by_entidad
            .groupby(['cluster_comercial', 'periodo'], observed=True)
            .agg(
                total_desembolsos_mercado=('desembolsos', 'sum'),
                monto_desembolsado_mercado=('monto_desembolsado', 'sum'),
                suma_tasa_x_monto_bucket=('suma_tasa_x_monto', 'sum')
            )
            .assign(tasa_mercado=lambda df: df['suma_tasa_x_monto_bucket'] / df['monto_desembolsado_mercado'])
            .reset_index()
        )
        
        # Procesamiento: sensibilidad interna
        datos_oferta_relevante = (
            df_iter
            .sort_values(by=['llave_interna', 'flg_exito_4'], ascending=[True, True])
            .drop_duplicates(subset=['llave_interna'], keep='first')
            .assign(indicador_exito=lambda df: (df['flg_exito_4'] == 'exito_desembolso_ibk_ct').astype(int))
        )
        
        datos_oferta_relevante['monto_ofertado_ajustado'] = np.where(
            datos_oferta_relevante['monto_desembolsado'].isna(),
            datos_oferta_relevante['monto_min'],
            np.where(
                datos_oferta_relevante['monto_desembolsado'] > datos_oferta_relevante['monto_min'],
                datos_oferta_relevante['monto_desembolsado'],
                datos_oferta_relevante['monto_min']
            )
        )
        
        columnas_si = ['periodo', 'llave_interna', 'cluster_comercial', 'indicador_exito', 'flg_exito_4', 'monto_min', 'monto_ofertado_ajustado', 'tasa_min', 'monto_desembolsado', 'tasa_desembolsada']
        df_oi = datos_oferta_relevante[columnas_si].copy()
        
        def promedio_ponderado(valores, pesos):
            mask = valores.notna() & pesos.notna() & (pesos > 0)
            if mask.any():
                return np.average(valores[mask], weights=pesos[mask])
            return np.nan

        data_panel_sensibilidad_interna = (
            df_oi
            .groupby(["cluster_comercial", "periodo"], observed=True)
            .agg(
                leads=("llave_interna", "size"),
                monto_desembolsado_ibk=("monto_desembolsado", "sum"),
                monto_ofertado=("monto_ofertado_ajustado", "sum"),
                desembolsos=("indicador_exito", "sum"),
                tasa_desembolsada=("tasa_desembolsada", lambda x: promedio_ponderado(x, df_oi.loc[x.index, "monto_desembolsado"])),
                tasa_ofertada=("tasa_min", lambda x: promedio_ponderado(x, df_oi.loc[x.index, "monto_ofertado_ajustado"])),
            )
            .assign(
                efectividad=lambda df: np.where(df["monto_ofertado"] > 0, df["monto_desembolsado_ibk"] / df["monto_ofertado"], 0),
                conversion=lambda df: np.where(df["leads"] > 0, df["desembolsos"] / df["leads"], 0),
            )
            .reset_index()
        )
        
        # Validación de Materialidad
        aux_mat = (
            data_panel_sensibilidad_interna.groupby(['periodo', 'cluster_comercial'], observed=True)
            .agg(
                leads=('leads', 'sum'),
                monto_ofertado=('monto_ofertado', 'sum'),
                desembolsos=('desembolsos', 'sum'),
                monto_desembolsado_ibk=('monto_desembolsado_ibk', 'sum'),
            )
            .assign(
                participacion_monto_desembolsado=lambda df: df.groupby(level='periodo')['monto_desembolsado_ibk'].transform(lambda x: x / x.sum()),
                participacion_desembolsos=lambda df: df.groupby(level='periodo')['desembolsos'].transform(lambda x: x / x.sum()),
            )
            .reset_index()
        )
        
        aux_mat['flg_mat_monto'] = np.where(aux_mat['participacion_monto_desembolsado'] > umbral_materialidad, 1, 0)
        aux_mat['flg_mat_n'] = np.where(aux_mat['participacion_desembolsos'] > umbral_materialidad, 1, 0)
        
        buckets_val = (
            aux_mat
            .query(f'desembolsos != 0 and periodo >= {periodo_inicio} and periodo <= {periodo_fin}')
            .groupby('cluster_comercial')
            .agg(conteo=('cluster_comercial', 'size'),
                 mat_n=('flg_mat_n', 'sum'),
                 mat_monto=('flg_mat_monto', 'sum'))
            .reset_index()
        )
        
        n_buckets_iniciales = len(buckets_val)
        buckets_val_filtrados = buckets_val.query(f'conteo == 13 and mat_monto >={meses_tol_min_materialidad} and mat_n >= {meses_tol_min_materialidad}')
        n_buckets_validos = len(buckets_val_filtrados)
        listado_val = buckets_val_filtrados.cluster_comercial.to_list()
        
        if n_buckets_validos == 0:
            return None
            
        # Panel de sensibilidad completo
        dp_sens = pd.merge(data_panel_sensibilidad_mercado, data_panel_sensibilidad_interna, how='outer', on=['cluster_comercial', 'periodo'])
        dp_sens = dp_sens.query(f"cluster_comercial in {listado_val}")
        dp_sens['spread'] = dp_sens['tasa_ofertada'] - dp_sens['tasa_mercado']
        
        dp_sens["log_efectividad"] = np.log(dp_sens["efectividad"].replace(0, np.nan))
        dp_sens["log_tasa"] = np.log(dp_sens["tasa_ofertada"])
        
        # Demeaning Simultáneo (Two-Way)
        def demeaning_sim(data, col):
            g_mean = data[col].mean()
            b_mean = data.groupby("cluster_comercial")[col].transform("mean")
            t_mean = data.groupby("periodo")[col].transform("mean")
            return data[col] - b_mean - t_mean + g_mean

        dp_sim = dp_sens.copy()
        dp_sim["log_efect_dem"] = demeaning_sim(dp_sim, "log_efectividad")
        dp_sim["log_tasa_dem"] = demeaning_sim(dp_sim, "log_tasa")
        dp_sim["spread_dem"] = demeaning_sim(dp_sim, "spread")
        
        # Modelamiento OLS por bucket
        resultados_sim = []
        for grupo in dp_sim["cluster_comercial"].unique():
            dg = dp_sim[dp_sim["cluster_comercial"] == grupo].dropna(subset=["log_efect_dem", "log_tasa_dem", "spread_dem"])
            if len(dg) < 5: continue
            
            m_conj = ols("log_efect_dem ~ log_tasa_dem + spread_dem - 1", data=dg).fit()
            m_tasa = ols("log_efect_dem ~ log_tasa_dem - 1", data=dg).fit()
            m_spread = ols("log_efect_dem ~ spread_dem - 1", data=dg).fit()
            
            resultados_sim.append({
                "grupo": grupo,
                "conj_elast_tasa": m_conj.params["log_tasa_dem"],
                "conj_pval_tasa": m_conj.pvalues["log_tasa_dem"],
                "conj_sens_spread": m_conj.params["spread_dem"],
                "conj_pval_spread": m_conj.pvalues["spread_dem"],
                "tasa_elast": m_tasa.params["log_tasa_dem"],
                "tasa_pval": m_tasa.pvalues["log_tasa_dem"],
                "spread_sens": m_spread.params["spread_dem"],
                "spread_pval": m_spread.pvalues["spread_dem"],
            })
            
        df_res_sim = pd.DataFrame(resultados_sim)
        df_res_sim['familia'] = " & ".join(combinacion_vars)
        
        # Métricas de resumen
        sig_conj_tasa = (df_res_sim['conj_pval_tasa'] < p_value_umbral).sum()
        sig_conj_spread = (df_res_sim['conj_pval_spread'] < p_value_umbral).sum()
        
        sig_conj_ambos = ((df_res_sim['conj_pval_tasa'] < p_value_umbral) & 
                          (df_res_sim['conj_pval_spread'] < p_value_umbral)).sum()
        
        sig_tasa_m2 = (df_res_sim['tasa_pval'] < p_value_umbral).sum()
        sig_spread_m3 = (df_res_sim['spread_pval'] < p_value_umbral).sum()
        
        resumen_info = {
            "familia": " & ".join(combinacion_vars),
            "n_buckets_iniciales": n_buckets_iniciales,
            "n_buckets_validos": n_buckets_validos,
            "buckets_inferencia_conjunto": df_res_sim['grupo'].to_list(),
            "cant_sig_modelo1_tasa": int(sig_conj_tasa),
            "cant_sig_modelo1_spread": int(sig_conj_spread),
            "cant_sig_modelo1_ambos": int(sig_conj_ambos),
            "cant_sig_modelo2_tasa": int(sig_tasa_m2),
            "cant_sig_modelo3_spread": int(sig_spread_m3)
        }
        
        return df_res_sim, resumen_info

    except Exception as e:
        print(f"Error evaluando familia {combinacion_vars}: {e}")
        return None, None

#~~~~ 7.3) Busqueda ----

listado_variables = list(vars_categorizadas.columns)

combinaciones_totales = []
for r in [1,2,3]:
    combinaciones_totales.extend(list(itertools.combinations(listado_variables, r)))

print(f"Evaluando {len(combinaciones_totales)} familias de variables en paralelo...")

resultados_paralelos = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluar_familia_variables)(
        comb, 
        datos, 
        p_value_umbral=0.10, 
        umbral_materialidad=0.03,
        periodo_inicio=202504,
        periodo_fin=202604,
        umbral_winsorizacion=0.025,
        meses_tol_min_materialidad=9
    ) 
    for comb in combinaciones_totales
)

valid_results = [res for res in resultados_paralelos if res is not None and res[0] is not None]

df_elasticidades_todas = pd.concat(
    [res[0] for res in valid_results], 
    ignore_index=True
)

lista_diccionarios_resumen = [
    res[1] for res in valid_results if res[1] is not None
]

df_resumen_familias = pd.DataFrame(lista_diccionarios_resumen)

if not df_resumen_familias.empty:
    df_resumen_familias = df_resumen_familias.sort_values(
        by=['cant_sig_modelo1_ambos', 'n_buckets_validos'], 
        ascending=False
    ).reset_index(drop=True)

print("\n--- Top 5 Familias de Variables ---")
print(df_resumen_familias.head())
