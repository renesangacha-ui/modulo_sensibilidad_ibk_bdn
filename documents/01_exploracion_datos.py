################### Exploración de los datos  ##################

################################################################

print('-' * 100)
print('\t\t\t\t\tExploración de los datos')
print('-'*100)

#____________________ Paquetes ----

import pandas as pd
import numpy as np

from scipy.optimize import minimize

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import time
from pathlib import Path
import re
import warnings

#____________________ Direcciones ----

# Dirección del Proyecto:
dir_p = str(Path.cwd().parent)

# Dirección de la carpeta con las bases de datos:
dir_b = os.path.join(dir_p, "data")

# Dirección de la carpeta de resultados:
dir_r = os.path.join(dir_p, "results")

# Dirección de la carpeta de scripts:
dir_s = os.path.join(dir_p, "scripts")

#____________________ Lectura de archivos ----

datos=pd.read_csv(dir_b+r'/raw/V01/data_sensibilidad.csv',sep=',')

datos.head().T
datos.info()
datos.isnull().mean().sort_values(ascending=False)
datos.describe()
datos.periodo.min(), datos.periodo.max()


mapeo_clusters = {
     'cluster_1_sow_no_nulo': 'Cliente Ancla',
     'cluster_2_sow_no_nulo': 'Socio en Crecimiento',
     'cluster_3_sow_no_nulo': 'Cuenta Estratégica',
     'cluster_4_sow_no_nulo': 'Gran Cuenta por Conquistar',
     'cluster_5_sow_no_nulo': 'Maduro Multibanco',
     'cluster_6_sow_no_nulo': 'Cliente por Reactivar',     
     'cluster_1_nulo': 'Emprededor Izipay',
     'cluster_2_nulo': 'Nuevo en Expansión',
     'cluster_3_nulo': 'Próximo Recurrente',
}


datos['cluster_comercial'] = datos['cluster'].map(mapeo_clusters).fillna(datos['cluster'])
datos[['cluster_comercial','cluster']].value_counts().reset_index()

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

# ============================================================
# 1. Oferta relevante
# ============================================================

# La relevancia se evalúa por banco, comparando el desembolso observado
# con el monto ofertado por IBK (<= 1.2x oferta), y no sobre la suma de
# desembolsos de todas las entidades.
#
# Esto permite capturar casos en los que el cliente distribuyó su necesidad
# de financiamiento entre varias entidades: aunque la suma de desembolsos
# supere 1.2x la oferta, un desembolso individual dentro del umbral evidencia
# que el monto ofertado por IBK era competitivo y el cliente pudo haber
# considerado IBK para cubrir parte de su necesidad y distribuir el resto
# entre otras entidades.
#
# Decisión: si al menos un banco cumple el criterio de relevancia, la oferta
# IBK se considera relevante y el cliente ingresa a la población de efectividad.
#
# Para efectividad, el cliente se contabiliza una sola vez, dado que recibió
# una única oferta IBK, independientemente de la cantidad de entidades en las
# que posteriormente se observen desembolsos relevantes.
#
# En resumen: oferta única por cliente -> relevancia evaluada por banco ->
# si algún banco es relevante, cliente entra a efectividad -> 1 lead por cliente.



# 2. Inferencia de tasa
#
# Antecedente:
# La inferencia de tasa se obtiene ex post; una de las metodologías utilizadas consiste
# en optimizar la curva de desgaste de saldos observada para una cosecha. Por ello,
# para un crédito actual no se cuenta con una inferencia propia, sino que se utiliza
# como referencia la última inferencia disponible de su bucket, asumiendo que el
# comportamiento observado en cosechas pasadas es representativo para los nuevos créditos.
#
# Alineamiento temporal:
# ¿Sería recomendable que el modelo de elasticidad replique, para cada cosecha histórica,
# la información que habría estado disponible en producción en ese momento? Por ejemplo,
# si se requieren 3 meses de maduración, a inicios de mayo-26 la última inferencia
# disponible sería la de enero-26. Si bien hoy conocemos la inferencia de mayo-26,
# esta se obtiene posteriormente, utilizando información que aún no estaba disponible
# en ese momento. En ese sentido, ¿utilizar la inferencia de la misma cosecha no podría
# estar generando un posible riesgo de data leakage? ¿No sería más recomendable utilizar
# la última inferencia disponible a la fecha de referencia de cada cosecha, tal como
# habría ocurrido en producción?
#
# Alineamiento con la señal de producción:
# Actualmente se cuenta con inferencias para múltiples bancos, pero en producción podría
# utilizarse solo un subconjunto o una tasa consolidada. ¿Qué bancos o inferencias se
# utilizarán finalmente como input? Para mantener la consistencia entre el modelamiento
# y su aplicación productiva, ¿convendría construir el modelo de elasticidad utilizando
# esa misma definición de señal?

#____________________ Duplicados ----

datos.query("flg_exito_4=='fracaso_desembolso_no_ibk_monto_similar_con_gestion'").value_counts(['llave_interna']).reset_index().value_counts('count', normalize=True)
datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").value_counts(['llave_interna']).reset_index().value_counts('count')
registros_duplicados=datos.value_counts(['llave_interna']).reset_index().query('count>1')

(
pd.merge(datos,
         registros_duplicados,
         how='inner',
         on='llave_interna')
.sort_values('llave_interna'))

(
    pd.merge(datos,
         registros_duplicados,
         how='inner',
         on='llave_interna')
    .sort_values('llave_interna')
    .query("flg_exito_4 in ['fracaso_desembolso_no_ibk_monto_similar_con_gestion','exito_desembolso_ibk_ct']")
    .groupby('llave_interna')
    .agg(conteo_tasa_ultima=('tasa_ultima','size'),
         )
)

datos.query("flg_exito_4=='exito_desembolso_ibk_ct'").query("tasa_ultima.notna()")
datos.query("flg_exito_4=='fracaso_desembolso_no_ibk_monto_similar_con_gestion'").query("tasa_ultima.isna()")
datos.query("flg_exito_4=='fracaso_desembolso_no_ibk_monto_similar_con_gestion'").query("tasa_ultima.notna()")
datos.query("flg_exito_4=='fracaso_desembolso_no_ibk_monto_similar_con_gestion' and saldo_desembolso.notna()").value_counts(['llave_interna']).reset_index()

datos.query('llave_interna=="A856AF5722BA74872630AA590D7EA00A5FB49CE0FD2D36F3A1392ABBF102F0A7-202603"')

(
     datos
         .query("flg_exito_4 in ['fracaso_desembolso_no_ibk_monto_similar_con_gestion', 'exito_desembolso_ibk_ct']")
)

(
     datos
     .query("flg_exito_4 in ['fracaso_desembolso_no_ibk_monto_similar_con_gestion', 'exito_desembolso_ibk_ct']")[['llave_interna', 'flg_exito_4']]    
     .drop_duplicates()
     .value_counts('llave_interna')
     .reset_index()
     .query('count > 1')
)

datos.query("llave_interna=='5FEE499B9A31EF7B6BFC825F6675A9F9450389BAE41FC2826A02FC8C7EFB1898-202511'")

(
    datos
    .query("flg_exito_4 in ['fracaso_desembolso_no_ibk_monto_similar_con_gestion', 'exito_desembolso_ibk_ct']")
    .assign(indicador_exito=lambda df: (df['flg_exito_4'] == 'exito_desembolso_ibk_ct').astype(int))
    .groupby('llave_interna', as_index=False)['indicador_exito']
    .max()
)

datos_oferta_relevante = (
    datos
    .query("flg_exito_4 in ['fracaso_desembolso_no_ibk_monto_similar_con_gestion', 'exito_desembolso_ibk_ct']")
    .assign(
        indicador_exito=lambda df: (df['flg_exito_4'] == 'exito_desembolso_ibk_ct').astype(int),
        prioridad=lambda df: (df['flg_exito_4'] != 'exito_desembolso_ibk_ct').astype(int)
    )
    .sort_values(by=['llave_interna', 'prioridad'])
    .drop_duplicates(subset=['llave_interna'], keep='first')
    .drop(columns=['prioridad'])
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _aplicar_estilo_ejecutivo(ax, titulo, ylabel, color_titulo='#1D4290'):
    ax.set_title(
        titulo,
        fontsize=11,
        fontweight='bold',
        color=color_titulo,
        pad=15,
    )
    ax.set_xlabel('')
    ax.set_ylabel(
        ylabel,
        fontsize=9,
        fontweight='bold',
        color=color_titulo,
    )
    ax.tick_params(
        axis='x',
        rotation=90,
        labelsize=8,
        labelcolor='black',
    )
    ax.tick_params(
        axis='y',
        labelsize=8,
        labelcolor='black',
    )
    ax.grid(
        True,
        axis='y',
        linestyle='--',
        alpha=0.3,
        color='gray',
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_materialidad_leads_cluster_param(
    df,
    col_periodo='periodo',
    col_cluster='cluster',
    modo='global',
):
    df_pivot = (
        df.groupby([col_periodo, col_cluster], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if modo == 'promedio':
        df_pct_mensual = (
            df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100
        )
        s_res = df_pct_mensual.mean(axis=0)
        s_res = s_res / s_res.sum() * 100

        titulo = (
            'Leads Oferta Relevante:\n'
            'Materialidad N (%) (Promedio Mensual)'
        )
    else:
        df_total_global = df_pivot.sum(axis=0)
        s_res = df_total_global / df_total_global.sum() * 100

        titulo = (
            'Leads Oferta Relevante:\n'
            'Materialidad N (%) (Acumulado Global)'
        )

    s_res = s_res.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    barras = ax.bar(
        np.arange(len(s_res)),
        s_res.values,
        width=1.0,
        color='#1D4290',
        edgecolor='black',
        linewidth=0.6,
        alpha=0.9,
    )

    ax.set_xticks(np.arange(len(s_res)))
    ax.set_xticklabels(s_res.index)

    ax.set_ylim(0, s_res.max() * 1.18)

    for barra, valor in zip(barras, s_res):
        ax.annotate(
            f'{valor:.1f}%',
            (
                barra.get_x() + barra.get_width() / 2,
                valor,
            ),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=7.5,
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.22',
                fc='#E8F1F5',
                ec='none',
                alpha=0.9,
            ),
        )

    _aplicar_estilo_ejecutivo(
        ax,
        titulo,
        'Participación de Leads (%)',
    )

    plt.tight_layout()
    plt.show()

    return s_res


def plot_materialidad_monto_cluster_ibk_param(
    df,
    col_periodo='periodo',
    col_cluster='cluster',
    col_monto='monto_desembolsado',
    modo='global',
):
    df_exitos = df[df['indicador_exito'] == 1]

    df_pivot = (
        df_exitos.groupby(
            [col_periodo, col_cluster],
            observed=True,
        )[col_monto]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    if modo == 'promedio':
        df_pct_mensual = (
            df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100
        )
        s_res = df_pct_mensual.mean(axis=0)
        s_res = s_res / s_res.sum() * 100

        titulo = (
            'Desembolsos IBK:\n'
            'Materialidad Monto (%) (Promedio Mensual)'
        )
    else:
        df_total_global = df_pivot.sum(axis=0)
        s_res = df_total_global / df_total_global.sum() * 100

        titulo = (
            'Desembolsos IBK:\n'
            'Materialidad Monto (%) (Acumulado Global)'
        )

    s_res = s_res.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    barras = ax.bar(
        np.arange(len(s_res)),
        s_res.values,
        width=1.0,
        color='#27D75A',
        edgecolor='black',
        linewidth=0.6,
        alpha=0.9,
    )

    ax.set_xticks(np.arange(len(s_res)))
    ax.set_xticklabels(s_res.index)

    ax.set_ylim(0, s_res.max() * 1.18)

    for barra, valor in zip(barras, s_res):
        ax.annotate(
            f'{valor:.1f}%',
            (
                barra.get_x() + barra.get_width() / 2,
                valor,
            ),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=7.5,
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.22',
                fc='#E8F8F5',
                ec='none',
                alpha=0.9,
            ),
        )

    _aplicar_estilo_ejecutivo(
        ax,
        titulo,
        'Participación del Monto IBK (%)',
        color_titulo='#27D75A',
    )

    plt.tight_layout()
    plt.show()

    return s_res



res_leads_global = plot_materialidad_leads_cluster_param(
    datos_oferta_relevante,
    modo='global',
    col_cluster='cluster_comercial',
)

res_monto_global = plot_materialidad_monto_cluster_ibk_param(
    datos_oferta_relevante,
    modo='global',
    col_cluster='cluster_comercial',
)


res_leads_promedio = plot_materialidad_leads_cluster_param(
    datos_oferta_relevante,
    modo='promedio',
    col_cluster='cluster_comercial',
)

res_monto_promedio = plot_materialidad_monto_cluster_ibk_param(
    datos_oferta_relevante,
    modo='promedio',
    col_cluster='cluster_comercial',
)

# Función clara para calcular el promedio ponderado de forma segura
def promedio_ponderado(valores, pesos):
  mask = valores.notna() & pesos.notna() & (pesos > 0)
  if mask.any():
    return np.average(valores[mask], weights=pesos[mask])
  return np.nan


# Pipeline de agregación con las nuevas métricas de efectividad y conversión
resultado = (
    datos_oferta_relevante.groupby(["cluster_comercial", "periodo"], observed=True).agg(
        leads=("llave_interna", "size"),
        monto_ibk=("monto_desembolsado", "sum"),
        monto_oferta=("monto_min", "sum"),
        desembolsos=("indicador_exito", "sum"),
        tasa_des_ponderada=(
            "tasa_2022",
            lambda x: promedio_ponderado(
                x, datos_oferta_relevante.loc[x.index, "monto_desembolsado"]
            ),
        ),
        tasa_oferta_ponderada=(
            "tasa_min",
            lambda x: promedio_ponderado(
                x, datos_oferta_relevante.loc[x.index, "monto_min"]
            ),
        ),
    )
    .assign(
        # Efectividad
        efectividad=lambda df: np.where(
            df["monto_oferta"] > 0, df["monto_ibk"] / df["monto_oferta"], 0
        ),
        # Ratio de conversión:
        conversion=lambda df: np.where(
            df["leads"] > 0, df["desembolsos"] / df["leads"], 0
        ),
    )
    .reset_index()
)

def plot_evolutivo_tasa_metricas(
    df,
    metrica_derecha='efectividad',
    titulo=None,
    nombre_izquierda=None,
    nombre_derecha='Tasa Oferta',
    flg_materialidad=True,
    flg_etiqueta_izquierda=True,
    flg_etiqueta_derecha=True,
    color_izquierda='#27D75A',
    color_derecha='#1D4290',
):
    if df.empty:
        print('El DataFrame no contiene registros para graficar')
        return

    sub_df = df.sort_values('periodo').copy()
    nombres_metricas = {'efectividad': 'Efectividad', 'conversion': 'Conversión'}
    nombre_izquierda = nombre_izquierda or nombres_metricas.get(metrica_derecha, metrica_derecha)

    meses = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
             7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
    sub_df['periodo_label'] = sub_df['periodo'].apply(
        lambda x: f"{meses[int(x) % 100]}' {str(int(x) // 100)[-2:]}"
    )

    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sub_df))

    if flg_materialidad:
        ax_bar = ax1.twinx()
        ax_bar.set_zorder(0)
        ax1.set_zorder(2)
        ax1.patch.set_visible(False)

        barras = ax_bar.bar(
            x, sub_df['desembolsos'],
            width=0.65, color='lightgray', alpha=0.22, zorder=0
        )

        ax_bar.set_ylim(0, sub_df['desembolsos'].max() * 1.25)
        ax_bar.set_yticks([])

        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        for barra, valor in zip(barras, sub_df['desembolsos']):
            ax_bar.text(
                barra.get_x() + barra.get_width() / 2,
                sub_df['desembolsos'].max() * 0.02,
                f'n={int(valor):,}',
                ha='center',
                va='bottom',
                fontsize=6.5,
                color='gray',
                alpha=0.8,
            )

    # Efectividad / Conversión
    y_izq = sub_df[metrica_derecha] * 100

    linea1 = ax1.plot(
        x, y_izq,
        color=color_izquierda,
        marker='o',
        markersize=4,
        linewidth=2,
        label=nombre_izquierda,
        zorder=3,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_df['periodo_label'])
    ax1.set_xlabel('Periodo', fontsize=9, fontweight='bold')
    ax1.set_ylabel(nombre_izquierda, fontsize=9, fontweight='bold', color=color_izquierda)
    ax1.tick_params(axis='both', labelsize=8, labelcolor='black')
    ax1.tick_params(axis='x', rotation=90)

    # Tasa Oferta
    ax2 = ax1.twinx()
    y_der = sub_df['tasa_oferta_ponderada'] * 100

    linea2 = ax2.plot(
        x, y_der,
        color=color_derecha,
        marker='s',
        markersize=4,
        linewidth=2,
        linestyle='--',
        label=nombre_derecha,
        zorder=3,
    )

    ax2.set_ylabel(nombre_derecha, fontsize=9, fontweight='bold', color=color_derecha)
    ax2.tick_params(axis='y', labelsize=8, labelcolor='black')

    formato_pct = plt.FuncFormatter(lambda x, _: f'{x:.1f}%')
    ax1.yaxis.set_major_formatter(formato_pct)
    ax2.yaxis.set_major_formatter(formato_pct)

    # Etiquetas Efectividad / Conversión
    if flg_etiqueta_izquierda:
        for xi, valor in zip(x, y_izq):
            ax1.annotate(
                f'{valor:.1f}%',
                (xi, valor),
                xytext=(0, 9),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=7,
                color='black',
                bbox=dict(
                    boxstyle='round,pad=0.22',
                    fc='#E8F8F5',
                    ec=color_izquierda,
                    lw=0.5,
                    alpha=0.9,
                ),
            )

    # Etiquetas Tasa Oferta
    if flg_etiqueta_derecha:
        for xi, valor in zip(x, y_der):
            ax2.annotate(
                f'{valor:.1f}%',
                (xi, valor),
                xytext=(0, -10),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=7,
                color='black',
                bbox=dict(
                    boxstyle='round,pad=0.22',
                    fc='#E8F1F5',
                    ec=color_derecha,
                    lw=0.5,
                    alpha=0.9,
                ),
            )

    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    ax2.grid(False)

    lineas = linea1 + linea2
    etiquetas = [linea.get_label() for linea in lineas]

    ax1.legend(
        lineas, etiquetas,
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='lightgray',
        fontsize=8,
    )

    titulo = titulo or f'Evolución de {nombre_izquierda.lower()} y {nombre_derecha.lower()}'
    ax1.set_title(
        titulo,
        fontsize=11,
        fontweight='bold',
        color=color_derecha,
        pad=15,
    )

    fig.tight_layout()
    plt.show()

for bucket in resultado['cluster_comercial'].unique():
    df_filtrado = resultado.loc[resultado["cluster_comercial"] == bucket].copy()
    plot_evolutivo_tasa_metricas(
        df=df_filtrado,
        metrica_derecha='efectividad',
        titulo=f"Evolución de Tasa Oferta y Efectividad\n{bucket}",
         flg_materialidad=True,
         flg_etiqueta_izquierda=False,
         flg_etiqueta_derecha=False
    )

#     plot_evolutivo_tasa_metricas(
#         df=df_filtrado,
#         metrica_derecha='conversion',
#         titulo=f"Evolución de Tasa Oferta y Conversion\n{bucket}",
#          flg_materialidad=True,
#          flg_etiqueta_izquierda=False,
#          flg_etiqueta_derecha=False
#     )

datos_oferta_relevante.query('monto_desembolsado.notna()').value_counts('flg_exito_4') # Es porque para los ibk en otros bancos se toma el primer registro, por eso tienen monto desembolsado y flg_exito_4 = 'fracaso_desembolso_no_ibk_monto_similar_con_gestion' (no es un error, es por la lógica de selección de registros)
datos_oferta_relevante.query('monto_desembolsado.isna()').value_counts('flg_exito_4')
datos_oferta_relevante.query('monto_desembolsado.notna() and monto_desembolsado > monto_min+20')[['llave_interna', 'monto_desembolsado', 'monto_max', 'flg_exito_4']]


resultado[['cluster_comercial', 'periodo','tasa_oferta_ponderada', 'efectividad', 'conversion']]

