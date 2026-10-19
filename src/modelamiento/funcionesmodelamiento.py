################################################## funcionesmodelamiento ############################################

############################################################################################################
# ==========================================
# Librerías estándar
# ==========================================
import gc
import os
import random
import re
import string
import sys
import unicodedata
from itertools import combinations

# ==========================================
# Manipulación de datos
# ==========================================
import numpy as np
import pandas as pd

# ==========================================
# Exportación a Excel
# ==========================================
import xlsxwriter
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

# ==========================================
# Visualización
# ==========================================
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# Estadística
# ==========================================
from scipy.stats import chi2_contingency, norm, ttest_ind
from scipy.stats import ks_2samp, ttest_ind

# ==========================================
# Modelos estadísticos
# ==========================================
import statsmodels.api as sm
from linearmodels.panel import RandomEffects

# ==========================================
# Machine Learning
# ==========================================
from sklearn.calibration import calibration_curve
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

# ==========================================
# Gradient Boosting
# ==========================================
import xgboost as xgb

# ==========================================
# Explainability
# ==========================================
import shap

#_____________________________ Framework: Analisis Descriptivo ----

#---- resumen_tea ---- 
def resumen_tea(df, variable, var_tasa="tea_aplicada", var_ponderacion="monto_operacion"):
    return (
        df.groupby(variable)
        .apply(lambda x: pd.Series({
            "tea_ponderada": np.average(
                x[var_tasa],
                weights=x[var_ponderacion]
            ),
            "monto_desembolsado": x[var_ponderacion].sum(),
            "operaciones": len(x)
        }))
        .reset_index()
    )

#---- plot_distribucion ---- 
def plot_distribucion(df, variable, bins=50, figsize=(10, 5)):
    """
    Grafica la distribución y el boxplot de una variable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de entrada.
    variable : str
        Nombre de la variable a graficar.
    bins : int, default=50
        Número de bins del histograma.
    figsize : tuple, default=(10,5)
        Tamaño de la figura.
    """

    datos = df[variable].dropna()

    fig, ax = plt.subplots(
        2, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 4]},
        sharex=True
    )

    # Boxplot
    ax[0].boxplot(datos, vert=False)
    ax[0].set_title(f"Boxplot - {variable}")
    ax[0].set_yticks([])

    # Histograma
    ax[1].hist(datos, bins=bins)
    ax[1].set_title(f"Distribución - {variable}")
    ax[1].set_xlabel(variable)
    ax[1].set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

#---- plot_tea_ponderada_mes ----
def plot_tea_ponderada_mes(
    df,
    periodo="periodo",
    tasa="tea_aplicada",
    peso="monto_operacion",
    filtro=None,
    titulo=None
):
    """
    Grafica la evolución mensual de la TEA ponderada junto con la
    dispersión mensual de las tasas (P10, P25, P75 y P90).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de entrada.
    periodo : str, default="periodo"
        Variable de periodo en formato YYYYMM.
    tasa : str, default="tea_aplicada"
        Variable de tasa.
    peso : str, default="monto_operacion"
        Variable utilizada como ponderador.
    filtro : str, optional
        Filtro utilizando pandas.query().
    titulo : str, optional
        Título del gráfico.

    Returns
    -------
    pandas.DataFrame
        Resumen mensual utilizado para la gráfica.
    """

    data = df.copy()

    if filtro is not None:
        data = data.query(filtro)

    resumen = (
        data.groupby(periodo)
        .apply(
            lambda x: pd.Series({
                "tea_ponderada": np.average(
                    x[tasa],
                    weights=x[peso]
                ),
                "p10": x[tasa].quantile(0.10),
                "p25": x[tasa].quantile(0.25),
                "p75": x[tasa].quantile(0.75),
                "p90": x[tasa].quantile(0.90),
                "n_operaciones": len(x),
                "monto_desembolsado": x[peso].sum()
            })
        )
        .reset_index()
        .sort_values(periodo)
    )

    # Convertir YYYYMM a fecha
    resumen["fecha"] = pd.to_datetime(
        resumen[periodo].astype(str),
        format="%Y%m"
    )

    # -------------------------------------------------------------------------
    # Gráfico
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    # Banda intercuartílica (P25 - P75)
    ax.fill_between(
        resumen["fecha"],
        resumen["p25"],
        resumen["p75"],
        alpha=0.25,
        label="P25 - P75"
    )

    # Bigote inferior (P10 - P25)
    ax.vlines(
        resumen["fecha"],
        resumen["p10"],
        resumen["p25"],
        linewidth=1.3
    )

    # Bigote superior (P75 - P90)
    ax.vlines(
        resumen["fecha"],
        resumen["p75"],
        resumen["p90"],
        linewidth=1.3
    )

    # Topes de los bigotes
    ancho = pd.Timedelta(days=4)

    for _, row in resumen.iterrows():
        ax.hlines(
            row["p10"],
            row["fecha"] - ancho,
            row["fecha"] + ancho,
            linewidth=1.3
        )

        ax.hlines(
            row["p90"],
            row["fecha"] - ancho,
            row["fecha"] + ancho,
            linewidth=1.3
        )

    # TEA ponderada
    ax.plot(
        resumen["fecha"],
        resumen["tea_ponderada"],
        marker="o",
        linewidth=2.5,
        label="TEA ponderada"
    )

    # Formato del eje X
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    ax.set_xlabel("Periodo")
    ax.set_ylabel("TEA (%)")

    if titulo is None:
        titulo = "Evolución mensual de la TEA ponderada"

    ax.set_title(titulo)

    ax.grid(alpha=0.3)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#---- plot_tea_ponderada_categoria ----
def plot_tea_ponderada_categoria(
    df,
    categoria,
    tasa="tea_aplicada",
    peso="monto_operacion",
    filtro=None,
    titulo=None,
    figsize=(11, 6)
):
    """
    Grafica la distribución de la TEA por una variable categórica utilizando:
    - P10 y P90 como bigotes.
    - P25 y P75 como caja.
    - TEA ponderada como indicador central.

    Parameters
    ----------
    df : pandas.DataFrame
    categoria : str
        Variable categórica.
    tasa : str
        Variable de tasa.
    peso : str
        Variable utilizada como ponderador.
    filtro : str, optional
        Filtro utilizando pandas.query().
    titulo : str, optional
        Título del gráfico.

    Returns
    -------
    pandas.DataFrame
        Resumen por categoría.
    """

    data = df.copy()

    if filtro is not None:
        data = data.query(filtro)

    resumen = (
        data.groupby(categoria,observed=True)
        .apply(
            lambda x: {
                "tea_ponderada": np.average(
                    x[tasa],
                    weights=x[peso]
                ),
                "p10": x[tasa].quantile(0.10),
                "p25": x[tasa].quantile(0.25),
                "p75": x[tasa].quantile(0.75),
                "p90": x[tasa].quantile(0.90),
                "n": len(x),
                "monto": x[peso].sum()
            }
        )
        .apply(pd.Series)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    ancho = 0.30

    for i, row in resumen.iterrows():

        # Caja
        ax.add_patch(
            plt.Rectangle(
                (i - ancho / 2, row["p25"]),
                ancho,
                row["p75"] - row["p25"],
                alpha=0.25
            )
        )

        # Bigote inferior
        ax.vlines(
            i,
            row["p10"],
            row["p25"],
            linewidth=1.5
        )

        # Bigote superior
        ax.vlines(
            i,
            row["p75"],
            row["p90"],
            linewidth=1.5
        )

        # Topes
        ax.hlines(
            row["p10"],
            i - ancho / 4,
            i + ancho / 4,
            linewidth=1.5
        )

        ax.hlines(
            row["p90"],
            i - ancho / 4,
            i + ancho / 4,
            linewidth=1.5
        )

        # TEA ponderada
        ax.scatter(
            i,
            row["tea_ponderada"],
            s=60,
            zorder=3,
            label="TEA ponderada" if i == 0 else ""
        )

    ax.set_xticks(range(len(resumen)))
    ax.set_xticklabels(resumen[categoria], rotation=45, ha="right")

    ax.set_ylabel("TEA (%)")

    if titulo is None:
        titulo = f"Distribución de TEA por {categoria}"

    ax.set_title(titulo)

    ax.grid(alpha=0.30)

    ax.legend()

    plt.tight_layout()
    plt.show()

    return resumen

#---- heatmap_distribucion ----
def heatmap_distribucion(
    df,
    index,
    columns,
    value,
    aggfunc="sum",
    porcentaje=True,
    flg_dinero=False,
    flg_magnitud="MM",
    flg_ceros=False,
    figsize=(8, 6)
):

    # Agregación
    tabla = (
        df.groupby([index, columns],observed=True)
        .agg(valor=(value, aggfunc))
        .reset_index()
    )

    # Orden
    tabla = tabla.sort_values([index, columns])

    # Pivot
    tabla = (
        tabla
        .pivot(index=index, columns=columns, values="valor")
        .fillna(0)
    )

    tabla = tabla.sort_index(axis=0)
    tabla = tabla.sort_index(axis=1)
    tabla = tabla.sort_index(axis=0, ascending=False)

    #---------------------------------------
    # Conversión de magnitud
    #---------------------------------------

    sufijo = ""

    if not porcentaje and flg_dinero:

        if flg_magnitud == "MM":
            tabla = tabla / 1_000_000
            sufijo = " (MM)"

        elif flg_magnitud == "K":
            tabla = tabla / 1_000
            sufijo = " (K)"

    #---------------------------------------
    # Porcentaje
    #---------------------------------------

    if porcentaje:
        tabla = tabla / tabla.values.sum() * 100
        titulo = f"Distribución de {value} (%)"
        cbar_label = "Participación (%)"

    else:
        titulo = f"Distribución de {value}{sufijo}"
        cbar_label = value + sufijo

    #---------------------------------------
    # Ocultar ceros (NO pintarlos)
    #---------------------------------------

    tabla_plot = tabla.copy()

    if not flg_ceros:
        tabla_plot = tabla_plot.mask(np.isclose(tabla_plot, 0))

    #---------------------------------------
    # Anotaciones
    #---------------------------------------

    annot = tabla_plot.apply(
        lambda col: col.map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )
    )

    #---------------------------------------
    # Heatmap
    #---------------------------------------

    plt.figure(figsize=figsize)

    sns.heatmap(
        tabla_plot,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": cbar_label}
    )

    plt.title(titulo)
    plt.xlabel(columns)
    plt.ylabel(index)

    plt.tight_layout()
    plt.show()
    
#---- heatmap_participacion_columna ----
def heatmap_participacion_columna(
    df,
    index,
    columns,
    value=None,
    aggfunc="count",
    figsize=(8, 6)
):

    # Agregación
    if value is None:
        tabla = (
            df.groupby([index, columns])
            .size()
            .reset_index(name="valor")
        )
    else:
        tabla = (
            df.groupby([index, columns])
            .agg(valor=(value, aggfunc))
            .reset_index()
        )

    # Pivot
    tabla = (
        tabla
        .pivot(
            index=index,
            columns=columns,
            values="valor"
        )
        .fillna(0)
    )

    # Orden
    tabla = tabla.sort_index(axis=0)
    tabla = tabla.sort_index(axis=1)

    # Participación dentro de cada columna
    tabla_pct = (
        tabla
        .div(tabla.sum(axis=0), axis=1)
        * 100
    )

    # Orden vertical descendente
    tabla_pct = tabla_pct.sort_index(ascending=False)

    # Heatmap
    plt.figure(figsize=figsize)

    sns.heatmap(
        tabla_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues"
    )

    plt.title(
        f"Participación de {index} dentro de {columns} (%)"
    )

    plt.xlabel(columns)
    plt.ylabel(index)

    plt.tight_layout()
    plt.show()

    return tabla_pct

#---- heatmap_cv_bucket ----
def heatmap_cv_bucket(
    df,
    index,
    columns,
    variable,
    figsize=(10, 6)
):
    """
    Heatmap del coeficiente de variación (CV) por segmento.

    CV = desviación estándar / media

    Permite comparar dispersión relativa eliminando efecto de magnitud.
    """

    # Cálculo CV por bucket
    resumen = (
        df.groupby([index, columns],observed=True)
        .apply(
            lambda x: pd.Series({
                "media": x[variable].mean(),
                "std": x[variable].std(),
                "cv": (
                    x[variable].std() / x[variable].mean()
                    if x[variable].mean() != 0 else np.nan
                ),
                "n_operaciones": len(x)
            })
        )
        .reset_index()
    )

    # Pasar a porcentaje
    resumen["cv_pct"] = resumen["cv"] * 100

    # Pivot para heatmap
    tabla = resumen.pivot(
        index=index,
        columns=columns,
        values="cv_pct"
    )

    tabla = tabla.sort_index(axis=0, ascending=False)
    tabla = tabla.sort_index(axis=1)

    plt.figure(figsize=figsize)

    sns.heatmap(
        tabla,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "CV (%)"}
    )

    plt.title("Coeficiente de variación por segmento")
    plt.xlabel(columns)
    plt.ylabel(index)

    plt.tight_layout()
    plt.show()

    return resumen

#---- heatmap_tasa_ponderada ----
def heatmap_tasa_ponderada(
    df,
    index,
    columns,
    tasa,
    peso,
    figsize=(10, 6)
):
    """
    Heatmap de tasa ponderada por segmento.

    Detecta automáticamente si la tasa está en:
    - formato decimal (0-1)
    - formato porcentual (0-100)
    """

    # Cálculo tasa ponderada
    resumen = (
        df.groupby([index, columns],observed=True)
        .apply(
            lambda x: pd.Series({
                "tasa_ponderada": np.average(
                    x[tasa],
                    weights=x[peso]
                ),
                "monto_desembolsado": x[peso].sum(),
                "n_operaciones": len(x)
            })
        )
        .reset_index()
    )

    # Detectar escala de la tasa
    if resumen["tasa_ponderada"].max() <= 1:
        resumen["tasa_plot"] = resumen["tasa_ponderada"] * 100
    else:
        resumen["tasa_plot"] = resumen["tasa_ponderada"]

    # Pivot
    tabla = resumen.pivot(
        index=index,
        columns=columns,
        values="tasa_plot"
    )

    tabla = tabla.sort_index(axis=0, ascending=False)
    tabla = tabla.sort_index(axis=1)

    plt.figure(figsize=figsize)

    sns.heatmap(
        tabla,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "Tasa ponderada (%)"}
    )

    plt.title("Tasa ponderada por segmento")
    plt.xlabel(columns)
    plt.ylabel(index)

    plt.tight_layout()
    plt.show()

    return resumen

#---- plot_tea_ponderada_categoria_evolutivo ----
def plot_tea_ponderada_categoria_evolutivo(
    df,
    categoria="segmento_arbol_new",
    periodo="periodo",
    tasa="tea_aplicada",
    peso="monto_operacion",
    filtro=None,
    titulo=None
):

    data = df.copy()

    if filtro is not None:
        data = data.query(filtro)


    # ---------------------------------------------------------------
    # Resumen periodo + categoria
    # ---------------------------------------------------------------

    resumen = (
        data.groupby([periodo, categoria], observed=True)
        .apply(
            lambda x: pd.Series({
                "tea_ponderada": np.average(
                    x[tasa],
                    weights=x[peso]
                ),
                "p10": x[tasa].quantile(0.10),
                "p25": x[tasa].quantile(0.25),
                "p75": x[tasa].quantile(0.75),
                "p90": x[tasa].quantile(0.90),
                "n_operaciones": len(x),
                "monto_desembolsado": x[peso].sum()
            })
        )
        .reset_index()
        .sort_values([categoria, periodo])
    )


    # YYYYMM -> fecha
    resumen["fecha"] = pd.to_datetime(
        resumen[periodo].astype(str),
        format="%Y%m"
    )


    fig = go.Figure()

    colores = px.colors.qualitative.Plotly


    # ---------------------------------------------------------------
    # Capas inferiores: sombra + bigotes
    # ---------------------------------------------------------------

    for i, cat in enumerate(resumen[categoria].unique()):

        tmp = resumen[
            resumen[categoria] == cat
        ].copy()

        color = colores[i % len(colores)]


        # -----------------------------------------------------------
        # Banda P25-P75
        # -----------------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=list(tmp["fecha"]) + list(tmp["fecha"][::-1]),
                y=list(tmp["p75"]) + list(tmp["p25"][::-1]),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                hoverinfo="skip",
                legendgroup=str(cat),
                showlegend=False
            )
        )


        # -----------------------------------------------------------
        # Bigotes verticales + topes horizontales
        # -----------------------------------------------------------

        for _, row in tmp.iterrows():

            x = row["fecha"]

            ancho = pd.Timedelta(days=2)


            # Bigote inferior P10 -> TEA
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[
                        row["p10"],
                        row["tea_ponderada"]
                    ],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=1
                    ),
                    opacity=0.6,
                    hoverinfo="skip",
                    legendgroup=str(cat),
                    showlegend=False
                )
            )


            # Bigote superior TEA -> P90
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[
                        row["tea_ponderada"],
                        row["p90"]
                    ],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=1
                    ),
                    opacity=0.6,
                    hoverinfo="skip",
                    legendgroup=str(cat),
                    showlegend=False
                )
            )


            # Tope P10
            fig.add_trace(
                go.Scatter(
                    x=[x-ancho, x+ancho],
                    y=[row["p10"], row["p10"]],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=2
                    ),
                    opacity=0.6,
                    hoverinfo="skip",
                    legendgroup=str(cat),
                    showlegend=False
                )
            )


            # Tope P90
            fig.add_trace(
                go.Scatter(
                    x=[x-ancho, x+ancho],
                    y=[row["p90"], row["p90"]],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=2
                    ),
                    opacity=0.6,
                    hoverinfo="skip",
                    legendgroup=str(cat),
                    showlegend=False
                )
            )


    # ---------------------------------------------------------------
    # TEA ponderada encima de todo
    # ---------------------------------------------------------------

    for i, cat in enumerate(resumen[categoria].unique()):

        tmp = resumen[
            resumen[categoria] == cat
        ].copy()

        color = colores[i % len(colores)]


        fig.add_trace(
            go.Scatter(
                x=tmp["fecha"],
                y=tmp["tea_ponderada"],
                mode="lines+markers",
                name=str(cat),
                line=dict(
                    color=color,
                    width=4
                ),
                marker=dict(
                    size=9,
                    color=color,
                    line=dict(
                        color="white",
                        width=1.5
                    )
                ),
                legendgroup=str(cat)
            )
        )


    fig.update_layout(
        title=titulo or "Evolución mensual de TEA ponderada por categoría",
        xaxis_title="Periodo",
        yaxis_title="TEA ponderada (%)",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        legend_title=categoria,

        xaxis=dict(
            tickformat="%b %y",
            dtick="M1"
        )
    )

    fig.show()

    return resumen

#---- descriptivos_variables ----
def descriptivos_variables(df, variables):
    """
    Genera descriptivos para variables numéricas y categóricas.

    Parámetros:
    -----------
    df : DataFrame
        Base de datos.
    variables : list
        Lista de variables a evaluar.

    Retorna:
    --------
    df_numericas : DataFrame
        Estadísticos descriptivos de variables numéricas.
    df_categoricas : DataFrame
        Distribución de frecuencias de variables categóricas.
    """

    df_numericas = []
    df_categoricas = []

    for var in variables:
        
        if var not in df.columns:
            continue
        
        serie = df[var]

        # Variables numéricas
        if pd.api.types.is_numeric_dtype(serie):
            
            desc = {
                "Variable": var,
                "count": serie.count(),
                "mean": serie.mean(),
                "std": serie.std(),
                "min": serie.min(),
                "0%": serie.quantile(0),
                "10%": serie.quantile(0.10),
                "20%": serie.quantile(0.20),
                "30%": serie.quantile(0.30),
                "40%": serie.quantile(0.40),
                "50%": serie.quantile(0.50),
                "60%": serie.quantile(0.60),
                "70%": serie.quantile(0.70),
                "80%": serie.quantile(0.80),
                "90%": serie.quantile(0.90),
                "100%": serie.quantile(1),
                "max": serie.max(),
                "skewness": serie.skew(),
                "kurtosis": serie.kurtosis()
            }

            df_numericas.append(desc)

        # Variables categóricas
        else:
            
            freq = (
                serie.fillna("NULO")
                .value_counts(dropna=False)
                .reset_index()
            )

            freq.columns = ["Categorias", "Frecuencia"]

            freq["Variable"] = var
            freq["Frecuencia %"] = (
                freq["Frecuencia"] / freq["Frecuencia"].sum() * 100
            )

            freq = freq[
                ["Variable", "Categorias", "Frecuencia", "Frecuencia %"]
            ]

            df_categoricas.append(freq)

    df_numericas = pd.DataFrame(df_numericas)

    if len(df_categoricas) > 0:
        df_categoricas = pd.concat(
            df_categoricas,
            ignore_index=True
        )
    else:
        df_categoricas = pd.DataFrame(
            columns=["Variable", "Categorias", "Frecuencia", "Frecuencia %"]
        )

    return df_numericas, df_categoricas

#---- graficar_distribucion ----
def graficar_distribucion(df, variable, max_categorias=5, bins=30):
    """
    Grafica distribución de variables numéricas y categóricas.

    Numéricas:
    - Histograma
    - Estadísticos: media, P5, P25, P50, P75, P95

    Categóricas:
    - Barras horizontales
    - Agrupación de categorías poco representativas en OTRAS

    Parámetros:
    -----------
    df : DataFrame
    variable : str
    max_categorias : int
        Máximo de categorías a mostrar.
    bins : int
        Número de intervalos del histograma.
    """

    serie = df[variable]

    # ==========================
    # Variable numérica
    # ==========================
    if pd.api.types.is_numeric_dtype(serie):

        datos = serie.dropna()

        estadisticos = {
            "Media": datos.mean(),
            "P1": datos.quantile(0.01),
            "P5": datos.quantile(0.05),
            "P25": datos.quantile(0.25),
            "P50": datos.quantile(0.50),
            "P75": datos.quantile(0.75),
            "P95": datos.quantile(0.95),
            "P99": datos.quantile(0.99),
        }

        plt.figure(figsize=(9, 5))

        plt.hist(
            datos,
            bins=bins
        )

        plt.title(f"Distribución - {variable}")
        plt.xlabel(variable)
        plt.ylabel("Frecuencia")

        plt.grid(
            alpha=0.3
        )

        # Cuadro de estadísticos
        texto = "\n".join(
            [
                f"{k}: {v:,.2f}"
                for k, v in estadisticos.items()
            ]
        )

        plt.text(
            0.98,
            0.95,
            texto,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round"
            )
        )

        plt.show()


    # ==========================
    # Variable categórica
    # ==========================
    else:

        dist = (
            serie.fillna("NULO")
            .value_counts(normalize=True)
            .reset_index()
        )

        dist.columns = [
            "Categoria",
            "Porcentaje"
        ]

        dist["Porcentaje"] *= 100


        if len(dist) > max_categorias:

            principales = dist.head(max_categorias)

            otras = pd.DataFrame({
                "Categoria": ["OTRAS"],
                "Porcentaje": [
                    dist.iloc[max_categorias:]["Porcentaje"].sum()
                ]
            })

            dist = pd.concat(
                [
                    principales,
                    otras
                ],
                ignore_index=True
            )


        dist = dist.sort_values(
            "Porcentaje",
            ascending=True
        )


        plt.figure(figsize=(8, 4))

        plt.barh(
            dist["Categoria"],
            dist["Porcentaje"]
        )


        for i, valor in enumerate(dist["Porcentaje"]):
            plt.text(
                valor,
                i,
                f"{valor:.1f}%",
                va="center"
            )


        plt.title(f"Distribución - {variable}")
        plt.xlabel("% de casos")
        plt.ylabel("")

        plt.xlim(
            0,
            dist["Porcentaje"].max()*1.15
        )

        plt.grid(
            axis="x",
            alpha=0.3
        )

        plt.show()

#_____________________________ Framework: Modelamiento ----

#---- aplicar_reglas_negocio_na ----
def aplicar_reglas_negocio_na(
    X,
    reglas_imputacion
):
    """
    Aplica valores definidos por reglas de negocio para registros NA.

    Esta etapa corresponde a una homologación de información,
    donde el valor faltante tiene una interpretación conocida
    del negocio. No corresponde a una imputación estadística.

    Ejemplo:
    {
        "saldo_pasivo": 0,
        "nro_productos": 0,
        "estado_civil": "SIN_INFO",
        "profesion": "NO_DECLARADO"
    }

    """

    X = X.copy()

    resumen = []

    for variable, valor in reglas_imputacion.items():

        if variable in X.columns:

            registros_imputados = (
                X[variable]
                .isna()
                .sum()
            )

            X[variable] = (
                X[variable]
                .fillna(valor)
            )

            resumen.append(
                {
                    "variable": variable,
                    "valor_asignado": valor,
                    "registros_modificados": registros_imputados
                }
            )

    resumen = pd.DataFrame(resumen)

    return X, resumen

#---- comparar_distribuciones ----
def comparar_distribuciones(
    df_x,
    df_y,
    variable,
    peso=None,
    alpha=0.05,
    graficar=True,
    nombre_x="Base X",
    nombre_y="Base Y",
):
    """
    Compara la distribución de una variable entre dos bases.

    Hipótesis
    ----------
    KS Test
        H0: Ambas muestras provienen de la misma distribución.
        H1: Las distribuciones son diferentes.

    Welch t-test
        H0: Las medias poblacionales son iguales.
        H1: Las medias poblacionales son diferentes.
    """

    columnas = [variable] + ([peso] if peso else [])

    x = df_x[columnas].dropna()
    y = df_y[columnas].dropna()

    #===========================
    # Estadísticos descriptivos
    #===========================

    resultado = {
        "n_x": len(x),
        "n_y": len(y),

        "media_x": x[variable].mean(),
        "media_y": y[variable].mean(),
        "dif_media": x[variable].mean() - y[variable].mean(),

        "mediana_x": x[variable].median(),
        "mediana_y": y[variable].median(),

        "std_x": x[variable].std(),
        "std_y": y[variable].std(),
    }

    if peso is not None:

        resultado["media_ponderada_x"] = np.average(
            x[variable],
            weights=x[peso]
        )

        resultado["media_ponderada_y"] = np.average(
            y[variable],
            weights=y[peso]
        )

        resultado["dif_media_ponderada"] = (
            resultado["media_ponderada_x"]
            - resultado["media_ponderada_y"]
        )

    #===========================
    # KS Test
    #===========================

    ks_stat, ks_p = ks_2samp(
        x[variable],
        y[variable]
    )

    resultado["ks_stat"] = ks_stat
    resultado["ks_pvalue"] = ks_p

    resultado["ks_conclusion"] = (
        "No se rechaza H0: las distribuciones son estadísticamente similares."
        if ks_p >= alpha
        else
        "Se rechaza H0: las distribuciones son estadísticamente diferentes."
    )

    #===========================
    # Welch t-test
    #===========================

    t_stat, t_p = ttest_ind(
        x[variable],
        y[variable],
        equal_var=False
    )

    resultado["t_stat"] = t_stat
    resultado["t_pvalue"] = t_p

    resultado["t_conclusion"] = (
        "No se rechaza H0: no existe evidencia de diferencias en las medias."
        if t_p >= alpha
        else
        "Se rechaza H0: existe evidencia de diferencias en las medias."
    )

    #===========================
    # Gráficos
    #===========================

    if graficar:

        # Histograma
        plt.figure(figsize=(8,4))

        plt.hist(
            x[variable],
            bins=40,
            density=True,
            alpha=0.5,
            label=nombre_x,
        )

        plt.hist(
            y[variable],
            bins=40,
            density=True,
            alpha=0.5,
            label=nombre_y,
        )

        plt.xlabel(variable)
        plt.ylabel("Densidad")
        plt.title(f"Distribución de {variable}")
        plt.legend()
        plt.grid(alpha=0.2)

        plt.show()

        # ECDF
        def ecdf(v):
            v = np.sort(v)
            p = np.arange(1, len(v)+1) / len(v)
            return v, p

        x1, p1 = ecdf(x[variable])
        x2, p2 = ecdf(y[variable])

        plt.figure(figsize=(8,4))

        plt.plot(x1, p1, label=nombre_x)
        plt.plot(x2, p2, label=nombre_y)

        plt.xlabel(variable)
        plt.ylabel("Probabilidad acumulada")
        plt.title(f"ECDF de {variable}")
        plt.legend()
        plt.grid(alpha=0.2)

        plt.show()

    return pd.Series(resultado)

#---- normalizacion_variables_str ----
def normalizacion_variables_str(datos, variables_str):
    """
    Normaliza variables de texto mediante estandarización de formato:
    eliminación de acentos, conversión a mayúsculas y limpieza de espacios.

    Retorna:
    - DataFrame normalizado.
    - Resumen de cambios aplicados por variable.
    """

    datos = datos.copy()

    def quitar_acentos(texto):
        if texto is None:
            return texto

        return (
            unicodedata
            .normalize("NFKD", texto)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

    resumen = []

    for columna in variables_str:

        if columna in datos.columns:

            valores_antes = (
                datos[columna]
                .dropna()
                .nunique()
            )

            registros_con_info = (
                datos[columna]
                .notna()
                .sum()
            )

            # Normalización
            datos[columna] = (
                datos[columna]
                .astype(str)
                .replace(
                    ['nan', 'None', 'nan.0'],
                    np.nan
                )
            )

            mask = datos[columna].notna()

            datos.loc[mask, columna] = (
                datos.loc[mask, columna]
                .str.upper()
                .str.strip()
                .apply(quitar_acentos)
            )

            valores_despues = (
                datos[columna]
                .dropna()
                .nunique()
            )

            resumen.append(
                {
                    "variable": columna,
                    "registros_con_info": registros_con_info,
                    "categorias_antes": valores_antes,
                    "categorias_despues": valores_despues
                }
            )

    resumen = pd.DataFrame(resumen)

    return datos, resumen

#---- homologar_categorias ----
def homologar_categorias(
    X,
    reglas_homologacion
):
    """
    Homologa valores categóricos según reglas de negocio.

    Ejemplo:
    {
        "estado_civil": {
            "Married": "Casado",
            "CASADO": "Casado",
            "Viudo V": "Viudo"
        }
    }
    """

    X = X.copy()

    resumen = []

    for variable, mapa in reglas_homologacion.items():

        if variable in X.columns:

            registros_modificados = (
                X[variable]
                .isin(mapa.keys())
                .sum()
            )

            X[variable] = (
                X[variable]
                .replace(mapa)
            )

            resumen.append(
                {
                    "variable": variable,
                    "registros_homologados": registros_modificados
                }
            )

    return X, pd.DataFrame(resumen)

#---- detectar_variables_constantes ----
def detectar_variables_constantes(
    df,
    umbral_categoria=0.99,
    umbral_unico=0.99,
    tolerancia_std=1e-8
):
    """
    Detecta variables constantes o casi constantes.

    Parámetros
    ----------
    umbral_categoria : float
        Participación máxima de una sola categoría para marcar
        una variable categórica como casi constante.

    umbral_unico : float
        Participación del valor más frecuente en variables numéricas.

    tolerancia_std : float
        Desviación estándar mínima para considerar una variable constante.

    Retorna
    -------
    DataFrame resumen.
    """

    resultados = []

    for col in df.columns:

        s = df[col]
        n_na = s.isna().sum()

        if pd.api.types.is_numeric_dtype(s):

            x = s.dropna()

            if len(x) == 0:
                estado = "Todo NA"

            elif x.nunique() == 1:
                estado = "Constante"

            elif x.std() <= tolerancia_std:
                estado = "Casi constante (std≈0)"

            elif x.value_counts(normalize=True).iloc[0] >= umbral_unico:
                estado = "Casi constante (valor dominante)"

            else:
                estado = "OK"

            resultados.append({
                "variable": col,
                "tipo": "Numerica",
                "estado": estado,
                "n_unicos": x.nunique(),
                "valor_dominante": x.value_counts(normalize=True).iloc[0] if len(x)>0 else np.nan,
                "std": x.std() if len(x)>0 else np.nan,
                "pct_na": n_na/len(s)
            })

        else:

            x = s.dropna()

            if len(x)==0:
                estado="Todo NA"
                dominante=np.nan

            else:
                dominante=x.value_counts(normalize=True).iloc[0]

                if x.nunique()==1:
                    estado="Constante"

                elif dominante>=umbral_categoria:
                    estado="Casi constante"

                else:
                    estado="OK"

            resultados.append({
                "variable":col,
                "tipo":"Categorica",
                "estado":estado,
                "n_unicos":x.nunique(),
                "valor_dominante":dominante,
                "std":np.nan,
                "pct_na":n_na/len(s)
            })

    return (
        pd.DataFrame(resultados)
        .sort_values(["estado","valor_dominante"], ascending=[True,False])
        .reset_index(drop=True)
    )

#---- evaluar_distribucion_numericas ----
def evaluar_cardinalidad_categoricas(
    X,
    variables_categoricas,
    umbral_categorias=5,
    umbral_dominancia=0.95,
    umbral_frecuencia_minima=0.01
):
    """
    Evalúa cardinalidad y distribución de variables categóricas.

    Objetivo:
    - Identificar variables aptas para One Hot Encoding.
    - Detectar variables con alta cardinalidad,
      baja variabilidad o categorías poco frecuentes.

    No realiza tratamientos; genera diagnóstico.
    """

    resumen = []

    for var in variables_categoricas:

        serie = X[var]

        n_registros = len(serie)

        # =========================
        # Información general
        # =========================

        pct_nulos = (
            serie.isnull()
            .mean()
        )

        frecuencia = (
            serie
            .value_counts(
                dropna=True,
                normalize=True
            )
        )

        n_categorias = (
            serie
            .nunique(dropna=True)
        )

        if len(frecuencia) > 0:

            categoria_dominante = (
                frecuencia.index[0]
            )

            pct_dominante = (
                frecuencia.iloc[0]
            )

        else:

            categoria_dominante = None
            pct_dominante = 0


        # Categorías poco frecuentes
        categorias_baja_frecuencia = (
            frecuencia[
                frecuencia < umbral_frecuencia_minima
            ]
            .count()
        )


        # =========================
        # Diagnóstico
        # =========================

        if n_categorias > umbral_categorias:

            interpretacion = (
                "Alta cardinalidad."
            )

            tratamiento = (
                "Evaluar agrupación o encoding alternativo"
            )

            aplica_one_hot = False


        elif pct_dominante >= umbral_dominancia:

            interpretacion = (
                "Baja variabilidad. "
                "Una categoría concentra la mayoría de registros."
            )

            tratamiento = (
                "Evaluar flag de categoría dominante y revisar aporte predictivo"
            )

            aplica_one_hot = False


        elif categorias_baja_frecuencia > 0:

            interpretacion = (
                "Presenta categorías poco frecuentes."
            )

            tratamiento = (
                "Evaluar agrupación de categorías"
            )

            aplica_one_hot = True


        else:

            interpretacion = (
                "Cardinalidad adecuada."
            )

            tratamiento = (
                "Aplicar One Hot Encoding"
            )

            aplica_one_hot = True


        resumen.append(
            {
                "variable": var,
                "n_categorias": n_categorias,
                "pct_nulos": pct_nulos,
                "categoria_dominante": categoria_dominante,
                "pct_dominante": pct_dominante,
                "categorias_baja_frecuencia": categorias_baja_frecuencia,
                "interpretacion": interpretacion,
                "tratamiento_sugerido": tratamiento,
                "aplica_one_hot": aplica_one_hot
            }
        )


    resumen = (
        pd.DataFrame(resumen)
        .sort_values(
            "n_categorias",
            ascending=False
        )
    )


    variables_one_hot = (
        resumen
        .query("aplica_one_hot == True")
        ["variable"]
        .tolist()
    )


    variables_revision = (
        resumen
        .query("aplica_one_hot == False")
        ["variable"]
        .tolist()
    )


    return (
        resumen,
        variables_one_hot,
        variables_revision
    )

#---- evaluar_distribucion_numericas ----
def evaluar_distribucion_numericas(
    X,
    variables_numericas,
    umbral_cv=1,
    umbral_cardinalidad=10,
    umbral_outlier=0.01
):
    """
    Evalúa variables numéricas para identificar comportamientos
    que requieren revisión en Fase I.

    No realiza tratamientos; solo genera diagnóstico.
    """

    resumen = []

    for var in variables_numericas:

        serie = X[var].dropna()

        if len(serie) == 0:
            continue

        media = serie.mean()
        std = serie.std()

        coef_variacion = (
            std / media
            if media != 0
            else np.nan
        )

        n_unicos = serie.nunique()

        p01 = serie.quantile(0.01)
        p99 = serie.quantile(0.99)

        pct_extremos = (
            (
                (serie < p01) |
                (serie > p99)
            )
            .mean()
        )

        # =========================
        # Diagnóstico Fase I
        # =========================

        if n_unicos <= umbral_cardinalidad:

            interpretacion = (
                "Baja cantidad de valores únicos."
            )

            accion = (
                "Revisar categorización"
            )

        elif coef_variacion >= umbral_cv:

            interpretacion = (
                "Alta dispersión."
            )

            accion = (
                "Revisar transformación"
            )

        elif pct_extremos >= umbral_outlier:

            interpretacion = (
                "Presencia de valores extremos."
            )

            accion = (
                "Revisar outliers"
            )

        else:

            interpretacion = (
                "Comportamiento numérico razonable."
            )

            accion = (
                "Mantener"
            )

        resumen.append(
            {
                "variable": var,
                "interpretacion": interpretacion,
                "accion_fase_I": accion,
                "n_unicos": n_unicos,
                "coef_variacion": coef_variacion,
                "pct_extremos": pct_extremos
            }
        )

    return (
        pd.DataFrame(resumen)
        [
            [
                "variable",
                "interpretacion",
                "accion_fase_I",
                "n_unicos",
                "coef_variacion",
                "pct_extremos"
            ]
        ]
    )

#---- _psi ----
def _psi(expected, actual, eps=1e-6):
    expected = np.clip(np.asarray(expected, float), eps, None)
    actual = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))

#---- calcular_psi_temporal ----
def calcular_psi_temporal(
    df,
    variables,
    columna_periodo,
    periodo_referencia,
    variables_omitir=None,
    n_bins=5,
    top_n_categorias=10,
    epsilon=1e-6
):
    """
    Calcula el PSI temporal para variables numéricas y categóricas.

    Numéricas:
        * Constantes -> se omiten.
        * <=10 valores únicos -> cada valor es un bucket.
        * >10 valores únicos -> qcut (máximo n_bins).

    Categóricas:
        * Conserva las top_n_categorias del periodo de referencia.
        * El resto se agrupa como 'OTROS'.

    Retorna
    -------
    psi_detalle, psi_resumen
    """

    detalle = []
    resumen = []
    omitidas = []
    
    ref = df[df[columna_periodo] == periodo_referencia].copy()

    if ref.empty:
        raise ValueError("Periodo de referencia no encontrado.")

    periodos = sorted(df[columna_periodo].dropna().unique())

    # Variables a excluir del análisis
    if variables_omitir is None:
        variables_omitir = []

    variables_omitir = set(variables_omitir)

    for var in variables:

        if var not in df.columns:
            omitidas.append({
                "variable": var,
                "motivo": "Variable no existe en el DataFrame"
            })
            continue

        if var in variables_omitir:
            omitidas.append({
                "variable": var,
                "motivo": "Excluida por el usuario"
            })
            continue

        sref = ref[var].dropna()

        if sref.empty:
            omitidas.append({
                "variable": var,
                "motivo": "Sin información en el período de referencia"
            })
            continue

        psi_vals = []

        # ---------------- NUMERICAS ----------------

        if pd.api.types.is_numeric_dtype(df[var]):

            nunique = sref.nunique()

            if nunique <= 1:
                omitidas.append({
                    "variable": var,
                    "motivo": "Variable constante en el período de referencia"
                })
                continue

            if nunique <= n_bins:
                categorias_ref = sref.astype(str)
                ref_dist = categorias_ref.value_counts(normalize=True).sort_index()

                for p in periodos:
                    if p == periodo_referencia:
                        continue

                    s = (
                        df.loc[df[columna_periodo] == p, var]
                        .dropna()
                        .astype(str)
                    )

                    if s.empty:
                        continue

                    actual_dist = (
                        s.value_counts(normalize=True)
                        .reindex(ref_dist.index, fill_value=0)
                    )

                    psi = _psi(ref_dist.values, actual_dist.values, epsilon)
                    psi_vals.append(psi)

                    detalle.append({
                        "variable": var,
                        "tipo": "Numerica",
                        "periodo": p,
                        "psi": psi
                    })

                n_buckets = len(ref_dist)

            else:

                _, bins = pd.qcut(
                    sref,
                    q=n_bins,
                    retbins=True,
                    duplicates="drop"
                )

                bins[0] = -np.inf
                bins[-1] = np.inf

                ref_bucket = pd.cut(
                    sref,
                    bins=bins,
                    include_lowest=True
                )

                ref_dist = (
                    ref_bucket
                    .value_counts(normalize=True)
                    .sort_index()
                )

                n_buckets = len(ref_dist)

                for p in periodos:

                    if p == periodo_referencia:
                        continue

                    s = df.loc[
                        df[columna_periodo] == p,
                        var
                    ].dropna()

                    if s.empty:
                        continue

                    actual_bucket = pd.cut(
                        s,
                        bins=bins,
                        include_lowest=True
                    )

                    actual_dist = (
                        actual_bucket
                        .value_counts(normalize=True)
                        .reindex(ref_dist.index, fill_value=0)
                    )

                    psi = _psi(ref_dist.values, actual_dist.values, epsilon)
                    psi_vals.append(psi)

                    detalle.append({
                        "variable": var,
                        "tipo": "Numerica",
                        "periodo": p,
                        "psi": psi
                    })

        # ---------------- CATEGORICAS ----------------

        else:

            top = (
                sref.astype(str)
                .value_counts(sort=True, ascending=False)
                .head(top_n_categorias)
                .index
            )

            if len(top) <= 1:
                omitidas.append({
                    "variable": var,
                    "motivo": "Una única categoría en el período de referencia"
                })
                continue
            
            ref_cat = (
                sref.astype(str)
                .where(sref.astype(str).isin(top), "OTROS")
            )

            ref_dist = (
                ref_cat
                .value_counts(normalize=True)
                .sort_index()
            )

            n_buckets = len(ref_dist)
        
            for p in periodos:

                if p == periodo_referencia:
                    continue

                s = (
                    df.loc[df[columna_periodo] == p, var]
                    .dropna()
                    .astype(str)
                )

                if s.empty:
                    continue

                actual = s.where(s.isin(top), "OTROS")

                actual_dist = (
                    actual
                    .value_counts(normalize=True)
                    .reindex(ref_dist.index, fill_value=0)
                )

                psi = _psi(ref_dist.values, actual_dist.values, epsilon)

                psi_vals.append(psi)

                detalle.append({
                    "variable": var,
                    "tipo": "Categorica",
                    "periodo": p,
                    "psi": psi
                })

        if len(psi_vals):

            psi_max = float(np.max(psi_vals))
            psi_prom = float(np.mean(psi_vals))
            psi_std = float(np.std(psi_vals))
            psi_median= float(np.median(psi_vals))

            if psi_median < 0.10:
                estado = "Estable"
            elif psi_median < 0.25:
                estado = "Cambio moderado"
            else:
                estado = "Cambio significativo"

            resumen.append({
                "variable": var,
                "tipo": "Numerica" if pd.api.types.is_numeric_dtype(df[var]) else "Categorica",
                "periodo_referencia": periodo_referencia,
                "n_buckets": n_buckets,
                "psi_max": psi_max,
                "psi_promedio": psi_prom,
                "psi_median": psi_median,
                "psi_std": psi_std,
                "estado": estado,
            })
        else:
            omitidas.append({
                "variable": var,
                "motivo": "No existen períodos con información suficiente para calcular el PSI"
            })
            continue

    psi_detalle = pd.DataFrame(detalle)

    if not psi_detalle.empty:
        psi_detalle = psi_detalle.sort_values(
            ["variable", "periodo"]
        ).reset_index(drop=True)

    psi_resumen = pd.DataFrame(resumen)

    if not psi_resumen.empty:
        psi_resumen = psi_resumen.sort_values(
            "psi_max",
            ascending=False
        ).reset_index(drop=True)
    
    psi_omitidas = pd.DataFrame(omitidas)

    if not psi_omitidas.empty:
        psi_omitidas = psi_omitidas.sort_values("variable").reset_index(drop=True)

    return psi_detalle, psi_resumen, psi_omitidas

#---- create_all_features ----

# Las transformaciones generadas en esta etapa son determinísticas y no requieren
# aprendizaje de parámetros sobre train, por lo que no introducen data leakage.
# La función asume que las variables de entrada ya fueron previamente imputadas
# y tratadas en las etapas correspondientes del pipeline.

#____ Parametros ----
VENTANAS = ['u1m', 'u3m', 'u6m', 'u12m']

def _safe_ratio(num, den, eps=1e-8):
    return num / (den.abs() + eps)

def _get_window_col(columns, base, ventana):
    """
    Busca una variable temporal con o sin separador.

    Ejemplos:
    deuda_u1m  -> deuda + u1m
    deudau1m   -> deuda + u1m
    """

    for col in [
        f"{base}_{ventana}",
        f"{base}{ventana}"
    ]:
        if col in columns:
            return col

    return None

def _detect_families(features):
    """
    Agrupa variables por familia temporal.

    Ejemplo:
    deuda_u1m, deuda_u3m, deuda_u12m
    ->
    {'deuda': {'u1m','u3m','u12m'}}
    """

    families = {}

    for col in features:
        for v in VENTANAS:
            for sufijo in [f"_{v}", v]:
                if col.endswith(sufijo):
                    base = col[:-len(sufijo)]
                    if base.endswith("_"):
                        base = base[:-1]
                    families.setdefault(
                        base,
                        set()
                    ).add(v)
                    break
            else:
                continue
            break
    return families

#____ fe_temporal_trends ----
def fe_temporal_trends(df, families):
    """
    Genera features temporales:
    - cambios entre ventanas
    - velocidades
    - aceleración
    """

    out = {}

    combos = [
        ('u1m','u12m','trend_1_12',11),
        ('u1m','u3m','trend_1_3',2),
        ('u3m','u12m','trend_3_12',9)
    ]

    for base, ventanas in families.items():

        for v1, v2, name, meses in combos:

            if {v1,v2}.issubset(ventanas):

                c1 = _get_window_col(df.columns, base, v1)
                c2 = _get_window_col(df.columns, base, v2)

                if c1 and c2:

                    if df[[c1, c2]].isna().any().any():
                        raise ValueError(
                            f"NA detectados en {c1} o {c2} antes de generar tendencias."
                        )
            
                    out[f"{base}_{name}"] = (
                        df[c2] - df[c1]
                    )

        # if {'u1m','u3m','u12m'}.issubset(ventanas):

        #     c1 = _get_window_col(df.columns, base, 'u1m')
        #     c3 = _get_window_col(df.columns, base, 'u3m')
        #     c12 = _get_window_col(df.columns, base, 'u12m')

        #     if all([c1,c3,c12]):

        #         if df[[c1, c3, c12]].isna().any().any():
        #             raise ValueError(
        #                 f"NA detectados en ventanas temporales de {base} antes de calcular velocidad."
        #             )
                
        #         vel_reciente = (
        #             df[c3] - df[c1]
        #         ) / 2

        #         vel_mediano = (
        #             df[c12] - df[c3]
        #         ) / 9

        #         out[f"{base}_vel_reciente"] = vel_reciente
        #         out[f"{base}_vel_mediano"] = vel_mediano
        #         out[f"{base}_aceleracion"] = (
        #             vel_reciente - vel_mediano # + = deteriora más rápido
        #         )

    print(
        f"  ✓ Tendencias temporales: {len(out)} features"
    )

    return out

#____ fe_volatility ----
def fe_volatility(df, families):
    """
    Calcula volatilidad entre ventanas temporales.

    La desviación estándar entre ventanas representa
    variabilidad del comportamiento financiero.
    Alta volatilidad en deuda, saldos o utilización
    puede indicar cambios relevantes en el cliente.
    """

    out = {}

    for base, ventanas in families.items():

        cols = [
            _get_window_col(
                df.columns,
                base,
                v
            )
            for v in VENTANAS
            if v in ventanas
        ]

        cols = [
            c for c in cols
            if c is not None
        ]

        if len(cols) >= 3:

            if df[cols].isna().any().any():
                raise ValueError(
                    f"NA detectados en {base} antes de calcular volatilidad."
                )

            out[f"{base}_volatilidad"] = (
                df[cols]
                .std(axis=1)
            )

    print(
        f"  ✓ Volatilidad: {len(out)} features"
    )

    return out

#____ fe_log_transforms ----
def fe_log_transforms(
    df,
    features,
    patrones_log_transformacion=None,
    umbral_no_negatividad=0.95
):
    """
    Aplica transformación log1p a variables con distribución sesgada
    y valores no negativos.

    La función asume que las variables ya fueron imputadas previamente.
    La presencia de NA indica una inconsistencia en el flujo.
    """

    if patrones_log_transformacion is None:

        patrones_log_transformacion = [
            'monto',
            'saldo',
            'deuda',
            'pasivo',
            'ingreso'
        ]

    out = {}

    for col in features:

        if any(
            p in col.lower()
            for p in patrones_log_transformacion
        ):

            if col in df.columns:

                # Validación de flujo
                if df[col].isna().any():
                    raise ValueError(
                        f"La variable {col} contiene NA antes de transformación log."
                    )

                vals = df[col]

                if (vals >= 0).mean() > umbral_no_negatividad:

                    out[f'log_{col}'] = np.log1p(
                        vals.clip(lower=0)
                    )

    print(
        f"  ✓ Log-transformaciones: {len(out)} features"
    )

    return out

#____ fe_financial_ratios ----
def fe_financial_ratios(df, features):
    """
    Genera ratios financieros interpretables:
      - DTI (Debt-to-Income): deuda_total / ingreso_mensual
    """
    out = {}
    col = lambda name: name if name in df.columns else None

    def validar_na(cols, nombre):
        if df[cols].isna().any().any():
            raise ValueError(
                f"NA detectados en {nombre} antes de calcular ratio."
            )
      
    # — DTI (deuda total / ingreso bruto mensual)
    ing_1 = col('ingresospromu1m')
    d_tot_1  = col('deudarccpromu1m')
    if d_tot_1 and ing_1:
        validar_na(
            [d_tot_1, ing_1],
            "DTI u1m"
        )
        out['dti_u1m'] = _safe_ratio(df[d_tot_1], df[ing_1])

    # — DTI (deuda total / ingreso bruto mensual)
    ing_12 = col('ingresospromu12m')
    d_tot_12  = col('deudarccpromu12m')
    if d_tot_12 and ing_12:
        validar_na(
            [d_tot_12, ing_12],
            "DTI u12m"
        )
        out['dti_u12m'] = _safe_ratio(df[d_tot_12], df[ing_12])

    # — Tendencia DTI (crecimiento de deuda vs ingreso estable)
    if d_tot_12 and ing_12 and d_tot_1 and ing_1:
            out['dti_trend'] = out['dti_u1m'] - out['dti_u12m']

    print(f"  ✓ Ratios financieros: {len(out)} features")
    return out

#____ create_all_features ----
def create_all_features(
    df,
    features,
    use_trends=True,
    use_volatility=True,
    use_log=True,
    use_ratios=True,
    umbral_no_negatividad=0.95
):
    """
    Pipeline general de Feature Engineering:
    - Tendencias temporales.
    - Volatilidad entre ventanas.
    - Log-transformaciones.
    - Ratios financieros.
    """

    print("\n" + "="*80)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*80)

    families = _detect_families(features)
    all_new = {}

    if use_trends:
        print("\n[1] Tendencias temporales")
        all_new.update(
            fe_temporal_trends(df, families)
        )

    if use_volatility:
        print("\n[2] Volatilidad")
        all_new.update(
            fe_volatility(df, families)
        )

    if use_log:
        print("\n[3] Log-transformaciones")
        all_new.update(
            fe_log_transforms(df, features, umbral_no_negatividad=umbral_no_negatividad)
        )

    if use_ratios:
        print("\n[4] Ratios financieros")
        all_new.update(fe_financial_ratios(df, features))

    result = (
        pd.DataFrame(all_new, index=df.index)
        .replace([np.inf, -np.inf], np.nan)
    )

    if result.isna().any().any():
        raise ValueError(
            "Se generaron NA en las nuevas features."
        )

    print(
        f"\n✓ Features creadas: {len(result.columns)}"
    )

    return result

#---- seleccionar_variables_pipeline ----
def seleccionar_variables_pipeline(
    X_train, y_train, X_test, y_test,
    features_num, features_dummy,
    target_type='regression',
    n_top=20,
    umbral_corr=0.70,
    vars_baneadas=None,
    vars_forzadas=None,
    vars_hardcoded=None
):
    """
    Selección de variables en 3 pasos (adaptado de LGD_instrumentalizacion_LAZLO_v2):
    1. XGBoost + SHAP  → ranking de importancia
    2. Correlación de Pearson → eliminar redundantes (solo numéricas)
    3. Permutation Importance → confirmar aporte real

    Parameters
    ----------
    target_type    : 'regression' | 'classification'
    vars_hardcoded : list | None
        Si se especifica, OMITE los 3 pasos y devuelve directamente estas
        variables (separadas en numéricas y dummies según features_num/dummy).
        Útil cuando el set ya fue validado y no se quiere re-correr el pipeline.
    """
    if vars_baneadas is None:
        vars_baneadas = []
    if vars_forzadas is None:
        vars_forzadas = []

    # ── Modo hardcoded: devolver lista fija sin correr el pipeline ────────────
    if vars_hardcoded is not None:
        sel_num   = [v for v in vars_hardcoded if v in features_num]
        sel_dummy = [v for v in vars_hardcoded if v in features_dummy]
        print(f'[vars_hardcoded] Usando lista fija — {len(vars_hardcoded)} variables')
        print(f'  Numéricas  ({len(sel_num)}): {sel_num}')
        print(f'  Dummies    ({len(sel_dummy)}): {sel_dummy}')
        return sel_num, sel_dummy

    # ── Retiro de variables baneadas antes de los modelos ────────────
    features_num = [v for v in features_num if v not in vars_baneadas]
    features_dummy = [v for v in features_dummy if not any(b in v for b in vars_baneadas)]

    all_features = features_num + features_dummy

    def _to_float(df, cols):
        """
        Extrae columnas, convierte a numérico y retorna float.
        Usa iloc para la iteración y evita el fallo cuando hay
        nombres de columna duplicados en el DataFrame fuente.
        """
        out = df[cols].copy()
        out.columns = list(range(out.shape[1]))          # índices temporales únicos
        for i in out.columns:
            out[i] = pd.to_numeric(out[i], errors='coerce')
        out.columns = cols                                # restaurar nombres originales
        return out.fillna(0).astype(float).reset_index(drop=True)

    Xtr = _to_float(X_train, all_features)
    Xte = _to_float(X_test,  all_features)
    ytr = y_train.reset_index(drop=True)
    yte = y_test.reset_index(drop=True)

    # ── Paso 1: XGBoost + SHAP ────────────────────────────────────────────────
    if target_type == 'classification':
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1)
        model_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            scale_pos_weight=spw, eval_metric='aucpr', verbosity=0
        )
        scoring_perm = 'roc_auc'
    else:
        model_xgb = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        scoring_perm = 'r2'

    model_xgb.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    explainer = shap.TreeExplainer(model_xgb)
    shap_vals = explainer.shap_values(Xtr)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    df_shap = (
        pd.DataFrame({'feature': all_features,
                      'shap_mean': np.abs(shap_vals).mean(axis=0)})
        .sort_values('shap_mean', ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    top_plot = df_shap.head(20)
    ax.barh(top_plot['feature'][::-1], top_plot['shap_mean'][::-1], color='#2E75B6')
    ax.set_title(f'Top 20 — |SHAP| medio (XGBoost {target_type})')
    ax.set_xlabel('|SHAP| medio')
    plt.tight_layout()
    plt.show()

    # ── Paso 2: Correlación (solo numéricas) ──────────────────────────────────
    vars_eliminar = set()
    if len(features_num) > 1:
        corr_matrix = Xtr[features_num].corr().abs()
        corr_pairs = (
            corr_matrix
            .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack().reset_index()
            .rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'corr'})
            .query('corr >= @umbral_corr')
            .sort_values('corr', ascending=False)
        )

        # ── Heatmap top SHAP ──────────────────────────────────────────────────
        top_num = [v for v in df_shap.head(15)['feature'].tolist() if v in features_num]
        if top_num:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(Xtr[top_num].corr(), annot=True, fmt='.2f',
                        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                        ax=ax, linewidths=0.5)
            ax.set_title(f'Correlaciones — Top SHAP (umbral={umbral_corr})')
            plt.tight_layout()
            plt.show()
        # ─────────────────────────────────────────────────────────────────────
        for _, row in corr_pairs.iterrows():
            v1, v2 = row['var1'], row['var2']
            r1 = df_shap[df_shap['feature'] == v1].index[0] if v1 in df_shap['feature'].values else 999
            r2 = df_shap[df_shap['feature'] == v2].index[0] if v2 in df_shap['feature'].values else 999
            vars_eliminar.add(v2 if r1 <= r2 else v1)
        print(f'Pares correlación >= {umbral_corr}: {len(corr_pairs)}')
        if not corr_pairs.empty:
            print(corr_pairs.to_string(index=False))
        print(f'A eliminar por correlación: {sorted(vars_eliminar)}')
        

    # ── Paso 3: Permutation Importance ────────────────────────────────────────
    perm_result = permutation_importance(
        model_xgb, Xte, yte, n_repeats=10, random_state=42, scoring=scoring_perm
    )
    df_perm = (
        pd.DataFrame({'feature': all_features,
                      'perm_mean': perm_result.importances_mean,
                      'perm_std': perm_result.importances_std})
        .sort_values('perm_mean', ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    top_perm = df_perm.head(20)
    ax.barh(top_perm['feature'][::-1], top_perm['perm_mean'][::-1],
            xerr=top_perm['perm_std'][::-1], color='#375623', alpha=0.8)
    ax.set_xlabel(f'Caída en {scoring_perm} al permutar')
    ax.set_title('Permutation Importance (test)')
    plt.tight_layout()
    plt.show()

    vars_sin_aporte = df_perm[df_perm['perm_mean'] <= 0]['feature'].tolist()
    print(f'Sin aporte (perm_mean <= 0): {vars_sin_aporte}')

    # ── Selección final ────────────────────────────────────────────────────────
    candidatas = df_shap.head(n_top)['feature'].tolist()

    sel_num = list(dict.fromkeys(
        [v for v in candidatas
         if v in features_num
         and v not in vars_eliminar
         and v not in vars_sin_aporte]
        + [v for v in vars_forzadas if v in features_num]
    ))
    sel_dummy = [
        v for v in candidatas
        if v in features_dummy
        and v not in vars_eliminar
        and v not in vars_sin_aporte
        + [v for v in vars_forzadas if v in features_dummy]
    ]

    print(f'\nVariables seleccionadas: {len(sel_num + sel_dummy)}')
    print(f'  Numéricas  ({len(sel_num)}): {sel_num}')
    print(f'  Dummies    ({len(sel_dummy)}): {sel_dummy}')

    return sel_num, sel_dummy

#---- encontrar_knee_optimo ----
def encontrar_knee_optimo(k_values, inercias):
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    i_norm = (inercias - inercias.min()) / (inercias.max() - inercias.min())
    p1 = np.array([k_norm[0], i_norm[0]])
    pn = np.array([k_norm[-1], i_norm[-1]])
    distancias = [
        np.abs(np.cross(pn - p1, p1 - np.array([k_norm[i], i_norm[i]]))) / np.linalg.norm(pn - p1)
        for i in range(len(k_norm))
    ]
    return k_values[np.argmax(distancias)], distancias

#---- graficar_knee_metodo_distancia ----
def graficar_knee_metodo_distancia(k_values, inercias, k_optimo, distancias):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(k_values, inercias, 'bo-', label='Curva de Prepagos (PCT Acum)')
    ax1.plot([k_values[0], k_values[-1]], [inercias[0], inercias[-1]], 'r--', label='Recta Secante')
    pct_opt = inercias[np.where(k_values == k_optimo)[0][0]]
    ax1.scatter(k_optimo, pct_opt, color='green', s=100, zorder=5, label=f'K Óptimo: {k_optimo}')
    ax1.set_title('Sistema Original')
    ax1.set_xlabel('Meses de Maduración')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(k_values, distancias, 'g^-', label='Distancia Perpendicular')
    ax2.fill_between(k_values, distancias, color='green', alpha=0.1)
    ax2.axvline(x=k_optimo, color='red', linestyle=':')
    ax2.set_title('Máximo de Curvatura')
    ax2.set_xlabel('Meses de Maduración')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#---- generar_resumen_volumetria ----
def generar_resumen_volumetria(
    df,
    variable_segmento="segmento_arbol_new",
    var_ponderacion="var_ponderacion",
    variable_id="id_cliente"
):
    resumen = (
        df
        .groupby(variable_segmento)
        .agg(
            monto=(var_ponderacion, "sum"),
            cantidad=(variable_id, "size")
        )
        .assign(
            participacion_monto_pct=lambda df: (
                df["monto"] / df["monto"].sum()
            ),
            participacion_cantidad_pct=lambda df: (
                df["cantidad"] / df["cantidad"].sum()
            ),
        )
        .round(2)
        .reset_index()
        .sort_values(variable_segmento, ascending=True)
    )

    return resumen

#---- resumen_segmento_completo ----
def resumen_segmento_completo(
    df,
    index,
    columns,
    metrica,
    peso,
    periodo,
    nombre_resultado="metrica_ponderada",
):
    """Calcula un DataFrame plano con métricas agregadas, ponderadas, dispersión,

    promedios mensuales y participaciones (globales y mensuales) por cada
    combinación (index, columns). Todas las métricas de tasa/porcentaje en
    escala decimal.
    """
    df_temp = df.copy()

    # 0. Total de meses DIFERENTES en todo el universo de datos
    # Captura los 8 meses completos del estudio, independientemente de cada bucket
    total_meses_global = df_temp[periodo].nunique()

    # 1. Reescalado inicial a decimales (si la métrica viene en base 100)
    if df_temp[metrica].max() > 1:
        df_temp[metrica] = df_temp[metrica] / 100.0

    # 2. Agregación agrupada por combinación
    def calcular_metricas_grupo(x):
        # Media y desviación estándar ponderada dentro del grupo
        m_pond = np.average(x[metrica], weights=x[peso])
        var_pond = np.average((x[metrica] - m_pond) ** 2, weights=x[peso])
        std_pond = np.sqrt(var_pond)

        # Coeficiente de variación (en decimal)
        cv = std_pond / m_pond if m_pond != 0 else np.nan

        monto_total = x[peso].sum()
        n_ops = len(x)
        meses_activos_bucket = x[periodo].nunique()

        return pd.Series(
            {
                nombre_resultado: m_pond,
                "cv": cv,
                "monto": monto_total,
                # Divide siempre entre el total de meses diferentes globales (ej. 8)
                "monto_mensual": (
                    monto_total / total_meses_global
                    if total_meses_global > 0
                    else np.nan
                ),
                "n_operaciones": n_ops,
                "n_operaciones_mensuales": (
                    n_ops / total_meses_global
                    if total_meses_global > 0
                    else np.nan
                ),
                "n_meses_presente": meses_activos_bucket,
                "total_meses_estudio": total_meses_global,
            }
        )

    resumen = (
        df_temp.groupby([index, columns], observed=True)
        .apply(calcular_metricas_grupo)
        .reset_index()
    )

    # 3. Participaciones Globales
    monto_global = resumen["monto"].sum()
    n_global = resumen["n_operaciones"].sum()

    resumen["part_monto_global"] = (
        resumen["monto"] / monto_global if monto_global > 0 else 0.0
    )
    resumen["part_n_operaciones_global"] = (
        resumen["n_operaciones"] / n_global if n_global > 0 else 0.0
    )

    # 4. Participaciones Mensuales
    monto_mensual_total = resumen["monto_mensual"].sum()
    n_mensual_total = resumen["n_operaciones_mensuales"].sum()

    resumen["part_monto_mensual"] = (
        resumen["monto_mensual"] / monto_mensual_total
        if monto_mensual_total > 0
        else 0.0
    )
    resumen["part_n_operaciones_mensual"] = (
        resumen["n_operaciones_mensuales"] / n_mensual_total
        if n_mensual_total > 0
        else 0.0
    )

    # 5. Ordenamiento final
    resumen = resumen.sort_values([index, columns]).reset_index(drop=True)

    return resumen

#---- exportar_reporte_multicortes_fijo_segmento ----
def exportar_reporte_multicortes_fijo_segmento(
    df,
    dict_cortes_es_sentido_inverso,  # {variable: True (Sí es Inverso) / False (No es Inverso -> Directo)}
    metrica,
    peso,
    periodo,
    ruta_excel,
    col_segmento="segmento_arbol_new",
    nombre_resultado="tasa_ponderada",
    formato_metrica="0.00%",
    flg_heatmap_cv=True,
    cv_threshold_verde=0.15,
    cv_threshold_rojo=0.30,
    # --- Parámetros Escalera de Precios ---
    flg_heatmap_metrica=True,
    umbral_dif_contigua=0.0005,  # 5 bps (LOCAL)
    umbral_dif_global_abs=0.0050,  # 50 bps (GLOBAL)
    umbral_var_extremos=0.20,  # 20% variación relativa (GLOBAL)
    flg_heatmap_participaciones=True,
    # --- Parámetros Colores (HEX) ---
    hex_rojo="F4C7C3",
    hex_amarillo="FFF2CC",
    hex_verde="D9EAD3",
):
    # =========================================================================
    # PALETA DE COLORES Y ESTILOS
    # =========================================================================
    fill_rojo_global = PatternFill(
        start_color=hex_rojo, end_color=hex_rojo, fill_type="solid"
    )
    fill_amarillo_global = PatternFill(
        start_color=hex_amarillo, end_color=hex_amarillo, fill_type="solid"
    )

    font_rojo_global = Font(name="Calibri", size=10, color="78281F")
    font_amarillo_global = Font(name="Calibri", size=10, color="7D6608")

    fill_header = PatternFill(
        start_color="008080", end_color="008080", fill_type="solid"
    )
    fill_segmento = PatternFill(
        start_color="F3F6F5", end_color="F3F6F5", fill_type="solid"
    )

    font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    font_segmento = Font(name="Calibri", size=10, bold=True, color="000000")
    font_titulo = Font(name="Calibri", size=11, bold=True, color="008080")
    font_data = Font(name="Calibri", size=10, color="000000")

    align_center = Alignment(horizontal="center", vertical="center")
    border_normal = Side(border_style="thin", color="000000")
    border_grueso = Side(border_style="medium", color="000000")

    # =========================================================================
    # REGLAS CONDICIONALES
    # =========================================================================
    rule_cv = ColorScaleRule(
        start_type="num",
        start_value=cv_threshold_verde,
        start_color=hex_verde,
        mid_type="num",
        mid_value=(cv_threshold_verde + cv_threshold_rojo) / 2,
        mid_color=hex_amarillo,
        end_type="num",
        end_value=cv_threshold_rojo,
        end_color=hex_rojo,
    )

    rule_part = ColorScaleRule(
        start_type="min",
        start_color=hex_rojo,
        mid_type="percentile",
        mid_value=50,
        mid_color=hex_amarillo,
        end_type="max",
        end_color=hex_verde,
    )

    def evaluar_escalera_fila(
        valores_fila, 
        es_sentido_inverso, 
        umbral_var_extremos=0.20,       
        umbral_dif_global_abs=None,     
        umbral_dif_contigua=0.0         
    ):
        """
        Evalúa una fila de una escalera de tasas/riesgos aplicando un sistema de 3 jerarquías:
        
        PASO A PASO DE LA LÓGICA DE EVALUACIÓN:
        ---------------------------------------------------------------------------
        1. Limpieza y Preparación:
        - Convierte los datos a tipo float de manera segura y aísla los valores nulos (NaN) 
            para evitar que actúen como ceros (0.0).
        - Valida que existan al menos 2 valores válidos en la fila; de lo contrario, 
            retorna la fila sin evaluar.

        2. Jerarquía 1: Inconsistencia Continua (🔴 Rojo Individual por Quiebre)
        - Recorre la fila comparando parejas contiguas (valor actual vs. valor siguiente).
        - Si una transición rompe el sentido esperado de la escalera (definido por 
            'es_sentido_inverso'), marca estrictamente con ROJO a la celda de la derecha.
        - Esta es la regla principal: el ROJO tiene la máxima prioridad y aísla el error 
            en el punto exacto donde ocurre el quiebre.

        3. Jerarquía 2: Fila Plana Global (🟡 Amarillo en Toda la Fila)
        - Se ejecuta únicamente si la fila está completamente limpia (es decir, NO hay ROJOS).
        - Calcula la variación relativa y absoluta entre el valor mínimo y máximo de la fila.
        - Si los cambios globales cumplen con los umbrales permitidos, pinta de AMARILLO 
            a todas las celdas válidas de la fila.

        4. Jerarquía 3: Paso a Paso Plano (🟡 Amarillo Local con Blindaje)
        - Se activa si la fila no fue globalmente plana. Evalúa las parejas contiguas 
            buscando tramos estables paso a paso según el 'umbral_dif_contigua'.
        - Protección Anti-Sándwich y Límites: Antes de aplicar el amarillo local, valida 
            con protección de índices en los extremos que la celda no esté atrapada (en sándwich) 
            entre dos celdas vecinas que ya sean ROJAS. Si está aislada por errores, bloquea el 
            amarillo; de lo contrario, asigna AMARILLO de forma segura.
        ---------------------------------------------------------------------------
        """
        n = len(valores_fila)
        colores = [None] * n

        # 1. Limpieza segura: Convertir a float preservando los NaN reales (sin volverlos 0.0)
        valores_limpios = []
        for v in valores_fila:
            try:
                valores_limpios.append(float(v) if pd.notna(v) else None)
            except Exception:
                valores_limpios.append(None)

        # 2. Si hay menos de 2 valores válidos en toda la fila, no se puede evaluar nada
        vals_validos = [v for v in valores_limpios if v is not None]
        if len(vals_validos) < 2:
            return colores

        # =========================================================================
        # 3. Jerarquía 1: Inconsistencia Contínua (🔴 Rojo Individual por Quiebre)
        # =========================================================================
        for i in range(n - 1):
            v_actual = valores_limpios[i]
            v_siguiente = valores_limpios[i + 1]
            
            # Si alguno de los dos es un NaN, no hay transición que evaluar
            if v_actual is None or v_siguiente is None:
                continue
                
            delta = v_siguiente - v_actual
            if (es_sentido_inverso and delta > 0) or (not es_sentido_inverso and delta < 0):
                colores[i + 1] = "ROJO"

        # =========================================================================
        # 4. Jerarquía 2: Fila Plana Global (🟡 Amarillo Toda la Fila)
        # =========================================================================
        if "ROJO" not in colores:
            val_min = min(vals_validos)
            val_max = max(vals_validos)
            dif_absoluta_global = val_max - val_min
            var_relativa = (dif_absoluta_global / val_min) if val_min > 0 else float("inf")

            cumple_relativa = var_relativa < umbral_var_extremos
            
            if umbral_dif_global_abs is not None:
                cumple_absoluta = dif_absoluta_global <= umbral_dif_global_abs
            else:
                cumple_absoluta = True

            if cumple_relativa and cumple_absoluta:
                # Pinta de amarillo solo las posiciones que tienen datos válidos
                return ["AMARILLO" if v is not None else None for v in valores_limpios]

        # =========================================================================
        # 5. Jerarquía 3: Paso a Paso Plano (🟡 Amarillo Local con Validación de Sentido)
        # =========================================================================
        for i in range(n - 1):
            v_actual = valores_limpios[i]
            v_siguiente = valores_limpios[i + 1]
            
            if v_actual is None or v_siguiente is None:
                continue
                
            delta = v_siguiente - v_actual
            
            respeta_sentido = True
            if es_sentido_inverso and delta > 0:
                respeta_sentido = False
            elif not es_sentido_inverso and delta < 0:
                respeta_sentido = False

            if respeta_sentido and abs(delta) <= umbral_dif_contigua:
                
                # Validación segura de sándwich con protección estricta de límites (extremos)
                sandwich_i = False
                if i > 0 and (i + 1) < n:
                    if colores[i - 1] == "ROJO" and colores[i + 1] == "ROJO":
                        sandwich_i = True
                
                sandwich_siguiente = False
                if i > 0 and (i + 2) < n:
                    if colores[i] == "ROJO" and colores[i + 2] == "ROJO":
                        sandwich_siguiente = True

                # Aplicar color respetando la jerarquía del ROJO y evitando sándwiches
                if colores[i] != "ROJO" and not sandwich_i:
                    colores[i] = "AMARILLO"
                    
                if colores[i + 1] != "ROJO" and not sandwich_siguiente:
                    colores[i + 1] = "AMARILLO"

        return colores

    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        for (
            var_corte,
            es_inverso_corte,
        ) in dict_cortes_es_sentido_inverso.items():
            sheet_name = str(var_corte)[:30]

            resumen_corte = resumen_segmento_completo(
                df=df,
                index=col_segmento,
                columns=var_corte,
                metrica=metrica,
                peso=peso,
                periodo=periodo,
                nombre_resultado=nombre_resultado,
            )

            vistas = {
                "metrica_principal": resumen_corte.pivot(
                    index=col_segmento,
                    columns=var_corte,
                    values=nombre_resultado,
                ),
                "cv": resumen_corte.pivot(
                    index=col_segmento, columns=var_corte, values="cv"
                ),
                "part_monto": resumen_corte.pivot(
                    index=col_segmento,
                    columns=var_corte,
                    values="part_monto_mensual",
                ),
                "part_ops": resumen_corte.pivot(
                    index=col_segmento,
                    columns=var_corte,
                    values="part_n_operaciones_mensual",
                ),
                "monto_mm": resumen_corte.pivot(
                    index=col_segmento,
                    columns=var_corte,
                    values="monto_mensual",
                )
                / 1e6,
                "ops": resumen_corte.pivot(
                    index=col_segmento,
                    columns=var_corte,
                    values="n_operaciones_mensuales",
                ),
            }

            for k in vistas:
                vistas[k].index.name = "Segmento"
                vistas[k].columns.name = None

            cols_cat = vistas["metrica_principal"].shape[1]
            c_izq, c_der = 0, 1 + cols_cat + 2
            titulo_metrica = (
                f"1. {nombre_resultado.replace('_', ' ').title()}"
            )

            bloques = [
                (
                    titulo_metrica,
                    vistas["metrica_principal"],
                    c_izq,
                    formato_metrica,
                    None,
                    flg_heatmap_metrica,
                    "2. Coeficiente de Variación",
                    vistas["cv"],
                    c_der,
                    "0.00",
                    rule_cv,
                    flg_heatmap_cv,
                ),
                (
                    "% Participación Monto Mensual",
                    vistas["part_monto"],
                    c_izq,
                    "0.00%",
                    rule_part,
                    flg_heatmap_participaciones,
                    "% Participación Registros Mensual",
                    vistas["part_ops"],
                    c_der,
                    "0.00%",
                    rule_part,
                    flg_heatmap_participaciones,
                ),
                (
                    "Monto Mensual (MM S/.)",
                    vistas["monto_mm"],
                    c_izq,
                    "#,##0.00",
                    None,
                    False,
                    "Registros (Ops) Mensuales",
                    vistas["ops"],
                    c_der,
                    "#,##0",
                    None,
                    False,
                ),
            ]

            row_cursor = 0

            for (
                tit_izq,
                df_izq,
                col_i,
                fmt_izq,
                rule_i,
                flag_i,
                tit_der,
                df_der,
                col_d,
                fmt_der,
                rule_d,
                flag_d,
            ) in bloques:
                pd.DataFrame([[tit_izq]]).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=row_cursor,
                    startcol=col_i,
                    header=False,
                    index=False,
                )
                pd.DataFrame([[tit_der]]).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=row_cursor,
                    startcol=col_d,
                    header=False,
                    index=False,
                )

                df_izq.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=row_cursor + 1,
                    startcol=col_i,
                )
                df_der.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=row_cursor + 1,
                    startcol=col_d,
                )

                ws = writer.sheets[sheet_name]

                def dar_estilo_bloque(
                    start_row,
                    start_col,
                    df_data,
                    num_fmt,
                    rule_conditional,
                    aplicar_heatmap,
                    es_metrica_principal=False,
                ):
                    ws.cell(
                        row=start_row + 1, column=start_col + 1
                    ).font = font_titulo
                    n_filas, n_cols = df_data.shape

                    for r_rel in range(n_filas + 1):
                        curr_row = start_row + 2 + r_rel
                        colores_escalera = []
                        if (
                            es_metrica_principal
                            and aplicar_heatmap
                            and r_rel > 0
                        ):
                            valores_fila = df_data.iloc[r_rel - 1].values
                            colores_escalera = evaluar_escalera_fila(
                                valores_fila, es_inverso_corte, umbral_dif_contigua=umbral_dif_contigua,
                                umbral_var_extremos=umbral_var_extremos, umbral_dif_global_abs=umbral_dif_global_abs
                            )

                        for c_rel in range(n_cols + 1):
                            curr_col = start_col + 1 + c_rel
                            cell = ws.cell(row=curr_row, column=curr_col)

                            cell.border = Border(
                                top=(
                                    border_grueso
                                    if r_rel == 0
                                    else border_normal
                                ),
                                bottom=(
                                    border_grueso
                                    if r_rel == n_filas
                                    else border_normal
                                ),
                                left=(
                                    border_grueso
                                    if c_rel == 0
                                    else border_normal
                                ),
                                right=(
                                    border_grueso
                                    if c_rel == n_cols
                                    else border_normal
                                ),
                            )
                            cell.alignment = align_center

                            if r_rel == 0:
                                cell.fill, cell.font = fill_header, font_header
                            elif c_rel == 0:
                                cell.fill, cell.font = (
                                    fill_segmento,
                                    font_segmento,
                                )
                            else:
                                cell.font = font_data
                                if cell.value is not None:
                                    cell.number_format = num_fmt

                                if (
                                    es_metrica_principal
                                    and aplicar_heatmap
                                    and c_rel > 0
                                ):
                                    color_asignado = colores_escalera[c_rel - 1]
                                    if color_asignado == "ROJO":
                                        cell.fill, cell.font = (
                                            fill_rojo_global,
                                            font_rojo_global,
                                        )
                                    elif color_asignado == "AMARILLO":
                                        cell.fill, cell.font = (
                                            fill_amarillo_global,
                                            font_amarillo_global,
                                        )

                    if (
                        not es_metrica_principal
                        and aplicar_heatmap
                        and rule_conditional
                        and n_filas > 0
                    ):
                        rango_excel = f"{get_column_letter(start_col + 2)}{start_row + 3}:{get_column_letter(start_col + 1 + n_cols)}{start_row + 2 + n_filas}"
                        ws.conditional_formatting.add(
                            rango_excel, rule_conditional
                        )

                dar_estilo_bloque(
                    row_cursor,
                    col_i,
                    df_izq,
                    fmt_izq,
                    rule_i,
                    flag_i,
                    tit_izq == titulo_metrica,
                )
                dar_estilo_bloque(
                    row_cursor,
                    col_d,
                    df_der,
                    fmt_der,
                    rule_d,
                    flag_d,
                    False,
                )

                row_cursor += max(len(df_izq), len(df_der)) + 5



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_barras_comparacion_base(df, segment_col='segmento_arbol_new', target_col='target_precancelacion', base_col='base'):
    """
    Genera un gráfico de barras agrupadas comparando las bases (Train, Test, OOT) 
    por segmento, mostrando la tasa en porcentaje (%) con etiquetas de valor exacto.
    """
    df_plot = df.copy()
    
    # 1. Agrupar, calcular la tasa promedio y multiplicarla por 100 para llevarla a porcentaje
    df_agg = df_plot.groupby([segment_col, base_col])[target_col].mean().reset_index()
    df_agg[target_col + '_pct'] = df_agg[target_col] * 100
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    palette = sns.color_palette("muted")
    
    sns.barplot(
        data=df_agg, 
        x=segment_col, 
        y=target_col + '_pct', 
        hue=base_col,
        palette=palette,
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    
    # 2. Añadir tags de valor en porcentaje automáticamente sobre cada barra
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.2f%%',          # Muestra 2 decimales y el símbolo de porcentaje (ej. 15.42%)
            padding=3,               
            fontsize=8, 
            fontweight='bold',
            color='#333333'
        )
        
    # 3. Estética general y formato ejecutivo
    ax.set_title('Tasa de Conversión (%) por Segmento y Base (Train, Test, OOT)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Segmentos del Árbol', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tasa Promedio de Conversión (%)', fontsize=11, fontweight='bold')
    
    # Dar un respiro superior en el eje Y para que los textos no se corten
    y_max = df_agg[target_col + '_pct'].max()
    ax.set_ylim(0, y_max * 1.20)
    
    # Ejes, leyenda y grilla
    plt.xticks(rotation=0, fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(title='Base', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()





import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_evolucion_segmentos_grid(df, period_col='periodo_x', segment_col='segmento_arbol_new', target_col='target_precancelacion', base_col='base', ncols=3):
    df_plot = df.copy()
    
    # 1. Forzar el periodo a numérico para garantizar un ordenamiento cronológico real (ej. 202209, 202210...)
    df_plot['period_num'] = pd.to_numeric(df_plot[period_col])
    
    meses_dict = {
        '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr',
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago',
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
    }
    
    def formatear_periodo(p):
        p_str = str(int(p)) if pd.notnull(p) else ''
        if len(p_str) == 6:
            yy = p_str[2:4]
            mm = p_str[4:6]
            return f"{meses_dict.get(mm, mm)} '{yy}"
        return p_str

    df_plot['period_label'] = df_plot['period_num'].apply(formatear_periodo)
    
    # 2. Obtener lista global de periodos ordenados estrictamente por su valor numérico
    all_periods_sorted = sorted(df_plot['period_num'].unique())
    label_mapping = df_plot[['period_num', 'period_label']].drop_duplicates().set_index('period_num')['period_label'].to_dict()

    # 3. Agrupar usando el periodo numérico para mantener la secuencia correcta
    df_agg = (
        df_plot
        .groupby(['period_num', segment_col, base_col])[target_col]
        .mean()
        .reset_index()
    )
    
    # Multiplicar por 100 para visualizar en porcentaje (%)
    df_agg[target_col] = df_agg[target_col] * 100
    
    segmentos = sorted(df_agg[segment_col].unique())
    n_segs = len(segmentos)
    nrows = math.ceil(n_segs / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5.5, nrows * 4), sharex=True, sharey=True)
    axes = axes.flatten() if n_segs > 1 else [axes]
    
    # Ordenar las bases a la inversa (sorted con reverse=True) para controlar la jerarquía visual y de leyenda
    bases = sorted(df_agg[base_col].unique(), reverse=True)
    palette = sns.color_palette("Set1", len(bases))
    base_color_map = dict(zip(bases, palette))
    
    for i, seg in enumerate(segmentos):
        ax = axes[i]
        sub_seg = df_agg[df_agg[segment_col] == seg]
        
        for base in bases:
            # Reindexar usando la lista numérica ordenada para mantener continuidad temporal
            sub_base = sub_seg[sub_seg[base_col] == base].set_index('period_num').reindex(all_periods_sorted).reset_index()
            sub_base['period_label'] = sub_base['period_num'].map(label_mapping)
            
            sub_base_clean = sub_base.dropna(subset=[target_col])
            
            if not sub_base_clean.empty:
                ax.plot(
                    sub_base_clean['period_label'], 
                    sub_base_clean[target_col], 
                    marker='o', 
                    linewidth=2, 
                    markersize=4,
                    color=base_color_map[base],
                    label=base
                )
        
        # Storytelling visual por segmento
        ax.set_title(f'Segmento: {seg}', fontsize=11, fontweight='bold', color='#1f77b4')
        
        # Rotación vertical (90 grados) de los meses para evitar solapamientos
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(title='Submuestra (Base)', fontsize=8, title_fontsize=9, loc='upper left')
            
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Storytelling gerencial en el título principal
    plt.suptitle(
        'Validación de Estabilidad Temporal: Tasa de Conversión (%) por Segmento y Partición (Train / Test / OOT)', 
        fontsize=13, 
        fontweight='bold', 
        y=1.03
    )
    plt.tight_layout()
    plt.show()

def calcular_psi_temporal_por_base(df, period_col='periodo_x', segment_col='segmento_arbol_new', base_col='base', base_ref_nombre='train'):
    # 1. Identificar el último mes de la base 'train' como referencia fija
    train_df = df[df[base_col] == base_ref_nombre]
    ultimo_periodo_train = train_df[period_col].max()
    
    print(f"-> Periodo de referencia fijo (Último mes de Train): {ultimo_periodo_train}\n")
    
    # 2. Obtener la distribución de referencia (proporciones de los segmentos en ese mes base)
    ref_subset = train_df[train_df[period_col] == ultimo_periodo_train]
    ref_counts = ref_subset[segment_col].value_counts(normalize=True).sort_index()
    
    registros = []
    
    # 3. Calcular el PSI mes a mes separando por cada base
    for periodo in sorted(df[period_col].unique()):
        sub_periodo = df[df[period_col] == periodo]
        if sub_periodo.empty:
            continue
            
        for b in sub_periodo[base_col].unique():
            sub_base = sub_periodo[sub_periodo[base_col] == b]
            comp_counts = sub_base[segment_col].value_counts(normalize=True).sort_index()
            
            # Unir distribuciones y corregir ceros con epsilon
            tabla = pd.DataFrame({
                'ref': ref_counts,
                'comp': comp_counts
            }).fillna(0.0001)
            
            tabla['ref'] = tabla['ref'] / tabla['ref'].sum()
            tabla['comp'] = tabla['comp'] / tabla['comp'].sum()
            
            # Calcular PSI parcial y total
            tabla['psi_parcial'] = (tabla['comp'] - tabla['ref']) * np.log(tabla['comp'] / tabla['ref'])
            psi_val = tabla['psi_parcial'].sum()
            
            registros.append({
                'period': periodo,
                'base': b,
                'PSI': psi_val
            })
            
    df_psi = pd.DataFrame(registros)
    
    # 4. Obtener estadísticos resumen (Promedio, Mediana, Máximo) agrupados por base
    stats_por_base = df_psi.groupby('base')['PSI'].agg(['mean', 'median', 'max']).rename(
        columns={'mean': 'Promedio', 'median': 'Mediana', 'max': 'Máximo'}
    )
    
    return df_psi, stats_por_base


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_materialidad_global_temporal(df, period_col='periodo_x', segment_col='segmento_arbol_new'):
    """
    Genera un gráfico de barras apiladas al 100% con un enfoque ejecutivo, 
    rotación vertical de periodos y storytelling de estabilidad poblacional.
    """
    df_plot = df.copy()
    
    # 1. Asegurar ordenamiento numérico y cronológico del periodo
    df_plot['period_num'] = pd.to_numeric(df_plot[period_col])
    
    meses_dict = {
        '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr',
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago',
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
    }
    
    def formatear_periodo(p):
        p_str = str(int(p)) if pd.notnull(p) else ''
        if len(p_str) == 6:
            return f"{meses_dict.get(p_str[4:6], p_str[4:6])} '{p_str[2:4]}"
        return p_str

    df_plot['period_label'] = df_plot['period_num'].apply(formatear_periodo)
    
    # 2. Agrupar por periodo numérico y segmento para obtener conteos absolutos
    df_counts = df_plot.groupby(['period_num', 'period_label', segment_col]).size().unstack(fill_value=0)
    
    # Ordenar estrictamente por el índice numérico temporal
    df_counts = df_counts.sort_index(level='period_num')
    
    # Extraer las etiquetas ordenadas y asignarlas al índice
    labels_ordenadas = df_counts.index.get_level_values('period_label')
    df_counts.index = labels_ordenadas
    
    # 3. Convertir a proporciones (100% stacked)
    df_prop = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
    
    # 4. Configurar el gráfico con diseño ejecutivo
    fig, ax = plt.subplots(figsize=(14, 6))
    
    palette = sns.color_palette("Set2", len(df_prop.columns))
    
    df_prop.plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        color=palette, 
        edgecolor='black', 
        linewidth=0.3
    )
    
    # Storytelling gerencial y etiquetas claras
    ax.set_title('Evolución de la Estabilidad Poblacional: Materialidad (% de Cartera en Conteo) por Segmento', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Periodo de Originación', fontsize=11, fontweight='bold')
    ax.set_ylabel('Participación Relativa en la Cartera (%)', fontsize=11, fontweight='bold')
    
    # Leyenda limpia al costado
    ax.legend(title='Segmentos del Árbol', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize=10)
    
    # Rotación vertical (90°) de los meses para evitar solapamientos y lectura limpia
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_todas_series_juntas(df, period_col='periodo_x', segment_col='segmento_arbol_new', target_col='target_precancelacion'):
    """
    Genera un gráfico de líneas unificado para visualizar la evolución y separación 
    de la tasa de precancelación (%) por segmento en el tiempo, con formato ejecutivo.
    """
    df_plot = df.copy()
    
    # 1. Forzar el periodo a numérico para asegurar orden cronológico perfecto
    df_plot['period_num'] = pd.to_numeric(df_plot[period_col])
    
    meses_dict = {
        '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr',
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago',
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
    }
    
    def formatear_periodo(p):
        p_str = str(int(p)) if pd.notnull(p) else ''
        if len(p_str) == 6:
            yy = p_str[2:4]
            mm = p_str[4:6]
            return f"{meses_dict.get(mm, mm)} '{yy}"
        return p_str

    df_plot['period_label'] = df_plot['period_num'].apply(formatear_periodo)
    
    # Obtener lista global de periodos ordenados
    all_periods_sorted = sorted(df_plot['period_num'].unique())
    label_mapping = df_plot[['period_num', 'period_label']].drop_duplicates().set_index('period_num')['period_label'].to_dict()
    
    # 2. Agrupar para calcular la tasa media por periodo y segmento
    df_agg = (
        df_plot
        .groupby(['period_num', segment_col])[target_col]
        .mean()
        .reset_index()
    )
    
    # Multiplicar por 100 para visualizar en porcentaje (%)
    df_agg[target_col] = df_agg[target_col] * 100
    
    # 3. Crear la figura unificada con diseño ejecutivo
    fig, ax = plt.subplots(figsize=(14, 6))
    
    segmentos = sorted(df_agg[segment_col].unique())
    palette = sns.color_palette("tab10", len(segmentos))
    
    for i, seg in enumerate(segmentos):
        sub_seg = df_agg[df_agg[segment_col] == seg]
        
        # Reindexar para alinear con todos los periodos temporales
        sub_seg = sub_seg.set_index('period_num').reindex(all_periods_sorted).reset_index()
        sub_seg['period_label'] = sub_seg['period_num'].map(label_mapping)
        
        sub_clean = sub_seg.dropna(subset=[target_col])
        
        if not sub_clean.empty:
            ax.plot(
                sub_clean['period_label'], 
                sub_clean[target_col], 
                marker='o', 
                linewidth=2, 
                markersize=4,
                color=palette[i],
                label=str(seg)
            )
            
    # Storytelling gerencial y etiquetas claras
    ax.set_title('Evolución y Discriminación Temporal de la Tasa de Conversión (%) por Segmento', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Periodo de Originación', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tasa Promedio de Conversión (%)', fontsize=11, fontweight='bold')
    
    # Leyenda limpia al costado
    ax.legend(title='Segmentos del Árbol', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize=10)
    
    # Rotación vertical (90°) de los meses para evitar solapamientos
    ax.tick_params(axis='x', rotation=90, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_escalera_tasas_global(df, segment_col='segmento_arbol_new', target_col='target_precancelacion'):
    """
    Genera un gráfico de barras estático para la escalera de pricing, 
    mostrando la tasa promedio global de pre-cancelación en porcentaje (%) con storytelling ejecutivo.
    """
    # 1. Calcular la tasa promedio global por segmento y convertir a porcentaje (%)
    df_agg = df.groupby(segment_col)[target_col].mean().reset_index()
    df_agg['target_pct'] = df_agg[target_col] * 100
    
    # Ordenar estrictamente por segmento
    df_agg = df_agg.sort_values(by=segment_col)
    
    # 2. Configurar la figura con diseño ejecutivo
    fig, ax = plt.subplots(figsize=(10, 6))
    
    palette = sns.color_palette("Blues_d", len(df_agg))
    
    bars = ax.bar(
        df_agg[segment_col].astype(str), 
        df_agg['target_pct'], 
        color=palette, 
        edgecolor='black', 
        linewidth=0.6
    )
    
    # 3. Añadir etiquetas de valor en porcentaje (%) encima de cada barra
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),  # Margen superior
            textcoords="offset points",
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#333333'
        )
        
    # Storytelling gerencial
    ax.set_title('Comportamiento de Conversión por Segmento', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Segmentos del Árbol', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tasa de Conversión (%)', fontsize=11, fontweight='bold')
    
    # Dar un respiro superior al eje Y para que las etiquetas no se corten
    ax.set_ylim(0, df_agg['target_pct'].max() * 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=0, fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.show()

def evaluar_estabilidad_por_metrica(df_resumen, umbral=0.10):
    """
    Evalúa si todas las bases son estables para cada métrica (Promedio, Mediana, Máximo)
    y retorna un diccionario con el resultado ('Sí' o 'No').
    """
    # Columnas de métricas a evaluar
    metricas = ['Promedio', 'Mediana', 'Máximo']
    
    resultado_dict = {}
    for metrica in metricas:
        # Verifica si TODOS los valores de esa columna están por debajo del umbral
        es_estable = (df_resumen[metrica] < umbral).all()
        resultado_dict[metrica] = 'Sí' if es_estable else 'No'
        
    return resultado_dict


def calcular_reporte_cruces_por_cosecha(df, period_col, segment_col, target_col, ascending=True):
    """
    Calcula el número de cruces/inversiones de orden por cosecha (periodo) 
    a partir de una base a nivel de cliente, y retorna el detalle mensual 
    junto con su resumen estadístico global.
    
    Parámetros:
    - df: DataFrame a nivel de cliente.
    - period_col: Nombre de la columna de periodo/cosecha (ej. 'periodo_x').
    - segment_col: Nombre de la columna de segmentos (ej. 'segmento_arbol_new').
    - target_col: Nombre de la variable target (ej. 'target_precancelacion').
    - ascending: True si el orden esperado es de menor a mayor tasa (ej. Segmento A < B < C).
    
    Retorna:
    - df_detallado: Reporte 1 (Conteo de cruces por cada cosecha).
    - resumen_estadistico: Reporte 2 (Suma, promedio, mediana, máximo y mínimo).
    """
    # 1. Agrupar la base de clientes para obtener la tasa real por periodo y segmento
    # (tasa = suma de targets / cuenta de clientes, que es el promedio)
    df_agrupado = (
        df.groupby([period_col, segment_col])[target_col]
        .agg(tasa='mean', n_clientes='count', n_targets='sum')
        .reset_index()
    )
    
    # 2. Pivotar: Periodos en filas, Segmentos en columnas
    df_pivote = df_agrupado.pivot(index=period_col, columns=segment_col, values='tasa')
    
    # Determinar el orden global de referencia basado en el promedio histórico de toda la base
    orden_global = (
        df.groupby(segment_col)[target_col]
        .mean()
        .sort_values(ascending=ascending)
        .index.tolist()
    )
    
    # Alinear las columnas del pivote con el orden esperado
    segmentos_validos = [s for s in orden_global if s in df_pivote.columns]
    df_pivote = df_pivote[segmentos_validos]
    
    # 3. Contar cruces (inversiones de orden) por cada periodo (cosecha)
    conteo_cruces = []
    
    for periodo, row in df_pivote.iterrows():
        tasas = row.values
        if np.isnan(tasas).any():
            continue  # Omitir si falta algún segmento en ese periodo
            
        cruces_periodo = 0
        n = len(tasas)
        # Contar pares invertidos respecto al orden esperado
        for i in range(n):
            for j in range(i + 1, n):
                if ascending:
                    if tasas[i] > tasas[j]:
                        cruces_periodo += 1
                else:
                    if tasas[i] < tasas[j]:
                        cruces_periodo += 1
                        
        conteo_cruces.append({
            'periodo': periodo,
            'num_cruces': cruces_periodo
        })
        
    df_detallado = pd.DataFrame(conteo_cruces).sort_values('periodo').reset_index(drop=True)
    
    # 4. Calcular el resumen estadístico global requerido
    serie_cruces = df_detallado['num_cruces']
    resumen_estadistico = {
        'suma': int(serie_cruces.sum()),
        'promedio': float(serie_cruces.mean()),
        'mediana': float(serie_cruces.median()),
        'maximo': int(serie_cruces.max()),
        'minimo': float(serie_cruces.min())
    }
    
    return df_detallado, resumen_estadistico


def obtener_diccionario_cumplimiento_separacion(ratio_precancelacion, umbral=0.03):
    """
    Recibe el DataFrame ratio_precancelacion (con segmentos en filas y bases en columnas),
    calcula las diferencias entre buckets adyacentes y retorna un diccionario 
    con el porcentaje de cumplimiento del umbral para cada base.
    """
    porcentaje_cumplimiento_dict = {}
    
    for base in ratio_precancelacion.columns:
        # Extraer las tasas de la base actual y eliminar nulos
        tasas = ratio_precancelacion[base].dropna()
        
        # Calcular la diferencia entre buckets adyacentes (fila siguiente - fila actual)
        diferencias = tasas.diff().dropna()
        
        if len(diferencias) == 0:
            porcentaje_cumplimiento_dict[base] = 0.0
            continue
            
        # Evaluar cuántas diferencias cumplen con el umbral mínimo
        cumplen = (diferencias >= umbral).sum()
        total = len(diferencias)
        
        # Calcular el porcentaje de cumplimiento
        porcentaje_cumplimiento_dict[base] = float(cumplen / total) if total > 0 else 0.0
        
    return porcentaje_cumplimiento_dict

def consolidar_reporte_ejecutivo(dict_cumplimiento, estadisticos_globales, resultado_psi):
    """
    Recibe los diccionarios de cumplimiento, estadísticos globales y PSI,
    y los consolida en un DataFrame de una sola fila con columnas organizadas.
    """
    fila_datos = {}
    
    # 1. Agregar cumplimiento de separación por base
    for base, valor in dict_cumplimiento.items():
        fila_datos[f'cumplimiento_sep_{base}'] = valor
        
    # 2. Agregar estadísticos globales de cruces
    for stat, valor in estadisticos_globales.items():
        fila_datos[f'cruces_{stat}'] = valor
        
    # 3. Agregar resultado de estabilidad PSI
    for metrica, valor in resultado_psi.items():
        fila_datos[f'psi_estable_{metrica}'] = valor
        
    # Retornar como un DataFrame de una sola fila
    return pd.DataFrame([fila_datos])





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_materialidad_y_precancelacion(
    df,
    period_col='periodo_x',
    segment_col='segmento_arbol_new',
    target_col='target_precancelacion',
    oot_inicio='202210',
    figsize=(16, 7)
):
    """
    Gráfico combinado:

    - Líneas: tasa de precancelación por segmento
      -> eje izquierdo, métrica principal.

    - Barras apiladas: composición de la cartera
      -> eje derecho, métrica de estabilidad.

    - Sombra gris: periodo OOT.
    - Línea vertical entrecortada: inicio del OOT.
    """

    df_plot = df.copy()

    # ============================================================
    # 1. PREPARAR PERIODOS
    # ============================================================

    df_plot['period_num'] = pd.to_numeric(
        df_plot[period_col],
        errors='coerce'
    )

    df_plot = df_plot.dropna(
        subset=['period_num']
    )

    meses_dict = {
        '01': 'Ene',
        '02': 'Feb',
        '03': 'Mar',
        '04': 'Abr',
        '05': 'May',
        '06': 'Jun',
        '07': 'Jul',
        '08': 'Ago',
        '09': 'Sep',
        '10': 'Oct',
        '11': 'Nov',
        '12': 'Dic'
    }

    def formatear_periodo(p):

        p = str(int(p))

        if len(p) == 6:
            mm = p[4:6]
            yy = p[2:4]

            return f"{meses_dict.get(mm, mm)} '{yy}"

        return p

    periodos = sorted(
        df_plot['period_num'].unique()
    )

    labels = [
        formatear_periodo(p)
        for p in periodos
    ]

    pos = {
        periodo: i
        for i, periodo in enumerate(periodos)
    }

    # ============================================================
    # 2. COMPOSICIÓN DE CARTERA
    # ============================================================

    df_comp = (
        df_plot
        .groupby(
            ['period_num', segment_col]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(periodos)
    )

    df_comp = (
        df_comp
        .div(
            df_comp.sum(axis=1),
            axis=0
        )
        * 100
    )

    # ============================================================
    # 3. TASA DE PRECANCELACIÓN
    # ============================================================

    df_prec = (
        df_plot
        .groupby(
            ['period_num', segment_col]
        )[target_col]
        .mean()
        .unstack()
        .reindex(periodos)
        * 100
    )

    # ============================================================
    # 4. SEGMENTOS
    # ============================================================

    segmentos = sorted(
        df_plot[segment_col]
        .dropna()
        .unique()
    )

    palette = sns.color_palette(
        "Set2",
        len(segmentos)
    )

    color_map = dict(
        zip(segmentos, palette)
    )

    # ============================================================
    # 5. FIGURA
    # ============================================================

    fig, ax1 = plt.subplots(
        figsize=figsize
    )

    # Eje secundario
    ax2 = ax1.twinx()

    x = list(
        range(len(periodos))
    )

    # ============================================================
    # 6. SOMBRA OOT
    # ============================================================

    oot_inicio_num = int(
        oot_inicio
    )

    if oot_inicio_num in pos:

        x_inicio_oot = (
            pos[oot_inicio_num] - 0.5
        )

        x_fin_oot = (
            len(periodos) - 0.5
        )

        # Sombra
        ax1.axvspan(
            x_inicio_oot,
            x_fin_oot,
            color='gray',
            alpha=0.12,
            zorder=0
        )

        # Línea de corte
        ax1.axvline(
            x=x_inicio_oot,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            zorder=3
        )

        # Etiqueta
        ax1.text(
            (x_inicio_oot + x_fin_oot) / 2,
            0.97,
            'OOT',
            transform=ax1.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=10,
            fontweight='bold',
            color='dimgray'
        )

    # ============================================================
    # 7. BARRAS — COMPOSICIÓN
    #    EJE DERECHO
    # ============================================================

    bottom = [
        0
    ] * len(periodos)

    for seg in segmentos:

        if seg in df_comp.columns:

            valores = (
                df_comp[seg]
                .values
            )

        else:

            valores = [
                0
            ] * len(periodos)

        ax2.bar(
            x,
            valores,
            bottom=bottom,
            width=0.75,
            color=color_map[seg],
            alpha=0.18,
            edgecolor='white',
            linewidth=0.5,
            zorder=1
        )

        bottom = [
            b + v
            for b, v in zip(
                bottom,
                valores
            )
        ]

    # ============================================================
    # 8. LÍNEAS — PRECANCELACIÓN
    #    EJE IZQUIERDO
    # ============================================================

    for seg in segmentos:

        if seg not in df_prec.columns:
            continue

        ax1.plot(
            x,
            df_prec[seg].values,
            marker='o',
            markersize=4.5,
            linewidth=2.8,
            color=color_map[seg],
            label=str(seg),
            zorder=5
        )

    # ============================================================
    # 9. EJE X
    # ============================================================

    ax1.set_xticks(x)

    ax1.set_xticklabels(
        labels,
        rotation=90,
        fontsize=9
    )

    ax1.set_xlabel(
        'Periodo de originación',
        fontsize=11,
        fontweight='bold'
    )

    # ============================================================
    # 10. EJE IZQUIERDO
    #     PRECANCELACIÓN
    # ============================================================

    ax1.set_ylabel(
        'Tasa de conversión (%)',
        fontsize=11,
        fontweight='bold'
    )

    # ============================================================
    # 11. EJE DERECHO
    #     COMPOSICIÓN
    # ============================================================

    ax2.set_ylabel(
        'Composición de la cartera (%)',
        fontsize=11,
        fontweight='bold'
    )

    ax2.set_ylim(
        0,
        100
    )

    # ============================================================
    # 12. TÍTULO
    # ============================================================

    ax1.set_title(
        'Discriminación y estabilidad de la segmentación',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    # ============================================================
    # 13. GRID
    # ============================================================

    ax1.grid(
        axis='y',
        linestyle='--',
        alpha=0.30,
        zorder=0
    )

    ax2.grid(
        False
    )

    # ============================================================
    # 14. LEYENDA
    # ============================================================

    ax1.legend(
        title='Segmento',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=9,
        title_fontsize=10
    )

    # ============================================================
    # 15. AJUSTES
    # ============================================================

    ax1.tick_params(
        axis='y',
        labelsize=9
    )

    ax2.tick_params(
        axis='y',
        labelsize=9
    )

    plt.tight_layout()

    plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_precancelacion_barras_agrupadas(
    df,
    period_col='periodo_x',
    segment_col='segmento_arbol_new',
    target_col='target_precancelacion',
    oot_inicio='202210',
    mostrar_promedios=True,  # Parámetro para activar/desactivar líneas promedio
    figsize=(16, 7),
):
  """Gráfico de barras agrupadas por periodo (cosecha) para la tasa de precancelación,

  con opción de mostrar líneas horizontales de promedio por segmento.
  """
  df_plot = df.copy()

  # ============================================================
  # 1. PREPARAR PERIODOS
  # ============================================================
  df_plot['period_num'] = pd.to_numeric(df_plot[period_col], errors='coerce')
  df_plot = df_plot.dropna(subset=['period_num'])

  meses_dict = {
      '01': 'Ene',
      '02': 'Feb',
      '03': 'Mar',
      '04': 'Abr',
      '05': 'May',
      '06': 'Jun',
      '07': 'Jul',
      '08': 'Ago',
      '09': 'Sep',
      '10': 'Oct',
      '11': 'Nov',
      '12': 'Dic',
  }

  def formatear_periodo(p):
    p = str(int(p))
    if len(p) == 6:
      mm = p[4:6]
      yy = p[2:4]
      return f"{meses_dict.get(mm, mm)} '{yy}"
    return p

  periodos = sorted(df_plot['period_num'].unique())
  labels = [formatear_periodo(p) for p in periodos]

  # ============================================================
  # 2. TASA DE PRECANCELACIÓN
  # ============================================================
  df_prec = (
      df_plot.groupby(['period_num', segment_col])[target_col]
      .mean()
      .unstack()
      .reindex(periodos)
      * 100
  )

  # ============================================================
  # 3. SEGMENTOS Y COLORES
  # ============================================================
  segmentos = sorted(df_plot[segment_col].dropna().unique())
  palette = sns.color_palette('Set2', len(segmentos))
  color_map = dict(zip(segmentos, palette))

  # ============================================================
  # 4. FIGURA Y CONFIGURACIÓN DE BARRAS AGRUPADAS
  # ============================================================
  fig, ax = plt.subplots(figsize=figsize)

  x = np.arange(len(periodos))
  n_seg = len(segmentos)
  width = 0.8 / n_seg  # Ancho dinámico para que las barras queden pegadas

  # ============================================================
  # 5. SOMBRA OOT
  # ============================================================
  oot_inicio_num = int(oot_inicio)
  pos = {periodo: i for i, periodo in enumerate(periodos)}

  if oot_inicio_num in pos:
    x_inicio_oot = pos[oot_inicio_num] - 0.5
    x_fin_oot = len(periodos) - 0.5

    ax.axvspan(x_inicio_oot, x_fin_oot, color='gray', alpha=0.12, zorder=0)
    ax.axvline(
        x=x_inicio_oot, color='gray', linestyle='--', linewidth=1.5, zorder=3
    )
    ax.text(
        (x_inicio_oot + x_fin_oot) / 2,
        0.97,
        'OOT',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='top',
        fontsize=10,
        fontweight='bold',
        color='dimgray',
    )

  # ============================================================
  # 6. BARRAS AGRUPADAS Y PROMEDIOS
  # ============================================================
  for i, seg in enumerate(segmentos):
    if seg not in df_prec.columns:
      continue

    # Posición centrada para cada barra del grupo
    x_pos = x + (i - (n_seg - 1) / 2) * width
    valores = df_prec[seg].values

    # Graficar barras
    ax.bar(
        x_pos,
        valores,
        width=width,
        color=color_map[seg],
        edgecolor='white',
        linewidth=0.5,
        label=str(seg),
        zorder=3,
    )

    # Línea promedio por serie (si está activada)
    if mostrar_promedios:
      media_val = np.nanmean(valores)
      if not np.isnan(media_val):
        ax.axhline(
            y=media_val,
            color=color_map[seg],
            linestyle=':',
            linewidth=1.8,
            alpha=0.85,
            zorder=4,
        )

  # ============================================================
  # 7. EJES, TÍTULO Y ESTÉTICA
  # ============================================================
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=90, fontsize=9)
  ax.set_xlabel(
      'Periodo de originación (Cosecha)', fontsize=11, fontweight='bold'
  )
  ax.set_ylabel('Tasa de Conversión (%)', fontsize=11, fontweight='bold')
  ax.set_title(
      'Tasa de Conversión por Cosecha y Segmento',
      fontsize=14,
      fontweight='bold',
      pad=15,
  )

  ax.grid(axis='y', linestyle='--', alpha=0.30, zorder=0)
  ax.legend(
      title='Segmento',
      bbox_to_anchor=(1.02, 1),
      loc='upper left',
      fontsize=9,
      title_fontsize=10,
  )

  ax.tick_params(axis='y', labelsize=9)
  plt.tight_layout()
  plt.show()