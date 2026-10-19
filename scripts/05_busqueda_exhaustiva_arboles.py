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

from statsmodels.formula.api import ols

import xgboost as xgb
import shap
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm

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
UMBRAL_CRITICO_COMPLETITUD=0.2
ULTIMO_MES_DEV=202601

#____________________ 5) Funnel ----

datos=datos.query(f'periodo>={PERIODO_INICIO} and periodo<={PERIODO_FIN}')

funnel_oferta_relevante=['exito_desembolso_ibk_ct','fracaso_desembolso_no_ibk_monto_similar_con_gestion']

datos = datos.query("flg_exito_4 in @funnel_oferta_relevante").copy()

#____________________ 6) Preprocesamiento ----

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

CLUSTER_MACROPERFILES={
     'cluster_1_sow_no_nulo': 'Fidelizados',
     'cluster_2_sow_no_nulo': 'Fidelizados',
     'cluster_3_sow_no_nulo': 'Por Fidelizar - Mayor Escala',
     'cluster_4_sow_no_nulo': 'Por Fidelizar - Mayor Escala',
     'cluster_5_sow_no_nulo': 'Por Fidelizar - Escala Intermedia',
     'cluster_6_sow_no_nulo': 'Por Fidelizar - Escala Intermedia',     
     'cluster_1_nulo': 'Sin Deuda',
     'cluster_2_nulo': 'Sin Deuda',
     'cluster_3_nulo': 'Sin Deuda'
}

datos['macroperfil'] = datos['cluster'].map(CLUSTER_MACROPERFILES).fillna(datos['cluster'])
datos[['macroperfil','cluster']].value_counts().reset_index()

datos['nombre_ifi']=datos['cod_instit_financiera'].map({1.0:'BCP', 6.0: 'BBVA', 4.0:'SCOTIA', 123.0:'MI BANCO'}).fillna('OTROS')

#~~~~ 6.3) Nuevas Variables ----

datos['tasa_cliente']=np.where(datos['tasa_desembolsada'].isna(),datos['tasa_min'],datos['tasa_desembolsada'])
datos['monto_cliente']=np.where(datos['monto_desembolsado'].isna(),datos['monto_min'],datos['monto_desembolsado'])

#____________________ 7) Segmentación: Árbol de decisión ----

#~~~~ 7.1) Construcción: Base de Propension ----

base_propension = (
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

#~~~~ 7.2) Procesamiento Inicial: Base de Propensión ----

#---- 7.2.1) Tratamiento: Monto ofertado ----

base_propension['monto_ofertado_ajustado'] = np.where(
    base_propension['monto_desembolsado'].isna(),
    base_propension['monto_min'],
    np.where(
        base_propension['monto_desembolsado'] > base_propension['monto_min'],
        base_propension['monto_desembolsado'],
        base_propension['monto_min']
    )
)

base_propension.periodo.value_counts().sort_index()

#---- 7.2.2) Identificación de Variables ----

var_dependiente='indicador_exito'
var_ponderacion='monto_ofertado_ajustado'

VARIABLES_BASE_BRUTA= base_propension.columns.to_list().copy()

var_enteras = [
    'plazo_min',
    'plazo_max',
    'antiguedad_empresa',
    'nivel_ventas'
]

var_continuas = [
    'tasa_min',
    'tasa_max',
    'monto_min',
    'monto_max',
    'tasa_cliente',
    'monto_cliente',
    'pasivosprom_u3m',
    'pasivosprom_u6m',
    'pasivosprom_u12m',
    'salprom_creditos_u03m',
    'salprom_creditos_u06m',
    'salprom_creditos_u12m',
    'vol_izipay_u03m',
    'vol_izipay_u06m',
    'suma_uai_u12m',
    'promedio_uai_u12m',
    'deuda_directa_vig_ibk_sf_u03m',
    'deuda_directa_vig_ibk_sf_u06m',
    'deuda_directa_vig_ibk_sf_u12m',
    'deuda_directa_vig_total_sf_u03m',
    'deuda_directa_vig_total_sf_u06m',
    'deuda_directa_vig_total_sf_u12m',
    'avg_deuda_directa_vig_ibk_sf_u03m',
    'avg_deuda_directa_vig_ibk_sf_u06m',
    'avg_deuda_directa_vig_ibk_sf_u12m',
    'avg_deuda_directa_vig_total_sf_u03m',
    'avg_deuda_directa_vig_total_sf_u06m',
    'avg_deuda_directa_vig_total_sf_u12m',
    'saldo_alta',
    'sow_deuda_u12m',
    'monto_desembolsado',
    'tasa_2022',
    'tasa_evol',
    'delta_deuda_final',
    'saldo_desembolso',
    'tasa_ultima',
    'tasa_desembolsada',
    'monto_ofertado_ajustado'
]

var_binarios = [
    'flag_izipay_u03m',
    'flag_izipay_u06m',
    'flag_izipay_u12m',
    'flg_desembolso_ibk_all_products',
    'indicador_exito'
]

var_categoricas = [
    'conteo',
    'segmento_abc',
    'sei_mcdo',
    'departamento',
    'cod_instit_financiera',
    'cluster',
    'gestion_final',
    'flg_gestion',
    'producto',
    'campana_vpc',
    'flg_exito_4',
    'cod_entidad',
    'conteo_inferencia',
    'cluster_comercial',
    'macroperfil',
    'nombre_ifi'
    
]
var_fechas = [
    'periodo',
    'codmes',
    'antiguedad_izipay',
    'codmes_2',
    'periodo_dt',
    'fecha',
    'anio',
    'trimestre',
    'periodo_q',
    'fec_desembolso'
]

var_ids = [
    'llave_interna',
    'key_value'
]

var_numericas = var_enteras+var_continuas

VARIABLES_CLAVE_FASE_I= list(dict.fromkeys(
    [var_dependiente, var_ponderacion]

))

#---- 7.2.3) Imputacion by Negocio ----

# Para las siguientes variables de comportamiento financiero IBK/RCC, la ausencia de
# información se interpreta como ausencia de relación o producto financiero,
# por lo que se asigna valor cero. Para variables donde la falta de información
# no implica ausencia del comportamiento (ej. ingresos o datos personales), se
# aplican otros tratamientos de imputación.

base_propension.isnull().sum()

reglas_imputacion_negocio={
    v: 0
    for v in [
        'salprom_creditos_u03m',
        'salprom_creditos_u06m',
        'salprom_creditos_u12m',
        'avg_deuda_directa_vig_ibk_sf_u12m',
        'avg_deuda_directa_vig_ibk_sf_u06m',
        'avg_deuda_directa_vig_total_sf_u12m',
        'avg_deuda_directa_vig_total_sf_u06m',
        'avg_deuda_directa_vig_ibk_sf_u03m',
        'avg_deuda_directa_vig_total_sf_u03m'
    ]
}

base_propension, resumen_imputacion_negocio = mdl.aplicar_reglas_negocio_na(
    base_propension,
    reglas_imputacion_negocio
)

resumen_imputacion_negocio

#~~~~ 7.3) Identificación de Variables Aptas ----

#---- 7.3.1) Completitud ----

# Identificación de variables aptas respecto a completitud de información,
# descartando aquellas con niveles de missing que puedan afectar el modelamiento.

completitud_global=(
    100*base_propension.isnull()
    .mean()
    .sort_values(ascending=False)
    .to_frame('pct_faltantes')
)

completitud_global.query(f"pct_faltantes>=100*{UMBRAL_CRITICO_COMPLETITUD}")
completitud_global

VAR_BANNED_INCOMPLETITUD=completitud_global.query(f"pct_faltantes>=100*{UMBRAL_CRITICO_COMPLETITUD}").index.to_list()
print("⚠️ Variables con problemas de completitud:")
print("\n".join([f" - {v}" for v in VAR_BANNED_INCOMPLETITUD]))

VAR_VALIDAS_COMPLETITUD=[var for var in VARIABLES_BASE_BRUTA if var not in VAR_BANNED_INCOMPLETITUD]

if set(VARIABLES_CLAVE_FASE_I).issubset(VAR_VALIDAS_COMPLETITUD):
    print("✅ Todas las variables clave superan la validación de completitud.")
else:
    vars_no_validas = sorted(set(VARIABLES_CLAVE_FASE_I) - set(VAR_VALIDAS_COMPLETITUD))
    print(f"⚠️ Las siguientes variables clave presentan problemas de completitud:")
    print("\n".join([f" - {v}" for v in vars_no_validas]))

#---- 7.3.2) Variabilidad ----

# Identificación de variables con suficiente variabilidad para aportar información
# relevante al proceso de modelamiento.

constantes_global=mdl.detectar_variables_constantes(base_propension[VAR_VALIDAS_COMPLETITUD],     
                              umbral_categoria=0.98,
                              umbral_unico=0.98,
                              tolerancia_std=1e-8)

constantes_global.query('estado=="OK"').shape[0]/len(VAR_VALIDAS_COMPLETITUD)

constantes_global.query('tipo=="Categorica" and estado=="OK"')
constantes_global.query('tipo!="Categorica" and estado=="OK"')

constantes_global.query('tipo=="Categorica" and estado!="OK"')
constantes_global.query('tipo!="Categorica" and estado!="OK"')

constantes_global.query(f"variable in {[var_dependiente, var_ponderacion]}")

VAR_BANNED_CONSTANTES=constantes_global.query('estado!="OK"').variable.to_list()
print("⚠️ Variables con problemas de poca variabilidad:")
print("\n".join([f" - {v}" for v in VAR_BANNED_CONSTANTES]))

VAR_VALIDAS_VARIABILIDAD=[var for var in VAR_VALIDAS_COMPLETITUD if var not in VAR_BANNED_CONSTANTES]

if set(VARIABLES_CLAVE_FASE_I).issubset(VAR_VALIDAS_VARIABILIDAD):
    print("✅ Todas las variables clave superan la validación de variabilidad.")
else:
    vars_no_validas = sorted(set(VARIABLES_CLAVE_FASE_I) - set(VAR_VALIDAS_VARIABILIDAD))
    print(f"⚠️ Las siguientes variables clave presentan problemas de variabilidad:")
    print("\n".join([f" - {v}" for v in vars_no_validas]))

#---- 7.3.3) Estabilidad ----

# Identificación de variables con baja estabilidad temporal, que podrían afectar
# la robustez y capacidad de generalización del modelo.

# Omitimos variables, principalmente de identificación, para los analisis de estabilidad
VAR_OMITIDAS_ESTABILIDAD=var_ids+var_fechas

psi_detalle, psi_resumen, psi_omitidas = mdl.calcular_psi_temporal(
    df=base_propension,
    variables=VAR_VALIDAS_VARIABILIDAD,
    columna_periodo="periodo",
    periodo_referencia=ULTIMO_MES_DEV,
    n_bins=5,
    top_n_categorias=5,
    variables_omitir=VAR_OMITIDAS_ESTABILIDAD
)

psi_detalle
psi_resumen
psi_omitidas

VAR_BANNED_ESTABILIDAD=(psi_resumen.query('estado not in ["Estable","Cambio moderado"]').variable.to_list() 
                        + psi_omitidas.query('motivo not in ["Excluida por el usuario"]').variable.to_list())

print("⚠️ Variables con problemas de estabilidad:")
print("\n".join([f" - {v}" for v in VAR_BANNED_ESTABILIDAD]))

VAR_VALIDAS_ESTABILIDAD=[var for var in VAR_VALIDAS_VARIABILIDAD if var not in VAR_BANNED_ESTABILIDAD]

if set(VARIABLES_CLAVE_FASE_I).issubset(VAR_VALIDAS_ESTABILIDAD):
    print("✅ Todas las variables clave superan la validación de estabilidad temporal.")
else:
    vars_no_validas = sorted(set(VARIABLES_CLAVE_FASE_I) - set(VAR_VALIDAS_ESTABILIDAD))
    print(f"⚠️ Las siguientes variables clave presentan problemas de estabilidad temporal:")
    print("\n".join([f" - {v}" for v in vars_no_validas]))

#---- 7.3.4) Validacion de consistencia ----

# Se evalúa la consistencia y razonabilidad de la información, verificando que las
# variables presenten comportamientos coherentes con el contexto del negocio.
# Asimismo, se revisan posibles inconsistencias, relaciones inesperadas entre
# variables y valores no esperados que puedan afectar la calidad del modelamiento.

#____ Razonabilidad informacion (univariada) ----

VAR_ANALISIS_CONSISTENCIA=[var for var in VAR_VALIDAS_ESTABILIDAD if var not in (VAR_OMITIDAS_ESTABILIDAD)]

#____ Estadísticos ----

descriptivos_num, descriptivos_cat =mdl.descriptivos_variables(base_propension, VAR_ANALISIS_CONSISTENCIA)

descriptivos_num

#____ Plots----

i=1
for var in VAR_ANALISIS_CONSISTENCIA:
    print(f"\t\t\t{var} - {i} de {len(VAR_ANALISIS_CONSISTENCIA)}")
    mdl.graficar_distribucion(
        base_propension,
        var,
        max_categorias=5
    )
    i=i+1

# Análisis:
# Se observa que las variables presentan comportamientos consistentes y acordes con
# lo esperado según el contexto del negocio.

#____ Caracterización de Variables: Númericas ----

mdl.evaluar_distribucion_numericas(
    base_propension,
    [var for var in VAR_ANALISIS_CONSISTENCIA if var in var_numericas],
    umbral_cv=1,
    umbral_cardinalidad=10,
    umbral_outlier=0.01
)

#____ Caracterización de Variables: Categoricas y Binarias ----

diagnostico_categoricas, variables_one_hot, variables_revision = (
    mdl.evaluar_cardinalidad_categoricas(
        base_propension,
        [var for var in VAR_ANALISIS_CONSISTENCIA if var in (var_categoricas+var_binarios)],
        umbral_categorias=5,
        umbral_dominancia=0.95,
        umbral_frecuencia_minima=0.01
    )
)

diagnostico_categoricas

#____ Exclusion por inconsistencia univariada ----

VAR_BANNED_CONSISTENCIA_UNIVARIADA=[]

print("⚠️ Variables con problemas de consistencia univariada:")
print("\n".join([f" - {v}" for v in VAR_BANNED_CONSISTENCIA_UNIVARIADA]))

VAR_VALIDAS_CONSISTENCIA_UNIVARIADA=[var for var in VAR_VALIDAS_ESTABILIDAD if var not in VAR_BANNED_CONSISTENCIA_UNIVARIADA]

if set(VARIABLES_CLAVE_FASE_I).issubset(VAR_VALIDAS_CONSISTENCIA_UNIVARIADA):
    print("✅ Todas las variables clave superan la validación de consistencia univariada.")
else:
    vars_no_validas = sorted(set(VARIABLES_CLAVE_FASE_I) - set(VAR_VALIDAS_CONSISTENCIA_UNIVARIADA))
    print(f"⚠️ Las siguientes variables clave presentan problemas de consistencia univariada:")
    print("\n".join([f" - {v}" for v in vars_no_validas]))

#---- 7.3.5) Consistencia entre variables ----

#____ Exclusion por inconsistencia multivariada ----

VAR_BANNED_CONSISTENCIA_MULTIVARIADA=[]

print("⚠️ Variables con problemas de consistencia multivariada:")
print("\n".join([f" - {v}" for v in VAR_BANNED_CONSISTENCIA_MULTIVARIADA]))

VAR_VALIDAS_CONSISTENCIA_MULTIVARIADA=[var for var in VAR_VALIDAS_CONSISTENCIA_UNIVARIADA if var not in VAR_BANNED_CONSISTENCIA_MULTIVARIADA]

if set(VARIABLES_CLAVE_FASE_I).issubset(VAR_VALIDAS_CONSISTENCIA_MULTIVARIADA):
    print("✅ Todas las variables clave superan la validación de consistencia multivariada.")
else:
    vars_no_validas = sorted(set(VARIABLES_CLAVE_FASE_I) - set(VAR_VALIDAS_CONSISTENCIA_MULTIVARIADA))
    print(f"⚠️ Las siguientes variables clave presentan problemas de consistencia multivariada:")
    print("\n".join([f" - {v}" for v in vars_no_validas]))

#---- 7.3.6) Variables seleccionadas posterior al EDA ----

# Se consolidan las variables que cumplen los criterios de diagnóstico definidos
# en la Fase I (completitud, variabilidad, estabilidad y consistencia), conformando
# el conjunto de variables aptas para continuar con la etapa de tratamiento y modelamiento.

VAR_VALIDAS_POST_EDA = VAR_VALIDAS_CONSISTENCIA_MULTIVARIADA.copy()

#~~~~ 7.4) Partición: Train/Test/OOT ----

oot=base_propension.query(f'periodo>{ULTIMO_MES_DEV}')
dev=base_propension.query(f'periodo<={ULTIMO_MES_DEV}')

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

#~~~~ 7.5) Framework: Ingeniería de Variables - Tratamiento y Transformación ----

#---- 7.5.1) Identificacion de variables ----

# En esta fase se aplican los tratamientos definidos sobre las variables aptas
# identificadas en la Fase I, ajustando los parámetros únicamente con la base de
# entrenamiento para evitar fuga de información y asegurar una correcta
# generalización hacia los conjuntos de validación.

# Variables identificadoras (a excluir):
VAR_IDENTIFICADORAS=var_ids+var_fechas

# Variables válidas para ingeniería:
VAR_ING_NUMERICAS = [
    var for var in VAR_VALIDAS_POST_EDA 
    if var not in VAR_IDENTIFICADORAS and var in var_numericas
]

VAR_ING_CATEGORICAS = [
    var for var in VAR_VALIDAS_POST_EDA 
    if var not in VAR_IDENTIFICADORAS and var in var_categoricas
]

VAR_ING_BINARIAS = [
    var for var in VAR_VALIDAS_POST_EDA 
    if var not in VAR_IDENTIFICADORAS and var in var_binarios
]

#---- 7.5.2) Caracterización de Variables ----

mdl.evaluar_distribucion_numericas(
    base_propension.query('base=="train"'),
    [var for var in VAR_ING_NUMERICAS],
    umbral_cv=1,
    umbral_cardinalidad=10,
    umbral_outlier=0.01
)

diagnostico_categoricas, variables_one_hot, variables_revision = (
    mdl.evaluar_cardinalidad_categoricas(
        base_propension.query('base=="train"'),
        [var for var in (VAR_ING_CATEGORICAS+VAR_ING_BINARIAS) ],
        umbral_categorias=5,
        umbral_dominancia=0.95,
        umbral_frecuencia_minima=0.01
    )
)

diagnostico_categoricas

#---- 7.5.3) Funciones ----

# Framework de preparación de variables para modelamiento.
# Aprende parámetros únicamente con train y los aplica sobre train, test y OOT.
#
# Incluye:
# - Imputación de valores faltantes.
# - Tratamiento de valores extremos mediante winsorización (opcional).
# - Validación y transformación de variables binarias/categóricas.
# - One Hot Encoding con cardinalidad controlada.
# - Generación de variables derivadas.
#
# Objetivo: asegurar consistencia en la preparación de variables y evitar
# fuga de información entre muestras.

def _clave_woe(valor):
    """Genera una clave consistente y conserva NaN como categoría propia."""
    if pd.isna(valor):
        return ("__MISSING__",)

    return (type(valor).__name__, repr(valor))

def _categorizar_numerica_woe(serie, cortes):
    """Aplica los intervalos aprendidos en Train."""
    return pd.cut(
        serie.astype(float),
        bins=cortes,
        labels=False,
        include_lowest=True
    )

def fit_transformaciones(
    X_train,
    var_ing_categoricas,
    var_ing_numericas,
    var_ing_binarias,
    y_train=None,
    vars_woe=None,
    n_bins_woe=5,
    vars_cruce_imputacion=None,
    umbral_nas_mediana_global=0.05,
    umbral_nas_imputacion_grupo=0.20,
    variables_no_transformar=None,
    max_categorias_one_hot_encoding=5,
    alpha_winsorizacion=0.01,
    alpha_woe=0.5
):
    """
    Aprende parámetros usando exclusivamente Train.

    WoE:
    - Target binario: 0 = no evento, 1 = evento.
    - Numéricas: intervalos por cuantiles aprendidos en Train.
    - NaN: categoría independiente.
    - Suavizado configurable mediante alpha_woe.
    """

    X_train = X_train.copy()

    var_ing_categoricas = list(var_ing_categoricas)
    var_ing_numericas = list(var_ing_numericas)
    var_ing_binarias = list(var_ing_binarias)
    vars_woe = list(vars_woe or [])
    vars_cruce_imputacion = list(
        vars_cruce_imputacion or ['bin', 'segmento']
    )
    variables_no_transformar = list(variables_no_transformar or [])

    parametros = {}

    # --------------------------------------------------------
    # 1. Validaciones
    # --------------------------------------------------------

    if len(vars_woe) != len(set(vars_woe)):
        raise ValueError("vars_woe contiene variables duplicadas.")

    faltantes = [v for v in vars_woe if v not in X_train.columns]

    if faltantes:
        raise ValueError(f"Variables WoE no encontradas: {faltantes}")

    if vars_woe:
        if y_train is None:
            raise ValueError("Debes proporcionar y_train para aprender WoE.")

        if isinstance(y_train, pd.Series):
            if not X_train.index.equals(y_train.index):
                raise ValueError(
                    "X_train e y_train deben tener el mismo índice y orden."
                )
            y = y_train.copy()
        else:
            if len(y_train) != len(X_train):
                raise ValueError(
                    "X_train e y_train deben tener la misma longitud."
                )
            y = pd.Series(y_train, index=X_train.index)

        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError("y_train debe contener únicamente 0 y 1, sin NaN.")

        if y.nunique() != 2:
            raise ValueError("y_train debe contener ambas clases: 0 y 1.")

        if alpha_woe <= 0:
            raise ValueError("alpha_woe debe ser mayor que cero.")

        if n_bins_woe < 2:
            raise ValueError("n_bins_woe debe ser al menos 2.")

    # Proteger variables de cruce
    for var in vars_cruce_imputacion:
        if var not in variables_no_transformar:
            variables_no_transformar.append(var)

    # --------------------------------------------------------
    # 2. Aprendizaje WoE
    # --------------------------------------------------------

    parametros["vars_woe"] = vars_woe
    parametros["woe_mappings"] = {}
    parametros["woe_global"] = {}
    parametros["woe_cortes_numericos"] = {}
    parametros["woe_variables_numericas"] = []
    parametros["alpha_woe"] = alpha_woe

    for var in vars_woe:

        serie = X_train[var]

        # Identificar variables numéricas continuas
        es_numerica = (
            pd.api.types.is_numeric_dtype(serie)
            and not pd.api.types.is_bool_dtype(serie)
        )

        if es_numerica:

            valores = serie.dropna().astype(float)

            if not np.isfinite(valores).all():
                raise ValueError(
                    f"La variable '{var}' contiene valores infinitos."
                )

            # Aprender cortes usando exclusivamente Train
            if valores.empty:
                cortes = [-np.inf, np.inf]
            else:
                cuantiles = np.linspace(0, 1, n_bins_woe + 1)

                cortes_internos = np.unique(
                    valores.quantile(cuantiles).to_numpy()
                )

                cortes_internos = cortes_internos[
                    (cortes_internos > valores.min())
                    & (cortes_internos < valores.max())
                ]

                cortes = np.concatenate([
                    [-np.inf],
                    cortes_internos,
                    [np.inf]
                ]).tolist()

            parametros["woe_cortes_numericos"][var] = cortes
            parametros["woe_variables_numericas"].append(var)

            categorias = _categorizar_numerica_woe(serie, cortes)

        else:
            # Las categóricas mantienen sus valores originales
            categorias = serie.copy()

        # NaN constituye una categoría propia
        claves = categorias.map(_clave_woe)

        datos_woe = pd.DataFrame({
            "__key": claves,
            "__target": y
        })

        conteos = (
            datos_woe
            .groupby(["__key", "__target"], sort=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
        )

        n_categorias = len(conteos)

        # Suavizado aditivo para evitar proporciones iguales a cero
        dist_no_eventos = (
            conteos[0] + alpha_woe
        ) / (
            (y == 0).sum() + alpha_woe * n_categorias
        )

        dist_eventos = (
            conteos[1] + alpha_woe
        ) / (
            (y == 1).sum() + alpha_woe * n_categorias
        )

        # WoE = ln(% no eventos / % eventos)
        parametros["woe_mappings"][var] = np.log(
            dist_no_eventos / dist_eventos
        ).to_dict()

        # Respaldo neutral para categorías desconocidas
        parametros["woe_global"][var] = 0.0

    # --------------------------------------------------------
    # 3. Filtrar variables protegidas
    # --------------------------------------------------------

    var_ing_categoricas = [
        v for v in var_ing_categoricas
        if v not in variables_no_transformar
    ]
    var_ing_numericas = [
        v for v in var_ing_numericas
        if v not in variables_no_transformar
    ]
    var_ing_binarias = [
        v for v in var_ing_binarias
        if v not in variables_no_transformar
    ]

    parametros["variables_no_transformar"] = variables_no_transformar
    parametros["vars_cruce_imputacion"] = vars_cruce_imputacion

    # --------------------------------------------------------
    # 4. Imputación categóricas y binarias
    # --------------------------------------------------------

    vars_cat_bin = var_ing_categoricas + var_ing_binarias

    if vars_cat_bin:
        modas_df = X_train[vars_cat_bin].mode(dropna=True)
        parametros["modas"] = (
            modas_df.iloc[0].to_dict() if not modas_df.empty else {}
        )
    else:
        parametros["modas"] = {}

    # --------------------------------------------------------
    # 5. Imputación numéricas y winsorización
    # --------------------------------------------------------

    if var_ing_numericas:

        X_train[var_ing_numericas] = (
            X_train[var_ing_numericas].astype(float)
        )

        pct_nas = X_train[var_ing_numericas].isna().mean()

        vars_mediana_global = pct_nas[
            pct_nas <= umbral_nas_mediana_global
        ].index.tolist()

        vars_imputacion_grupo = pct_nas[
            (pct_nas > umbral_nas_mediana_global)
            & (pct_nas <= umbral_nas_imputacion_grupo)
        ].index.tolist()

        vars_descartadas_nas = pct_nas[
            pct_nas > umbral_nas_imputacion_grupo
        ].index.tolist()

        parametros["vars_mediana_global"] = vars_mediana_global
        parametros["vars_imputacion_grupo"] = vars_imputacion_grupo
        parametros["vars_descartadas_nas"] = vars_descartadas_nas

        vars_num_activas = (
            vars_mediana_global + vars_imputacion_grupo
        )

        parametros["medianas"] = (
            X_train[vars_num_activas].median().to_dict()
        )
        parametros["promedios"] = (
            X_train[vars_num_activas].mean().to_dict()
        )

        parametros["matrices_grupo"] = {}

        for var in vars_imputacion_grupo:

            cols_filtro = [var] + vars_cruce_imputacion
            datos_validos = X_train.dropna(subset=cols_filtro)

            if not datos_validos.empty:
                matriz = (
                    datos_validos
                    .groupby(vars_cruce_imputacion)[var]
                    .median()
                    .to_dict()
                )
                parametros["matrices_grupo"][var] = matriz
            else:
                parametros["matrices_grupo"][var] = {}

        if vars_num_activas:

            X_wins = X_train[vars_num_activas].copy()

            for var in vars_num_activas:
                X_wins[var] = X_wins[var].fillna(
                    parametros["medianas"][var]
                )

            q_inf = alpha_winsorizacion / 2
            q_sup = 1 - alpha_winsorizacion / 2

            cuantiles = X_wins.quantile([q_inf, q_sup])

            parametros["limites_winsorizacion"] = (
                cuantiles.T.rename(columns={
                    q_inf: "limite_inferior",
                    q_sup: "limite_superior"
                }).to_dict("index")
            )
        else:
            parametros["limites_winsorizacion"] = {}

    else:
        parametros["vars_mediana_global"] = []
        parametros["vars_imputacion_grupo"] = []
        parametros["vars_descartadas_nas"] = []
        parametros["medianas"] = {}
        parametros["promedios"] = {}
        parametros["matrices_grupo"] = {}
        parametros["limites_winsorizacion"] = {}

    # --------------------------------------------------------
    # 6. Validación de variables binarias
    # --------------------------------------------------------

    binarias_validas = []
    binarias_categoricas = []

    for var in var_ing_binarias:

        valores = X_train[var].dropna().unique().tolist()

        if set(valores).issubset({0, 1, 0.0, 1.0}):
            binarias_validas.append(var)
        else:
            binarias_categoricas.append(var)

    parametros["binarias_validas"] = binarias_validas
    parametros["binarias_categoricas"] = binarias_categoricas

    # --------------------------------------------------------
    # 7. One-Hot Encoding
    # --------------------------------------------------------

    vars_encoding = [
        v for v in var_ing_categoricas + binarias_categoricas
        if v not in vars_woe
    ]

    if vars_encoding:

        X_cat = X_train[vars_encoding].fillna(parametros["modas"])

        variables_one_hot = []
        variables_no_one_hot = []
        resumen_one_hot = []

        for var in vars_encoding:

            n_categorias = X_cat[var].nunique()
            aplica = n_categorias <= max_categorias_one_hot_encoding

            (variables_one_hot if aplica else variables_no_one_hot).append(var)

            resumen_one_hot.append({
                "variable": var,
                "n_categorias_train": n_categorias,
                "aplica_one_hot": aplica
            })

        parametros["variables_one_hot"] = variables_one_hot
        parametros["variables_no_one_hot"] = variables_no_one_hot
        parametros["resumen_one_hot"] = pd.DataFrame(resumen_one_hot)

        if variables_one_hot:

            dummies = pd.get_dummies(
                X_cat[variables_one_hot],
                prefix=variables_one_hot,
                drop_first=True
            )

            dummies = (
                dummies.loc[:, dummies.nunique() > 1]
                .astype(int)
            )

            parametros["columnas_dummies"] = dummies.columns.tolist()
        else:
            parametros["columnas_dummies"] = []

    else:
        parametros["variables_one_hot"] = []
        parametros["variables_no_one_hot"] = []
        parametros["resumen_one_hot"] = pd.DataFrame()
        parametros["columnas_dummies"] = []

    return parametros

def transformar(
    X,
    parametros,
    var_ing_categoricas,
    var_ing_numericas,
    var_ing_binarias,
    config_feature_engineering=None,
    variables_no_transformar=None,
    aplicar_winsorizacion=False
):
    """
    Aplica parámetros aprendidos en Train sin recalcularlos.

    Genera columnas *_woe y conserva las variables originales.
    """

    X = X.copy()
    columnas_iniciales = X.columns.tolist()

    variables_no_transformar = list(set(
        list(variables_no_transformar or [])
        + parametros.get("variables_no_transformar", [])
    ))

    # --------------------------------------------------------
    # 1. Aplicar WoE antes de imputar las variables originales
    # --------------------------------------------------------

    for var in parametros.get("vars_woe", []):

        if var not in X.columns:
            raise ValueError(
                f"La variable '{var}' requerida para WoE no existe en X."
            )

        if var in parametros["woe_variables_numericas"]:

            categorias = _categorizar_numerica_woe(
                X[var],
                parametros["woe_cortes_numericos"][var]
            )
        else:
            categorias = X[var].copy()

        claves = categorias.map(_clave_woe)

        X[f"{var}_woe"] = (
            claves
            .map(parametros["woe_mappings"][var])
            .fillna(parametros["woe_global"].get(var, 0.0))
            .astype(float)
        )

    # --------------------------------------------------------
    # 2. Filtrar variables excluidas
    # --------------------------------------------------------

    var_ing_categoricas = [
        v for v in var_ing_categoricas
        if v not in variables_no_transformar
    ]
    var_ing_numericas = [
        v for v in var_ing_numericas
        if v not in variables_no_transformar
        and v not in parametros["vars_descartadas_nas"]
    ]
    var_ing_binarias = [
        v for v in var_ing_binarias
        if v not in variables_no_transformar
    ]

    # --------------------------------------------------------
    # 3. Imputación numéricas
    # --------------------------------------------------------

    if var_ing_numericas:

        X[var_ing_numericas] = X[var_ing_numericas].astype(float)

        vars_grupo = [
            v for v in parametros["vars_imputacion_grupo"]
            if v in var_ing_numericas
        ]
        vars_cruce = parametros["vars_cruce_imputacion"]

        if vars_grupo and vars_cruce:

            if len(vars_cruce) == 1:
                llave = X[vars_cruce[0]]
            else:
                llave = list(zip(*[X[col] for col in vars_cruce]))

            for var in vars_grupo:

                matriz = parametros["matrices_grupo"].get(var, {})
                mediana = parametros["medianas"].get(var, np.nan)

                valores_grupo = pd.Series(
                    llave, index=X.index
                ).map(matriz)

                X[var] = (
                    X[var]
                    .fillna(valores_grupo)
                    .fillna(mediana)
                )

        for var in parametros["vars_mediana_global"]:
            if var in var_ing_numericas:
                X[var] = X[var].fillna(parametros["medianas"][var])

        if aplicar_winsorizacion:

            for var, limites in parametros["limites_winsorizacion"].items():

                if var in X.columns:
                    X[var] = X[var].clip(
                        lower=limites["limite_inferior"],
                        upper=limites["limite_superior"]
                    )

    # --------------------------------------------------------
    # 4. Imputación categóricas y binarias
    # --------------------------------------------------------

    vars_cat_bin = var_ing_categoricas + var_ing_binarias

    if vars_cat_bin:
        X[vars_cat_bin] = X[vars_cat_bin].fillna(parametros["modas"])

    if parametros["binarias_validas"]:
        vars_bin = [
            v for v in parametros["binarias_validas"]
            if v in X.columns
        ]
        X[vars_bin] = X[vars_bin].astype(float)

    # --------------------------------------------------------
    # 5. One-Hot Encoding
    # --------------------------------------------------------

    variables_one_hot = parametros["variables_one_hot"]

    if variables_one_hot:

        X_cat = X[variables_one_hot].fillna(parametros["modas"])

        dummies = pd.get_dummies(
            X_cat,
            prefix=variables_one_hot,
            drop_first=True
        )

        dummies = (
            dummies
            .reindex(
                columns=parametros["columnas_dummies"],
                fill_value=0
            )
            .astype(int)
        )

        X = pd.concat([X, dummies], axis=1)

    # --------------------------------------------------------
    # 6. Feature Engineering
    # --------------------------------------------------------

    if config_feature_engineering is not None:

        nuevas_features = mdl.create_all_features(
            df=X,
            features=var_ing_numericas,
            **config_feature_engineering
        )

        X = pd.concat([X, nuevas_features], axis=1)

    # --------------------------------------------------------
    # 7. Variables creadas
    # --------------------------------------------------------
    
    X['flg_departamento_lima']=X['departamento'].isin(['LIMA'])*1
    X['flg_comercio']=X['sei_mcdo'].isin(['COMERCIO'])*1

    variables_creadas = [
        col for col in X.columns
        if col not in columnas_iniciales
    ]

    return X, variables_creadas

#---- 7.5.4) Aplicacion ----

VAR_ING_CATEGORICAS
VAR_ING_NUMERICAS
VAR_ING_BINARIAS

VARIABLES_NO_TRANSFORMAR_ING = list(dict.fromkeys(
    [var_dependiente, var_ponderacion]
        + [ 'conteo',
        'cod_instit_financiera',
        'gestion_final',
        'flg_gestion',
        'flg_exito_4',
        'cod_entidad',
        'nombre_ifi',
        'flg_desembolso_ibk_all_products',
        'tasa_min',
        'tasa_max',
        'tasa_cliente',
        'monto_cliente',
        'plazo_min',
        'plazo_max',
        'delta_deuda_final',
        'saldo_alta',
        'salprom_creditos_u03m',
        'salprom_creditos_u06m',
        'salprom_creditos_u12m',
        'avg_deuda_directa_vig_ibk_sf_u03m',
        'avg_deuda_directa_vig_ibk_sf_u06m',
        'avg_deuda_directa_vig_ibk_sf_u12m',
        'avg_deuda_directa_vig_total_sf_u03m',
        'avg_deuda_directa_vig_total_sf_u06m,'
        'avg_deuda_directa_vig_total_sf_u12m']
))

X_ini_tr = base_propension.query('base=="train"')
y_ini_tr = base_propension.query('base=="train"')[var_dependiente]

X_ini_te = base_propension.query('base=="test"')
y_ini_te = base_propension.query('base=="test"')[var_dependiente]

vars_woe = ['sow_deuda_u12m','macroperfil']

# 1. Ajuste de Parámetros (Aprendizaje en Train)
parametros_transformacion = fit_transformaciones(
    X_train=X_ini_tr,
    y_train=y_ini_tr,
    var_ing_categoricas=VAR_ING_CATEGORICAS,
    var_ing_numericas=VAR_ING_NUMERICAS,
    var_ing_binarias=VAR_ING_BINARIAS,
    vars_cruce_imputacion=['macroperfil'],     # Variables de corte
    umbral_nas_mediana_global=0.05,               # <= 5% -> Mediana global
    umbral_nas_imputacion_grupo=0.20,              # > 5% y <= 20% -> Grupo | > 20% -> Descarte
    variables_no_transformar=VARIABLES_NO_TRANSFORMAR_ING,
    max_categorias_one_hot_encoding=3,
    vars_woe=vars_woe,
    n_bins_woe=4,
    alpha_winsorizacion=0.01,
    alpha_woe=0.5
)

config_feature_engineering = {
    "use_trends": False,
    "use_volatility": False,
    "use_log": False,
    "use_ratios": False
}

# 2. Aplicación de Transformaciones
base_propension_feature, NUEVAS_VARIABLES = transformar(
    X=base_propension,
    parametros=parametros_transformacion,
    var_ing_categoricas=VAR_ING_CATEGORICAS,
    var_ing_numericas=VAR_ING_NUMERICAS,
    var_ing_binarias=VAR_ING_BINARIAS,
    variables_no_transformar=VARIABLES_NO_TRANSFORMAR_ING,
    config_feature_engineering=config_feature_engineering,
    aplicar_winsorizacion=True
)

base_propension_feature.head()

print("Nuevas Variables:")
print("\n".join([f" - {v}" for v in NUEVAS_VARIABLES]))

#~~~~ 7.6) Selección de variables ----

#---- 7.6.1) Variables a evaluar ----

VARIABLES_OMITIR_SEGMENTACION = list(
    dict.fromkeys(
        [
            var_dependiente,
            var_ponderacion
        ]
        + [ 'conteo',
        'cod_instit_financiera',
        'gestion_final',
        'flg_gestion',
        'flg_exito_4',
        'cod_entidad',
        'nombre_ifi',
        'flg_desembolso_ibk_all_products',
        'tasa_min',
        'tasa_max',
        'tasa_cliente',
        'monto_cliente',
        'plazo_min',
        'plazo_max',
        'delta_deuda_final',
        'saldo_alta',
        'salprom_creditos_u03m',
        'salprom_creditos_u06m',
        'salprom_creditos_u12m',
        'avg_deuda_directa_vig_ibk_sf_u03m',
        'avg_deuda_directa_vig_ibk_sf_u06m',
        'avg_deuda_directa_vig_ibk_sf_u12m',
        'avg_deuda_directa_vig_total_sf_u03m',
        'avg_deuda_directa_vig_total_sf_u06m,'
        'avg_deuda_directa_vig_total_sf_u12m',
        'sow_deuda_u12m',
        'sow_deuda_u12m_woe',
    ]
    )
)
print(f"Variables a omitir - framework selección ML:")
print("\n".join([f" - {v}" for v in VARIABLES_OMITIR_SEGMENTACION]))

variables_candidatas_numericas=(
        NUEVAS_VARIABLES
        +VAR_ING_NUMERICAS
        +[var for var in VAR_ING_BINARIAS if var not in parametros_transformacion['binarias_categoricas']]
)

variables_candidatas_numericas = list(
    dict.fromkeys(
        var for var in variables_candidatas_numericas
        if var not in VARIABLES_OMITIR_SEGMENTACION
    )
)

variables_forzadas=[var_ponderacion]
variables_candidatas_categoricas=[]
variables_candidatas=variables_candidatas_numericas+variables_candidatas_categoricas+variables_forzadas

print(f"Variables a evaluar - framework selección ML:")
print("\n".join([f" - {v}" for v in variables_candidatas]))

#---- 7.6.2) Selección datos ----

X_ini_tr=base_propension_feature.query('base=="train"')[variables_candidatas]
y_ini_tr=base_propension_feature.query('base=="train"')[var_dependiente]

X_ini_te=base_propension_feature.query('base=="test"')[variables_candidatas]
y_ini_te=base_propension_feature.query('base=="test"')[var_dependiente]

#---- 7.6.3) Framework de seleccion de variables por ML ----

sel_num_ini, sel_dummy_ini = mdl.seleccionar_variables_pipeline(
    X_ini_tr, y_ini_tr,
    X_ini_te, y_ini_te,
    features_num=variables_candidatas,
    features_dummy=[],
    target_type='classification',
    n_top=30,
    umbral_corr=0.70,
    vars_baneadas=[]
)

variables_arbol_ini = sel_num_ini + sel_dummy_ini

print("Variables a evaluar (con relevancia por ML) en el AD:")
print("\n".join([f" - {v}" for v in variables_arbol_ini]))

#~~~~ 7.7) Busqueda Exhaustiva: Arbol de decisión ----

#---- 7.7.1) funcion: Modelo AD Unitario ----
def _evaluar_modelo_ad_unitario(
    id_modelo, subset_vars, max_leaves, porc_leafs, usar_peso,
    base_propension, X_ini_tr, y_ini_tr, 
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
        
        leaf_ids_total = arbol.apply(base_propension[list(subset_vars)])
        df_feat = base_propension.copy()
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

#---- 7.7.2) funcion:  Grid Search ----
def ejecutar_grid_search_con_segmentos_reales(
    base_propension, 
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
            base_propension=base_propension,
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

#---- 7.7.3) Ejecución: Grid Search ----
df_grid_maestro = ejecutar_grid_search_con_segmentos_reales(
    base_propension=base_propension_feature,
    X_ini_tr=X_ini_tr,
    y_ini_tr=y_ini_tr,
    variables_disponibles=variables_arbol_ini,
    min_vars=5,
    max_vars=5,
    lista_max_leafs=[3,4,5],
    lista_porc_leafs=[0.10, 0.15],
    lista_ponderacion=[True,False],
    umbral_separacion=0.03,
    col_peso=var_ponderacion,
    period_col='periodo',
    base_col='base',
    var_dependiente=var_dependiente
)

#---- 7.7.4) Modelos Aprrobados ----
df_grid_maestro['estado_final'] = np.where(
    (df_grid_maestro['psi_estable_Mediana'] == 'Sí') & 
    (df_grid_maestro['cruces_suma'] <= 3) & 
    (df_grid_maestro['cumplimiento_sep_train'] >= 0.90),
    'SÍ SALE (Aprobado)', 
    'NO SALE (Rechazado)'
)

modelos_aprobados = df_grid_maestro[df_grid_maestro['estado_final'] == 'SÍ SALE (Aprobado)']

#____________________ 8) Modelo de sensibilidad: Data Panel ----

#~~~~ 8.1) Funciones ----

def evaluar_modelo_unitario_elasticidad(
    id_modelo,
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

        # Procesamiento: mercado
        datos_tasa_mercado = df_iter.query("flg_exito_4 not in ['exito_desembolso_ibk_ct'] and tasa_ultima.notna() ")
        
        limites_winsor = datos_tasa_mercado.groupby('cluster_modelar')['tasa_ultima'].agg(
            limite_inferior=lambda x: x.quantile(umbral_winsorizacion),
            limite_superior=lambda x: x.quantile(1 - umbral_winsorizacion)
        ).reset_index()
        
        datos_tasa_mercado = datos_tasa_mercado.merge(limites_winsor, on='cluster_modelar', how='left')
        datos_tasa_mercado['tasa_mercado_winsorizada'] = datos_tasa_mercado['tasa_ultima'].clip(
            lower=datos_tasa_mercado['limite_inferior'], 
            upper=datos_tasa_mercado['limite_superior']
        )
        
        prom_pond = lambda v, w: (
            np.average(v[v.notna() & w.notna() & (w > 0)], weights=w[v.notna() & w.notna() & (w > 0)])
            if (v.notna() & w.notna() & (w > 0)).any()
            else np.nan
        )
                
        data_panel_sensibilidad_mercado = (
            datos_tasa_mercado
            .assign(tasa_x_monto=lambda df: df['tasa_mercado_winsorizada'] * df['delta_deuda_final'])
            .groupby(['cluster_modelar', 'periodo'], observed=True)
            .agg(
                total_desembolsos_mercado=('llave_interna', 'size'),
                monto_desembolsado_mercado=('delta_deuda_final', 'sum'),
                suma_tasa_x_monto_bucket=('tasa_x_monto', 'sum'),
                tasa_mercado=('tasa_mercado_winsorizada', lambda x: prom_pond(x, datos_tasa_mercado.loc[x.index, 'delta_deuda_final']))
            )
            .reset_index()
        )
        
        # Procesamiento: sensibilidad interna
        base_propension = (
            df_iter
            .sort_values(by=['llave_interna', 'flg_exito_4'], ascending=[True, True])
            .drop_duplicates(subset=['llave_interna'], keep='first')
            .assign(indicador_exito=lambda df: (df['flg_exito_4'] == 'exito_desembolso_ibk_ct').astype(int))
        )
        
        columnas_si = ['periodo', 'llave_interna', 'cluster_modelar', 'indicador_exito', 'flg_exito_4', 'monto_min', 'monto_ofertado_ajustado', 'tasa_min', 'monto_desembolsado', 'tasa_desembolsada']
        df_oi = base_propension[columnas_si].copy()
        
        def promedio_ponderado(valores, pesos):
            mask = valores.notna() & pesos.notna() & (pesos > 0)
            if mask.any():
                return np.average(valores[mask], weights=pesos[mask])
            return np.nan

        data_panel_sensibilidad_interna = (
            df_oi
            .groupby(["cluster_modelar", "periodo"], observed=True)
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
            data_panel_sensibilidad_interna.groupby(['periodo', 'cluster_modelar'], observed=True)
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
            .groupby('cluster_modelar')
            .agg(conteo=('cluster_modelar', 'size'),
                 mat_n=('flg_mat_n', 'sum'),
                 mat_monto=('flg_mat_monto', 'sum'))
            .reset_index()
        )
        
        n_buckets_iniciales = len(buckets_val)
        buckets_val_filtrados = buckets_val.query(f'conteo == 13 and mat_monto >={meses_tol_min_materialidad} and mat_n >= {meses_tol_min_materialidad}')
        n_buckets_validos = len(buckets_val_filtrados)
        listado_val = buckets_val_filtrados.cluster_modelar.to_list()
        
        if n_buckets_validos == 0:
            return None
            
        # Panel de sensibilidad completo
        dp_sens = pd.merge(data_panel_sensibilidad_mercado, data_panel_sensibilidad_interna, how='outer', on=['cluster_modelar', 'periodo'])
        dp_sens = dp_sens.query(f"cluster_modelar in {listado_val}")
        dp_sens['spread'] = dp_sens['tasa_ofertada'] - dp_sens['tasa_mercado']
        
        dp_sens["log_efectividad"] = np.log(dp_sens["efectividad"].replace(0, np.nan))
        dp_sens["log_tasa"] = np.log(dp_sens["tasa_ofertada"])
        
        # Demeaning Simultáneo (Two-Way)
        def demeaning_sim(data, col):
            g_mean = data[col].mean()
            b_mean = data.groupby("cluster_modelar")[col].transform("mean")
            t_mean = data.groupby("periodo")[col].transform("mean")
            return data[col] - b_mean - t_mean + g_mean

        dp_sim = dp_sens.copy()
        dp_sim["log_efect_dem"] = demeaning_sim(dp_sim, "log_efectividad")
        dp_sim["log_tasa_dem"] = demeaning_sim(dp_sim, "log_tasa")
        dp_sim["spread_dem"] = demeaning_sim(dp_sim, "spread")
        
        # Modelamiento OLS por bucket
        resultados_sim = []
        for grupo in dp_sim["cluster_modelar"].unique():
            dg = dp_sim[dp_sim["cluster_modelar"] == grupo].dropna(subset=["log_efect_dem", "log_tasa_dem", "spread_dem"])
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
        df_res_sim['modelo'] = id_modelo
        
        # Métricas de resumen filtrando por p-value Y por elasticidad/sensibilidad negativa (< 0)
        sig_conj_tasa = ((df_res_sim['conj_pval_tasa'] < p_value_umbral) & 
                        (df_res_sim['conj_elast_tasa'] < 0)).sum()

        sig_conj_spread = ((df_res_sim['conj_pval_spread'] < p_value_umbral) & 
                        (df_res_sim['conj_sens_spread'] < 0)).sum()

        sig_conj_ambos = ((df_res_sim['conj_pval_tasa'] < p_value_umbral) & 
                        (df_res_sim['conj_elast_tasa'] < 0) &
                        (df_res_sim['conj_pval_spread'] < p_value_umbral) & 
                        (df_res_sim['conj_sens_spread'] < 0)).sum()

        sig_tasa_m2 = ((df_res_sim['tasa_pval'] < p_value_umbral) & 
                    (df_res_sim['tasa_elast'] < 0)).sum()

        sig_spread_m3 = ((df_res_sim['spread_pval'] < p_value_umbral) & 
                        (df_res_sim['spread_sens'] < 0)).sum()

        resumen_info = {
            "id_modelo": id_modelo,
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
        print(f"Error evaluando modeo {id_modelo}: {e}")
        return None, None

def procesar_y_evaluar_modelo(row_modelo, base_propension_feature, datos):
    """
    Función que procesa la transformación de variables y evalúa un modelo individual.
    """
    id_modelo = row_modelo['id_modelo']
    arbol_evaluado = row_modelo['_objeto_arbol']
    vars_modelo_evaluado = list(row_modelo['_subset_vars'])
    recod_evaluado = row_modelo['_recodificacion']
    
    # Aplicar árbol y recodificación
    leaf_ids = arbol_evaluado.apply(base_propension_feature[vars_modelo_evaluado])
    
    base_propension_evaluado = base_propension_feature[['llave_interna', 'monto_ofertado_ajustado']].copy()
    base_propension_evaluado['segmento_arbol'] = leaf_ids
    base_propension_evaluado['segmento_arbol_new'] = base_propension_evaluado['segmento_arbol'].map(recod_evaluado)
    
    aux_modelamiento_segmento = base_propension_evaluado[
        ['llave_interna', 'monto_ofertado_ajustado', 'segmento_arbol_new']
    ].rename(columns={'segmento_arbol_new': 'cluster_modelar'})
    
    # Merge con los datos principales
    datos_aux_panel = pd.merge(
        datos,
        aux_modelamiento_segmento,
        how='outer',
        on='llave_interna'
    )
    
    # Evaluación del modelo
    resultado = evaluar_modelo_unitario_elasticidad(
        id_modelo,
        datos_aux_panel, 
        p_value_umbral=0.05, 
        umbral_materialidad=0.03,
        periodo_inicio=202504,
        periodo_fin=202604,
        umbral_winsorizacion=0.025,
        meses_tol_min_materialidad=9
    )
    
    return resultado

#~~~~ 8.2) Ejecución ----

resultados = Parallel(n_jobs=-1,backend='loky', verbose=10)(
    delayed(procesar_y_evaluar_modelo)(
        row, 
        base_propension_feature, 
        datos
    ) for _, row in modelos_aprobados.query('n_segmentos_reales in [4,5]').copy().iterrows()
)

#~~~~ 8.3) Analisis ----

#---- 8.3.1) Funciones ----
def analizar_y_reportar_modelo_segmentacion(
    mejor_modelo, 
    base_propension_feature, 
    X_ini_tr, 
    y_ini_tr, 
    mdl, 
    oot_inicio='202602'
):
    """
    Analiza, grafica y genera todos los reportes de desempeño, estabilidad y visualizaciones 
    para un modelo de árbol específico seleccionado.
    
    Parámetros:
    -----------
    mejor_modelo : pd.Series o dict
        Fila o registro del modelo a evaluar (contiene '_objeto_arbol', '_subset_vars', etc.).
    base_propension_feature : pd.DataFrame
        Base de datos completa de características y períodos.
    X_ini_tr : pd.DataFrame
        Variables independientes de entrenamiento (Train).
    y_ini_tr : pd.Series o np.array
        Variable objetivo (target) de entrenamiento (Train).
    mdl : module o object
        Módulo personalizado que contiene las funciones de ploteo y reporte (ej. mdl.plot_...).
    oot_inicio : str
        Período de inicio para los reportes OOT (Out Of Time). Por defecto '202602'.
    
    Retorna:
    --------
    base_propension_ganador : pd.DataFrame
        Base con las predicciones y segmentos pegados.
    reporte_mensual : pd.DataFrame
        Reporte mensual de cruces por cosecha.
    estadisticos_globales : dict o pd.DataFrame
        Resumen estadístico global.
    df_psi_temporal : pd.DataFrame
        Cálculo del PSI temporal por base.
    """
    
    print("=" * 65)
    print("🏆 ¡ANALIZANDO MODELO SELECCIONADO! 🚀")
    print("=" * 65)
    print(f"• ID del Modelo: {mejor_modelo['id_modelo']}")
    print(f"• Estado: {mejor_modelo['estado_final']}")
    print(f"• Variables Candidatas: {mejor_modelo['variables_usadas']}")
    print(f"• Variables Efectivas (Usadas): {mejor_modelo['variables_efectivas_nombres']}")
    print(f"• Max Leaf Nodes: {mejor_modelo['max_leaf_nodes_config']}")
    print(f"• Suma de Cruces: {mejor_modelo['cruces_suma']}")
    print(f"• PSI Máximo Estable: {mejor_modelo['psi_estable_Máximo']}")
    print("=" * 65)

    # PASO 3.1: VISUALIZAR LOS CORTES DEL ÁRBOL
    arbol_ganador = mejor_modelo['_objeto_arbol']
    vars_modelo = mejor_modelo['_subset_vars']  
    
    # 1. Ver reglas en formato de texto por consola
    print("\n🌳 REGLAS Y CORTES DEL ÁRBOL SELECCIONADO:")
    print(export_text(arbol_ganador, feature_names=list(vars_modelo)))
    
    # 2. Graficar el árbol visualmente
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

    # PASO 4: REPLICACIÓN AUTOMÁTICA Y PREDICCIÓN DE TRAIN
    recod_ganadora = mejor_modelo['_recodificacion']
    
    # Aplicar el árbol usando el subset exacto de variables con el que fue entrenado
    leaf_ids_ganador = arbol_ganador.apply(base_propension_feature[list(vars_modelo)])
    base_propension_ganador = base_propension_feature.copy()
    base_propension_ganador['segmento_arbol'] = leaf_ids_ganador
    base_propension_ganador['segmento_arbol_new'] = base_propension_ganador['segmento_arbol'].map(recod_ganadora)
    
    # 🎯 ADJUNTAR LA PREDICCIÓN/RATIO DE TRAIN A CADA BUCKET EN EL MODELO GENERAL
    train_con_segmentos = X_ini_tr.copy()
    train_con_segmentos['segmento_arbol'] = arbol_ganador.apply(X_ini_tr[list(vars_modelo)])
    train_con_segmentos['segmento_arbol_new'] = train_con_segmentos['segmento_arbol'].map(recod_ganadora)
    train_con_segmentos['indicador_exito'] = y_ini_tr.values
    
    dict_ratio_train = train_con_segmentos.groupby('segmento_arbol_new')['indicador_exito'].mean().to_dict()
    base_propension_ganador['prediccion'] = base_propension_ganador['segmento_arbol_new'].map(dict_ratio_train)
    
    print("📌 ¡Predicción/Ratio de train pegada exitosamente a cada bucket en 'base_propension_ganador' (Columna: 'ratio_train_esperado')!")

    print("\n📊 Generando reportes y gráficos finales con el modelo seleccionado...\n")
    
    # Ejecutar visualizaciones finales de forma automática
    mdl.plot_escalera_tasas_global(base_propension_ganador, segment_col='segmento_arbol_new', target_col='indicador_exito')
    mdl.plot_escalera_tasas_global(base_propension_ganador.query('base=="train"'), segment_col='segmento_arbol_new', target_col='indicador_exito')
    mdl.plot_barras_comparacion_base(base_propension_ganador, segment_col='segmento_arbol_new', target_col='indicador_exito', base_col='base')
    mdl.plot_todas_series_juntas(base_propension_ganador, period_col='periodo', segment_col='segmento_arbol_new', target_col='indicador_exito')
    mdl.plot_evolucion_segmentos_grid(base_propension_ganador, period_col='periodo', segment_col='segmento_arbol_new', target_col='indicador_exito', base_col='base', ncols=3)
    mdl.plot_materialidad_global_temporal(base_propension_ganador, period_col='periodo', segment_col='segmento_arbol_new')
    
    mdl.plot_materialidad_y_precancelacion(
        base_propension_ganador,
        period_col='periodo',
        segment_col='segmento_arbol_new',
        target_col='indicador_exito',
        oot_inicio=oot_inicio
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

    df_psi_temporal, stats_bases = mdl.calcular_psi_temporal_por_base(
        base_propension_ganador, 
        period_col='periodo', 
        segment_col='segmento_arbol_new', 
        base_col='base', 
        base_ref_nombre='train'
    )

    print("--- Estadísticos Resumen del PSI Temporal por Base ---")
    print(stats_bases)

    print("\n--- Muestra de la Serie Temporal por Base ---")
    print(df_psi_temporal)

    mdl.plot_precancelacion_barras_agrupadas(
        base_propension_ganador,
        period_col='periodo',
        segment_col='segmento_arbol_new',
        target_col='indicador_exito',
        oot_inicio=oot_inicio,
        mostrar_promedios=False
    )
    
    return base_propension_ganador, reporte_mensual, estadisticos_globales, df_psi_temporal

#---- 8.3.2) Analisis ----

resultados_validos = [res for res in resultados if res[0] is not None]

df_detallado_global = pd.concat([res[0] for res in resultados_validos], ignore_index=True)

df_resumen_global = pd.DataFrame([res[1] for res in resultados_validos])

df_resumen_global[['n_buckets_iniciales','cant_sig_modelo2_tasa']].value_counts().reset_index()

df_resumen_global.n_buckets_iniciales.value_counts()

modelo_segmentacion_evaluado = modelos_aprobados.query('id_modelo==416').iloc[0]
    
base_ganadora, rep_mensual, est_globales, psi_temporal = analizar_y_reportar_modelo_segmentacion(
    mejor_modelo=modelo_segmentacion_evaluado,
    base_propension_feature=base_propension_feature,
    X_ini_tr=X_ini_tr,
    y_ini_tr=y_ini_tr,
    mdl=mdl,
    oot_inicio='202602'
)
