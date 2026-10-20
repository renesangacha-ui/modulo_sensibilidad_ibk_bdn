############################### Modelo de Propensión ####################################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tModelo de Propensión')
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


import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from statsmodels.formula.api import ols

import xgboost as xgb
import shap
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm

sys.path.append("..")
import src.modelamiento.funcionesmodelamiento as mdl
import src.graficos.graficos as graficos
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
     'cluster_3_sow_no_nulo': 'Grandes por fidelizar',
     'cluster_4_sow_no_nulo': 'Grandes por fidelizar',
     'cluster_5_sow_no_nulo': 'Medianos por fidelizar',
     'cluster_6_sow_no_nulo': 'Medianos por fidelizar',     
     'cluster_1_nulo': 'Sin historial crediticio',
     'cluster_2_nulo': 'Sin historial crediticio',
     'cluster_3_nulo': 'Sin historial crediticio'
}

datos['macroperfil'] = datos['cluster'].map(CLUSTER_MACROPERFILES).fillna(datos['cluster'])
datos[['macroperfil','cluster']].value_counts().reset_index()

datos['nombre_ifi']=datos['cod_instit_financiera'].map({1.0:'BCP', 6.0: 'BBVA', 4.0:'SCOTIA', 123.0:'MI BANCO'}).fillna('OTROS')

#~~~~ 6.3) Nuevas Variables ----

datos['tasa_cliente']=np.where(datos['tasa_desembolsada'].isna(),datos['tasa_min'],datos['tasa_desembolsada'])
datos['monto_cliente']=np.where(datos['monto_desembolsado'].isna(),datos['monto_min'],datos['monto_desembolsado'])

#____________________ 7) Ingenieria y Selección de Variables ----

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
        'monto_min',
        'monto_max',
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

#____________________ 8) Modelamiento: Propensión ----

#~~~~ 8.1) Creación Variables: Temporalidad ----

base_propension_feature['periodo_str'] = base_propension_feature['periodo'].astype(str)
base_propension_feature['mes'] = base_propension_feature['periodo_str'].str[-2:].astype(int)
base_propension_feature['q2'] = base_propension_feature['mes'].isin([4, 5, 6]).astype(int)
base_propension_feature['q3'] = base_propension_feature['mes'].isin([7, 8, 9]).astype(int)
base_propension_feature['q4'] = base_propension_feature['mes'].isin([10,11,12]).astype(int)

#~~~~ 8.2) Selección de Variables Final ----

variables_candidatas = variables_arbol_ini + ['tasa_cliente','monto_cliente']

listado_exclusion=['deuda_directa_vig_total_sf_u03m','monto_min','nivel_ventas',
                    'pasivosprom_u6m','pasivosprom_u12m', 'deuda_directa_vig_ibk_sf_u03m',
                    'suma_uai_u12m','avg_deuda_directa_vig_total_sf_u12m', 'monto_ofertado_ajustado']

variables_candidatas=[var for var in variables_candidatas if var not in listado_exclusion]

#~~~~ 8.3) Backward ----

#---- 8.3.1) Funcion ----
def seleccionar_logit_significativo(X_train, y_train, lista_variables, umbral_pvalue=0.05):
    
    vars_actuales = list(lista_variables)
    
    mean_train = X_train[vars_actuales].mean()
    std_train = X_train[vars_actuales].std().replace(0, 1)
    
    print(f"• Variables candidatas iniciales: {len(vars_actuales)}")
    print(f"• Umbral de p-value máximo permitido: {umbral_pvalue}")
    
    iteracion = 1
    while len(vars_actuales) > 0:
        X_scaled = (X_train[vars_actuales] - mean_train[vars_actuales]) / std_train[vars_actuales]
        X_const = sm.add_constant(X_scaled, has_constant='add')
        y = np.asarray(y_train)
        
        modelo = sm.Logit(y, X_const)
        modelo_fit = modelo.fit(disp=False)
        
        p_values = modelo_fit.pvalues.drop('const', errors='ignore')
        peor_var = p_values.idxmax()
        peor_pval = p_values.max()
        
        if peor_pval > umbral_pvalue:
            vars_actuales.remove(peor_var)
            iteracion += 1
        else:
            break
            
    if len(vars_actuales) == 0:
        raise ValueError("⚠️ Se eliminaron todas las variables. Ninguna cumplió con el umbral.")
        
    # Ajuste final con las sobrevivientes
    X_final_scaled = (X_train[vars_actuales] - mean_train[vars_actuales]) / std_train[vars_actuales]
    X_final_const = sm.add_constant(X_final_scaled)
    modelo_final = sm.Logit(np.asarray(y_train), X_final_const).fit(disp=True)
    
    print("\n" + "=" * 65)
    print(f"🏆 MODELO FINAL OBTENIDO ({len(vars_actuales)} variables significativas)")
    print("=" * 65)
    print(modelo_final.summary())
    print("=" * 65)
    
    return modelo_final, vars_actuales, mean_train, std_train

#---- 8.3.2) Estimación ----

df_train_raw = base_propension_feature.query('base=="train"')

logit_propension, variables_significativas, mean_train, std_train = seleccionar_logit_significativo(
    X_train=df_train_raw[variables_candidatas],
    y_train=df_train_raw[var_dependiente],
    lista_variables=variables_candidatas,
    umbral_pvalue=0.05
)

#~~~~ 8.4) Validación ----

#---- 8.4.1) Gini ----

def evaluar_gini_base_individual(modelo_fit, lista_vars, X_base, y_base, mean_tr, std_tr, nombre_base="Base"):

    X_scaled = (X_base[lista_vars].copy() - mean_tr[lista_vars]) / std_tr[lista_vars]
    X_const = sm.add_constant(X_scaled, has_constant='add')
    y_arr = np.asarray(y_base)
    
    y_pred = modelo_fit.predict(X_const)
    
    auc = roc_auc_score(y_arr, y_pred)
    gini = (2 * auc) - 1
    
    print(f"📊 Evaluación {nombre_base:<6} | Registros: {len(y_arr):<6} | AUC: {auc:.4f} | Gini: {gini:.4f}")
    
    # Curva ROC
    plt.figure(figsize=(5, 4), dpi=150)
    fpr, tpr, _ = roc_curve(y_arr, y_pred)
    plt.plot(fpr, tpr, label=f'{nombre_base} (Gini = {gini:.4f})', color='royalblue', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Azar')
    plt.xlabel('1 - Especificidad', fontsize=9)
    plt.ylabel('Sensibilidad', fontsize=9)
    plt.title(f'Curva ROC - {nombre_base}', fontsize=10, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    return {'base': nombre_base, 'auc': auc, 'gini': gini}

res_train = evaluar_gini_base_individual(logit_propension, variables_significativas, base_propension_feature.query('base=="train"'), base_propension_feature.query('base=="train"')[var_dependiente], mean_train, std_train, "Train")
res_test  = evaluar_gini_base_individual(logit_propension, variables_significativas, base_propension_feature.query('base=="test"'),  base_propension_feature.query('base=="test"')[var_dependiente],  mean_train, std_train, "Test")
res_oot   = evaluar_gini_base_individual(logit_propension, variables_significativas, base_propension_feature.query('base=="oot"'),   base_propension_feature.query('base=="oot"')[var_dependiente],   mean_train, std_train, "OOT")

#---- 8.4.1) Evolutivo Temporal ----

X_global_scaled = (base_propension_feature[variables_significativas] - mean_train[variables_significativas]) / std_train[variables_significativas]
base_propension_feature['prob_aceptacion_pred'] = logit_propension.predict(sm.add_constant(X_global_scaled))


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Agrupación base
df_temporal = base_propension_feature.groupby('periodo').agg(
    tasa_exito=(var_dependiente, 'mean'),
    registros=(var_dependiente, 'size'),
    prob_aceptacion_pred=('prob_aceptacion_pred', 'mean')
).reset_index()

# 2. Ordenamiento cronológico seguro
df_temporal['periodo'] = df_temporal['periodo'].astype(str)
df_temporal = df_temporal.sort_values('periodo').reset_index(drop=True)

# 3. Mapeo de YYYYMM a formato "Ene '25"
meses_dict = {
    '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr', 
    '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago', 
    '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
}

def formatear_periodo(p_str):
    p_str = str(p_str).strip()
    if len(p_str) == 6:
        anio = p_str[2:4]
        mes = p_str[4:6]
        return f"{meses_dict.get(mes, mes)} '{anio}"
    return p_str

df_temporal['periodo_label'] = df_temporal['periodo'].apply(formatear_periodo)

# Escalamos a porcentaje (0 a 100)
df_temporal['tasa_exito_pct'] = df_temporal['tasa_exito'] * 100
df_temporal['prob_aceptacion_pred_pct'] = df_temporal['prob_aceptacion_pred'] * 100

# Figura con proporciones óptimas para visualización ejecutiva
fig, ax1 = plt.subplots(figsize=(13, 6), dpi=150)

# 4. Eje secundario para la materialidad (Leads Oferta Relevante)
ax2 = ax1.twinx()
max_reg = df_temporal['registros'].max()
ax2.set_ylim(0, max_reg * 1.15)

barras = ax2.bar(
    df_temporal['periodo_label'], 
    df_temporal['registros'], 
    color='#B0B0B0', 
    alpha=0.30, 
    width=0.55, 
    label='Leads Oferta Relevante'
)
ax2.set_ylabel('Volumen (N Leads)', fontsize=10, color='#666666')
ax2.tick_params(axis='y', labelcolor='#666666')
ax2.grid(False)

# 5. Anotar el número $n$ ABAJO de las barritas
for bar, n_val in zip(barras, df_temporal['registros']):
    n_str = f"{n_val:,.0f}" if n_val < 10000 else f"{n_val*1e-3:.1f}k"
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0, 
        max_reg * 0.03,  
        n_str, 
        ha='center', 
        va='bottom', 
        fontsize=7.5, 
        color='#555555', 
        alpha=0.9
    )

# --- 6. MARCA Y SOMBRA OOT (Out-Of-Time) ---
# Modifica este string al mes exacto donde inicia tu OOT (ej: "Sep '25", "Oct '25", etc.)
mes_oot_label = "Feb '26" 

if mes_oot_label in df_temporal['periodo_label'].values:
    idx_oot = df_temporal[df_temporal['periodo_label'] == mes_oot_label].index[0]
    
    # Sombra que cubre desde el mes OOT hasta el final
    ax1.axvspan(idx_oot - 0.5, len(df_temporal) - 0.5, color='#1D4290', alpha=0.06, zorder=1, label='Periodo OOT')
    
    # Línea vertical punteada
    # ax1.axvline(x=idx_oot, color='#1D4290', linestyle='--', linewidth=1.5, zorder=6)
    
    # Etiqueta flotante OOT
    ax1.text(
        idx_oot, 
        max(df_temporal['tasa_exito_pct'].max(), df_temporal['prob_aceptacion_pred_pct'].max()) * 1.12, 
        ' OOT ', 
        ha='center', 
        va='bottom', 
        fontsize=8.5, 
        fontweight='bold', 
        color='#1D4290',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#1D4290', alpha=0.9)
    )

# 7. Eje principal: Línea azul para Probabilidad Predicha
ax1.plot(
    df_temporal['periodo_label'], 
    df_temporal['prob_aceptacion_pred_pct'], 
    marker='s', 
    markersize=5,
    label='Conversión Predicha', 
    color='#1D4290', 
    linestyle='-', 
    lw=2,
    zorder=4
)

# 8. Eje principal: Puntos verdes para Tasa Real de Conversión
ax1.scatter(
    df_temporal['periodo_label'], 
    df_temporal['tasa_exito_pct'], 
    label='Conversión Observada', 
    color='#27D75A', 
    s=65, 
    zorder=5
)

# Límites y formato del eje Y principal
max_val = max(df_temporal['tasa_exito_pct'].max(), df_temporal['prob_aceptacion_pred_pct'].max())
ax1.set_ylim(0, max_val * 1.25)
ax1.set_ylabel('Conversión (%)', fontsize=10)

# 9. Storytelling Ejecutivo y Grid limpio (SOLO horizontal)
ax1.set_title('Evolutivo de la Conversión de Oferta Relevante\nPredicho vs Observado', fontsize=12, fontweight='bold', pad=15)
ax1.set_xlabel('Período de Evaluación', fontsize=10, labelpad=8)
ax1.set_xticks(range(len(df_temporal['periodo_label'])))
ax1.set_xticklabels(df_temporal['periodo_label'], rotation=90, fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.3)  # Sin líneas verticales

# 10. Leyendas unificadas en la PARTE INFERIOR del gráfico
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines_1 + lines_2, 
    labels_1 + labels_2, 
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.16), 
    ncol=4,                      # Organizado en 4 columnas para que quepa perfecto el OOT
    frameon=True, 
    facecolor='white', 
    edgecolor='lightgray',
    framealpha=0.95,
    fontsize=9
)

plt.tight_layout()
plt.show()


#---- 8.4.1) Aperturas ----

import matplotlib.pyplot as plt
import pandas as pd

df_plot = (
    base_propension_feature
    .groupby(['periodo', 'macroperfil'])
    .agg(
        desembolsos=(var_dependiente,'sum'),
        tasa_exito=(var_dependiente, 'mean'),
        prob_aceptacion_pred=('prob_aceptacion_pred', 'mean')
    )
    .reset_index()
)



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_observado_vs_predicho(
    base,
    var_dependiente,
    columna_periodo='periodo',
    columna_grupo='macroperfil',
    columna_predicha='prob_aceptacion_pred',
    titulo_base='Evolutivo de la Conversión de Oferta Relevante\nPor Macroperfil Comercial',
    mes_oot=None,               # Parámetro para la marca OOT (ej: "202509" o "Sep '25")
    plots_por_figura=3          # Cantidad máxima de subplots por cada imagen/figura
):
    # 1. Agregación base
    df_plot = (
        base
        .groupby([columna_periodo, columna_grupo], dropna=False)
        .agg(
            n=(var_dependiente, 'size'),
            tasa_observada=(var_dependiente, 'mean'),
            tasa_predicha=(columna_predicha, 'mean')
        )
        .reset_index()
    )

    df_plot[columna_periodo] = df_plot[columna_periodo].astype(str)
    df_plot = df_plot.sort_values(columna_periodo)

    # 2. Mapeo inteligente de YYYYMM a formato "Ene '25"
    meses_dict = {
        '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr', 
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago', 
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
    }

    def formatear_periodo(p_str):
        p_str = str(p_str).strip()
        if len(p_str) == 6:
            anio = p_str[2:4]
            mes = p_str[4:6]
            return f"{meses_dict.get(mes, mes)} '{anio}"
        return p_str

    df_plot['periodo_label'] = df_plot[columna_periodo].apply(formatear_periodo)
    
    mes_oot_label = None
    if mes_oot:
        mes_oot_label = formatear_periodo(str(mes_oot)) if len(str(mes_oot)) == 6 else str(mes_oot)

    grupos = df_plot[columna_grupo].drop_duplicates().tolist()

    # 3. Paginación automática en bloques según 'plots_por_figura'
    for i in range(0, len(grupos), plots_por_figura):
        chunk_grupos = grupos[i:i + plots_por_figura]
        
        fig, axes = plt.subplots(
            len(chunk_grupos), 1,
            figsize=(13, 3.8 * len(chunk_grupos)),
            sharex=True,
            squeeze=False
        )

        for ax, grupo in zip(axes.ravel(), chunk_grupos):
            df = df_plot[df_plot[columna_grupo].eq(grupo)].sort_values(columna_periodo)
            x_indices = range(len(df))

            # Eje secundario para materialidad (Leads Oferta Relevante)
            ax_bar = ax.twinx()
            max_reg = df['n'].max() if len(df) > 0 else 1
            ax_bar.set_ylim(0, max_reg * 1.15)

            barras = ax_bar.bar(
                x_indices, df['n'],
                color='#B0B0B0', alpha=0.30, width=0.55,
                zorder=1, label='Leads Oferta Relevante'
            )
            ax_bar.set_ylabel('Volumen (N Leads)', color='#666666', fontsize=9)
            ax_bar.tick_params(axis='y', colors='#666666', labelsize=8)
            ax_bar.grid(False)

            # Anotar el número $n$ ABAJO de las barritas
            for bar, n_val in zip(barras, df['n']):
                n_str = f"{n_val:,.0f}" if n_val < 10000 else f"{n_val*1e-3:.1f}k"
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2.0, 
                    max_reg * 0.03,  
                    n_str, 
                    ha='center', 
                    va='bottom', 
                    fontsize=7, 
                    color='#555555', 
                    alpha=0.9
                )

            ax.set_zorder(2)
            ax.patch.set_visible(False)

            # --- SOLO SOMBRA Y TAG OOT (Sin línea vertical) ---
            if mes_oot_label and mes_oot_label in df['periodo_label'].values:
                pos_oot = list(df['periodo_label']).index(mes_oot_label)
                
                # Sombra de fondo OOT
                ax.axvspan(pos_oot - 0.5, len(df) - 0.5, color='#1D4290', alpha=0.06, zorder=1)
                
                # Tag flotante OOT en la parte superior
                max_tasa_y = max(df['tasa_observada'].max(), df['tasa_predicha'].max()) * 100
                ax.text(
                    pos_oot, 
                    max_tasa_y * 1.12, 
                    ' OOT ', 
                    ha='center', 
                    va='bottom', 
                    fontsize=8, 
                    fontweight='bold', 
                    color='#1D4290',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#1D4290', alpha=0.9),
                    zorder=6
                )

            # Línea de probabilidad predicha (Azul corporativo)
            ax.plot(
                x_indices, df['tasa_predicha'] * 100,
                color='#1D4290', linewidth=2,
                marker='s', markersize=4,
                label='Conversión Predicha', zorder=4
            )
            
            # Puntos de tasa real observada (Verde vibrante)
            ax.scatter(
                x_indices, df['tasa_observada'] * 100,
                color='#27D75A', edgecolors='white',
                linewidths=0.7, s=55,
                label='Conversión Observada', zorder=5
            )

            ax.set_title(f'Segmento / {columna_grupo.capitalize()}: {grupo}', loc='left', fontsize=11, fontweight='bold', color='#222222')
            ax.set_ylabel('Conversión (%)', fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            
            # Límites dinámicos para dar espacio al tag OOT
            max_val = max(df['tasa_observada'].max(), df['tasa_predicha'].max()) * 100
            ax.set_ylim(0, max_val * 1.25)
            
            # Grid limpio SOLO horizontal
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # Leyenda por subplot
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_bar.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='upper left', ncol=3, frameon=True, facecolor='white', edgecolor='lightgray', framealpha=0.9, fontsize=8)

        # Configuración eje X compartido en el último panel de la figura actual
        axes[-1, 0].set_xticks(list(x_indices))
        axes[-1, 0].set_xticklabels(df['periodo_label'], rotation=90, fontsize=9)
        axes[-1, 0].set_xlabel('Período de Evaluación', fontsize=10, labelpad=8)

        # Título de la página/bloque actual
        parte_num = (i // plots_por_figura) + 1
        total_partes = (len(grupos) + plots_por_figura - 1) // plots_por_figura
        titulo_fig = f"{titulo_base} " if total_partes > 1 else titulo_base
        
        fig.suptitle(titulo_fig, fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    return df_plot


df_plot = plot_observado_vs_predicho(
    base=base_propension_feature,
    var_dependiente=var_dependiente,
    columna_grupo='macroperfil',
    mes_oot='202602',
    plots_por_figura=2
)
