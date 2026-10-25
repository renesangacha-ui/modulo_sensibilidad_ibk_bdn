# =============================================================================

# ELASTICIDAD CON PANEL DATA + FIXED EFFECTS

# =============================================================================

# pip install pandas numpy linearmodels statsmodels matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from statsmodels.formula.api import ols
import warnings
#warnings.filterwarnings(“ignore”)

# =============================================================================

# 1. ESTRUCTURA ESPERADA DE LA DATA

# =============================================================================

# 

# grupo               | codmes  | tasa_ofertada | tasa_desembolsada | conversion

# ––––––––––|———|—————|—————––|————

# alto_monto_bajo_pd  | 2024-01 | 0.18          | 0.17              | 0.72

# alto_monto_bajo_pd  | 2024-02 | 0.19          | 0.18              | 0.68

# bajo_monto_alto_pd  | 2024-01 | 0.24          | 0.23              | 0.45

# 

# REGLAS:

# - ‘grupo’         : string que identifica el segmento (monto x PD)

# - ‘codmes’        : formato YYYY-MM, UN registro por grupo-mes (sin duplicados)

# - ‘tasa_ofertada’ : en decimales (0.18, NO 18)

# - ‘conversion’    : en decimales (0.72, NO 72)

# =============================================================================

# 2. DATA DE EJEMPLO — reemplaza con tu pd.read_excel() o pd.read_csv()

# =============================================================================

np.random.seed(42)

grupos = [
    "alto_monto_bajo_pd",
    "alto_monto_medio_pd",
    "medio_monto_bajo_pd",
    "medio_monto_medio_pd",
    "bajo_monto_bajo_pd",
    "bajo_monto_alto_pd",
]

meses = [
    "2024-01",
    "2024-02",
    "2024-03",
    "2024-04",
    "2024-05",
    "2024-06",
]

tasas_base = {
    "alto_monto_bajo_pd":   0.16,
    "alto_monto_medio_pd":  0.19,
    "medio_monto_bajo_pd":  0.20,
    "medio_monto_medio_pd": 0.23,
    "bajo_monto_bajo_pd":   0.22,
    "bajo_monto_alto_pd":   0.27,
}

conv_base = {
    "alto_monto_bajo_pd":   0.75,
    "alto_monto_medio_pd":  0.65,
    "medio_monto_bajo_pd":  0.70,
    "medio_monto_medio_pd": 0.55,
    "bajo_monto_bajo_pd":   0.60,
    "bajo_monto_alto_pd":   0.40,
}

elasticidades_reales = {
    "alto_monto_bajo_pd":   -0.3,
    "alto_monto_medio_pd":  -0.6,
    "medio_monto_bajo_pd":  -0.5,
    "medio_monto_medio_pd": -1.1,
    "bajo_monto_bajo_pd":   -0.8,
    "bajo_monto_alto_pd":   -1.4,
}

registros = []

for grupo in grupos:
    for mes in meses:
        variacion = np.random.choice(
        [-0.02, -0.01, 0, 0.01, 0.02, 0.03],
        p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.10]
        )
        tasa_of = tasas_base[grupo] + variacion
        e = elasticidades_reales[grupo]
        cambio_rel = (tasa_of - tasas_base[grupo]) / tasas_base[grupo]
        conv = conv_base[grupo] * (1 + e * cambio_rel)
        conv = np.clip(conv + np.random.normal(0, 0.01), 0.05, 0.99)
        tasa_desemb = tasa_of - np.random.uniform(0.005, 0.015)
        registros.append({
            "grupo": grupo,
            "codmes": mes,
            "tasa_ofertada": round(tasa_of, 4),
            "tasa_desembolsada": round(tasa_desemb, 4),
            "conversion": round(conv, 4),
        })


df = pd.DataFrame(registros)

# =============================================================================

# CARGA TU DATA REAL ASI (descomenta):

# =============================================================================

# df = pd.read_excel(“tu_archivo.xlsx”)

# df = pd.read_csv(“tu_archivo.csv”)

# 

# Si tus tasas estan en porcentaje (ej: 18.5 en vez de 0.185):

# df[“tasa_ofertada”]    = df[“tasa_ofertada”] / 100

# df[“tasa_desembolsada”] = df[“tasa_desembolsada”] / 100

# df[“conversion”]       = df[“conversion”] / 100

# =============================================================================

# 3. VALIDACION DE LA DATA

# =============================================================================

print("=" * 60)
print("PASO 1: VALIDACION")
print("=" * 60)

dup = df.duplicated(subset=["grupo", "codmes"]).sum()

nulos = df.isnull().sum().sum()

fuera = (
    (
        (df["conversion"] < 0)
        | (df["conversion"] > 1)
    )
).sum()

print(
    f"Duplicados grupo-mes    : {dup}  "
    f"{'OK' if dup == 0 else 'ERROR — revisar'}"
)

print(
    f"Valores nulos           : {nulos}  "
    f"{'OK' if nulos == 0 else 'ERROR — revisar'}"
)

print(
    f"Conversion fuera [0,1]  : {fuera}  "
    f"{'OK' if fuera == 0 else 'ERROR — revisar'}"
)

print(f"Grupos                  : {df['grupo'].nunique()}")
print(f"Meses                   : {df['codmes'].nunique()}")
print(f"Total filas             : {len(df)}")

print(f"\nMuestra:\n{df.head(8).to_string(index=False)}")

# =============================================================================

# 4. TRANSFORMACIONES

# =============================================================================

df["log_conversion"] = np.log(df["conversion"])

df["log_tasa_ofertada"] = np.log(
    df["tasa_ofertada"]
)

df["log_tasa_desembolsada"] = np.log(
    df["tasa_desembolsada"]
)

df["codmes_dt"] = pd.to_datetime(
    df["codmes"]
)

# =============================================================================

# 5. EXPLORACION — variacion de tasa por grupo ANTES de modelar

# =============================================================================

print("\n" + "=" * 60)
print("PASO 2: VARIACION DE TASA POR GRUPO")
print("=" * 60)

print(
    "(Grupos con poca variacion van a dar "
    "elasticidades poco confiables)"
)

print()

var_tasa = (
    df.groupby("grupo")["tasa_ofertada"]
    .agg(["min", "max", "std"])
    .assign(
        rango=lambda x: x["max"] - x["min"]
    )
    .round(4)
    .sort_values("rango", ascending=False)
)

var_tasa["rango_pp"] = (
    var_tasa["rango"] * 100
).round(2)

print(var_tasa.to_string())

UMBRAL_PP = 0.01

grupos_confiables = (
    var_tasa[var_tasa["rango"] >= UMBRAL_PP]
    .index
    .tolist()
)

grupos_poca_var = (
    var_tasa[var_tasa["rango"] < UMBRAL_PP]
    .index
    .tolist()
)

print(
    f"\nGrupos con variacion suficiente "
    f"(>=1pp): {grupos_confiables}"
)

print(
    f"Grupos con poca variacion (<1pp)       : "
    f"{grupos_poca_var}"
)

# =============================================================================

# 6. MODELO PRINCIPAL — Two-Way Fixed Effects (grupo + mes) pooled

# =============================================================================

# Este es el modelo correcto cuando tienes pocos meses por grupo.

# Usa TODOS los grupos juntos con efectos fijos de grupo y de mes.

# Identifica la elasticidad a partir de la variacion DENTRO de cada grupo.

# 

# Modelo: log(conv_it) = β * log(tasa_it) + α_i + δ_t + ε_it

# α_i = efecto fijo por grupo (diferencias estructurales entre grupos)

# δ_t = efecto fijo por mes  (shocks macro/estacionalidad compartidos)

# β   = elasticidad global

print("\n" + "=" * 60)
print("PASO 3: REGRESION GLOBAL (Two-Way FE, todos los grupos)")
print("=" * 60)

print(
    "Modelo: log(conversion) ~ "
    "log(tasa_ofertada) + FE_grupo + FE_mes"
)

print()

df_panel = df.copy()

df_panel = df_panel.set_index(
    ["grupo", "codmes_dt"]
)

mod_global = PanelOLS(
    dependent=df_panel["log_conversion"],
    exog=df_panel[["log_tasa_ofertada"]],
    entity_effects=True,
    time_effects=True,
    drop_absorbed=True,
)

res_global = mod_global.fit(
    cov_type="clustered",
    cluster_entity=True
)

e_global = res_global.params[
    "log_tasa_ofertada"
]

pval_global = res_global.pvalues[
    "log_tasa_ofertada"
]

r2_global = res_global.rsquared

print(f"Elasticidad global : {e_global:.4f}")

print(
    f"P-valor            : {pval_global:.4f}  "
    f"{'(significativa)' if pval_global < 0.05 else '(NO significativa)'}"
)

print(f"R2 within          : {r2_global:.4f}")

print()

print("Interpretacion:")

print(
    f"  Un aumento de 10% en tasa ofertada "
    f"se asocia con un cambio de {e_global*10:.1f}% "
    f"en conversion"
)

# =============================================================================

# 7. ELASTICIDAD POR GRUPO — within-group demeaning manual

# =============================================================================

# Problema con 6 meses: si hacemos OLS por grupo con dummies de mes,

# consumimos todos los grados de libertad y no hay inferencia.

# Solucion: demeaning manual (equivalente a FE) + pooled OLS por grupo.

# 

# Demeaning: restamos la media del grupo-mes a cada variable

# Esto elimina el efecto fijo de tiempo sin gastar grados de libertad en dummies.
print("\n" + "=" * 60)
print("PASO 4: ELASTICIDAD POR GRUPO (within-group demeaning)")
print("=" * 60)

print(
    "Nota: con pocos meses, tratar estos valores "
    "como PROXIES direccionales,"
)

print(
    "no como estimados causales precisos.\n"
)

# Demeaning por mes (dentro de todo el panel)

df["log_tasa_dm"] = (
    df["log_tasa_ofertada"]
    - df.groupby("codmes")["log_tasa_ofertada"]
    .transform("mean")
)

df["log_conv_dm"] = (
    df["log_conversion"]
    - df.groupby("codmes")["log_conversion"]
    .transform("mean")
)

# Demeaning por grupo (within)

df["log_tasa_within"] = (
    df["log_tasa_dm"]
    - df.groupby("grupo")["log_tasa_dm"]
    .transform("mean")
)

df["log_conv_within"] = (
    df["log_conv_dm"]
    - df.groupby("grupo")["log_conv_dm"]
    .transform("mean")
)

resultados = []

for grupo in df["grupo"].unique():

    dg = df[df["grupo"] == grupo].copy()

    n = len(dg)

    # Variacion de la tasa demeaned:
    # si es casi cero, no hay identificacion

    var_within = dg["log_tasa_within"].var()

    try:

        # OLS simple sobre variables ya demeaned
        # (equivale a FE sin gastar grados de libertad)

        m = ols(
            "log_conv_within ~ log_tasa_within - 1",
            data=dg
        ).fit()

        e = m.params["log_tasa_within"]

        pval = m.pvalues["log_tasa_within"]

        r2 = m.rsquared

        rango = (
            dg["tasa_ofertada"].max()
            - dg["tasa_ofertada"].min()
        )

        resultados.append({
            "grupo": grupo,
            "elasticidad": round(e, 4),
            "pvalor": round(pval, 4),
            "significativo": pval < 0.10,
            "r2": round(r2, 3),
            "rango_pp": round(rango * 100, 2),
            "var_within": round(var_within, 6),
            "confiable": (
                grupo in grupos_confiables
                and var_within > 0.0001
            ),
        })

    except Exception as ex:

        print(f"  {grupo}: error -> {ex}")

df_res = (
    pd.DataFrame(resultados)
    .sort_values("elasticidad")
)

print(
    df_res[
        [
            "grupo",
            "elasticidad",
            "pvalor",
            "significativo",
            "r2",
            "rango_pp",
            "confiable",
        ]
    ].to_string(index=False)
)

# =============================================================================

# 8. CLASIFICACION PARA DECISION DE NEGOCIO

# =============================================================================

print("\n" + "=" * 60)
print("PASO 5: CLASIFICACION Y RECOMENDACION")
print("=" * 60)

def clasificar(row):

    if not row["confiable"]:

        return (
            "SIN CONCLUSION — "
            "poca variacion de tasa"
        )

    if not row["significativo"]:

        return (
            "SIN CONCLUSION — "
            "no significativo"
        )

    if abs(row["elasticidad"]) < 0.5:

        return (
            "INELASTICO — "
            "candidato a subir tasa"
        )

    elif abs(row["elasticidad"]) < 1.0:

        return (
            "MODERADO — "
            "subir tasa con cautela"
        )

    else:

        return (
            "ELASTICO — "
            "NO subir tasa"
        )

df_res["recomendacion"] = df_res.apply(
    clasificar,
    axis=1
)

print(
    df_res[
        [
            "grupo",
            "elasticidad",
            "pvalor",
            "rango_pp",
            "recomendacion",
        ]
    ].to_string(index=False)
)

# =============================================================================

# 9. SIMULACION — cuanto puedo subir la tasa?

# =============================================================================
print("\n" + "=" * 60)
print("PASO 6: SIMULACION DE IMPACTO")
print("=" * 60)

SUBIDA_PP = -0.01
# Cambia esto:
# 0.01 = 1 punto porcentual
# 0.02 = 2pp
# etc.

stats = (
    df.groupby("grupo")
    .agg(
        tasa_actual=("tasa_ofertada", "last"),
        conv_actual=("conversion", "last"),
    )
    .reset_index()
)

mask = df_res["recomendacion"].str.contains(
    "INELASTICO|MODERADO"
)

candidatos = df_res[mask].copy()

if len(candidatos) == 0:

    print(
        "No hay candidatos con elasticidad "
        "confiable para subir tasa."
    )

else:

    sim = candidatos.merge(
        stats,
        on="grupo"
    )

    sim["nueva_tasa"] = (
        sim["tasa_actual"] + SUBIDA_PP
    )

    sim["cambio_pct_tasa"] = (
        SUBIDA_PP / sim["tasa_actual"]
    )

    sim["cambio_pct_conv"] = (
        sim["elasticidad"]
        * sim["cambio_pct_tasa"]
    )

    sim["nueva_conv"] = (
        sim["conv_actual"]
        * (1 + sim["cambio_pct_conv"])
    )

    sim["caida_conv_pp"] = (
        (
            sim["nueva_conv"]
            - sim["conv_actual"]
        ) * 100
    )

    print(
        f"Subida simulada: "
        f"+{SUBIDA_PP*100:.0f}pp "
        f"en tasa ofertada\n"
    )

    cols = [
        "grupo",
        "tasa_actual",
        "nueva_tasa",
        "conv_actual",
        "nueva_conv",
        "caida_conv_pp",
    ]

    print(
        sim[cols]
        .round(4)
        .to_string(index=False)
    )

print("\nNOTA IMPORTANTE:")

print(
    "- La simulacion asume elasticidad "
    "constante (lineal en logs)."
)

print(
    "- Para subidas grandes (>3pp) "
    "la relacion puede no ser lineal."
)

print(
    "- Usa este numero como orden "
    "de magnitud, no como prediccion exacta."
)

print(
    "- Valida siempre con un piloto "
    "antes de aplicar masivamente."
)
# =============================================================================
# 10. VISUALIZACIONES
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig.suptitle(
    "Analisis de Elasticidad por Grupo",
    fontsize=14,
    fontweight="bold"
)

# Tasa por grupo en el tiempo

ax1 = axes[0, 0]

for g in df["grupo"].unique():

    d = df[df["grupo"] == g]

    ax1.plot(
        d["codmes"],
        d["tasa_ofertada"],
        marker="o",
        label=g,
        alpha=0.8,
        linewidth=1.5,
    )

ax1.set_title(
    "Tasa Ofertada por Grupo y Mes"
)

ax1.set_xlabel("Mes")

ax1.set_ylabel("Tasa (decimal)")

ax1.legend(
    fontsize=7,
    loc="best"
)

ax1.tick_params(
    axis="x",
    rotation=45
)

ax1.grid(alpha=0.3)

# Conversion por grupo en el tiempo

ax2 = axes[0, 1]

for g in df["grupo"].unique():

    d = df[df["grupo"] == g]

    ax2.plot(
        d["codmes"],
        d["conversion"],
        marker="o",
        label=g,
        alpha=0.8,
        linewidth=1.5,
    )

ax2.set_title(
    "Conversion por Grupo y Mes"
)

ax2.set_xlabel("Mes")

ax2.set_ylabel("Conversion (decimal)")

ax2.legend(
    fontsize=7,
    loc="best"
)

ax2.tick_params(
    axis="x",
    rotation=45
)

ax2.grid(alpha=0.3)

# Elasticidades estimadas por grupo

ax3 = axes[1, 0]

color_map = {
    "INELASTICO — candidato a subir tasa": "#2ecc71",
    "MODERADO — subir tasa con cautela": "#f39c12",
    "ELASTICO — NO subir tasa": "#e74c3c",
    "SIN CONCLUSION — no significativo": "#95a5a6",
    "SIN CONCLUSION — poca variacion de tasa": "#bdc3c7",
}

colores = [
    color_map.get(r, "#95a5a6")
    for r in df_res["recomendacion"]
]

bars = ax3.barh(
    df_res["grupo"],
    df_res["elasticidad"],
    color=colores,
    edgecolor="white",
)

ax3.axvline(
    -1,
    color="#e74c3c",
    linestyle="--",
    linewidth=1.5,
    label="|e|=1 (umbral elastico)",
)

ax3.axvline(
    -0.5,
    color="#f39c12",
    linestyle=":",
    linewidth=1.2,
    label="|e|=0.5",
)

ax3.axvline(
    0,
    color="black",
    linewidth=0.8,
)

ax3.set_title(
    "Elasticidad Estimada por Grupo"
)

ax3.set_xlabel("Elasticidad")

ax3.legend(fontsize=8)

ax3.grid(
    axis="x",
    alpha=0.3
)

# Scatter tasa vs conversion

ax4 = axes[1, 1]

for g in df["grupo"].unique():

    d = df[df["grupo"] == g]

    ax4.scatter(
        d["tasa_ofertada"],
        d["conversion"],
        label=g,
        alpha=0.7,
        s=50,
    )

ax4.set_title(
    "Tasa vs Conversion (todos los meses)"
)

ax4.set_xlabel("Tasa Ofertada")

ax4.set_ylabel("Conversion")

ax4.legend(fontsize=7)

ax4.grid(alpha=0.3)

plt.tight_layout()

# plt.savefig(
#     "/home/claude/elasticidad_panel.png",
#     dpi=150,
#     bbox_inches="tight",
# )

print("\nGraficos guardados.")

plt.close()

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 60)

print("RESUMEN FINAL")

print("=" * 60)

cols_final = [
    "grupo",
    "elasticidad",
    "pvalor",
    "rango_pp",
    "confiable",
    "recomendacion",
]

print(
    df_res[cols_final]
    .to_string(index=False)
)

print(
    f"\nElasticidad global (Two-Way FE): "
    f"{e_global:.4f}  "
    f"(p={pval_global:.4f})"
)
