########################################### Graficos ##################################

#########################################################################################

print('-' * 100)
print('\t\t\t\t\tGraficos')
print('-'*100)

#____________________ Paquetes ----

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

#____________________ Funciones ----

def _aplicar_estilo_ejecutivo(
    ax, titulo, ylabel, color_titulo='#1D4290', rotacion_x=90
):
  ax.set_title(
      titulo, fontsize=11, fontweight='bold', color=color_titulo, pad=15
  )
  ax.set_xlabel('')
  ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', color=color_titulo)
  ax.tick_params(
      axis='x', rotation=rotacion_x, labelsize=8, labelcolor='black'
  )
  ax.tick_params(axis='y', labelsize=8, labelcolor='black')
  ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')

  for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)


def plot_participacion_segmento(
    df,
    col_segmento='segmento',
    col_participacion='participacion',
    titulo='Participación por Segmento',
    ylabel='Participación (%)',
    color_barras='#27D75A',  # Verde por defecto
    color_titulo='#1D4290',  # Azul por defecto
    rotacion_x=90,
):
  # Hacemos una copia para no alterar el DataFrame original del usuario
  df_plot = df.copy()

  # Validación automática de escala (si el valor máximo es <= 1.0, asumimos proporción de 0 a 1)
  max_val = df_plot[col_participacion].max()
  if max_val <= 1.0:
    df_plot[col_participacion] = df_plot[col_participacion] * 100

  # Ordenar de menor a mayor para que el gráfico de barras luzca ordenado de izquierda a derecha
  df_sorted = df_plot.sort_values(by=col_participacion, ascending=True)

  fig, ax = plt.subplots(figsize=(8, 4))

  barras = ax.bar(
      np.arange(len(df_sorted)),
      df_sorted[col_participacion].values,
      width=1.0,
      color=color_barras,
      edgecolor='black',
      linewidth=0.6,
      alpha=0.9,
  )

  ax.set_xticks(np.arange(len(df_sorted)))
  ax.set_xticklabels(df_sorted[col_segmento].values)

  ax.set_ylim(0, df_sorted[col_participacion].max() * 1.18)

  for barra, valor in zip(barras, df_sorted[col_participacion]):
    ax.annotate(
        f'{valor:.1f}%',
        (barra.get_x() + barra.get_width() / 2, valor),
        xytext=(0, 6),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=7.5,
        color='black',
        bbox=dict(
            boxstyle='round,pad=0.22', fc='#E8F8F5', ec='none', alpha=0.9
        ),
    )

  _aplicar_estilo_ejecutivo(
      ax,
      titulo,
      ylabel,
      color_titulo=color_titulo,
      rotacion_x=rotacion_x,
  )

  plt.tight_layout()
  plt.show()

def _formatear_periodo_str(periodo_val):
  """Convierte un periodo YYYYMM (numérico o string) a formato 'Ene '25'."""
  p_str = str(int(periodo_val)) if pd.notnull(periodo_val) else ''
  if len(p_str) == 6:
    anio = p_str[2:4]
    mes_num = int(p_str[4:6])
    meses_es = {
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
    mes_str = meses_es.get(mes_num, p_str[4:6])
    return f"{mes_str} '{anio}"
  return p_str


def _aplicar_estilo_ejecutivo(
    ax, titulo, ylabel, color_titulo='#1D4290', rotacion_x=90
):
  ax.set_title(
      titulo, fontsize=10, fontweight='bold', color=color_titulo, pad=12
  )
  ax.set_xlabel('')
  ax.set_ylabel(ylabel, fontsize=8, fontweight='bold', color=color_titulo)
  ax.tick_params(
      axis='x', rotation=rotacion_x, labelsize=7.5, labelcolor='black'
  )
  ax.tick_params(axis='y', labelsize=7.5, labelcolor='black')
  ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')

  for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)


def plot_participacion_segmento_mensual_grid(
    df_panel,
    col_periodo='periodo',
    col_segmento='cluster_comercial',
    col_participacion='participacion_leads',
    titulo_base='Materialidad por Segmento',
    ylabel='Participación (%)',
    color_barras='#27D75A',
    color_titulo='#1D4290',
    rotacion_x=90,
):
  """Genera una grilla de 3 filas x 1 columna por cada lote de periodos,

  transformando el periodo YYYYMM a un formato estético (ej. Ene '25).
  """
  # Asegurar orden cronológico de los periodos
  periodos = sorted(df_panel[col_periodo].unique())
  n_periodos = len(periodos)

  # Agrupamos en bloques de 3 gráficos (3 filas x 1 columna)
  chunk_size = 3
  for i in range(0, n_periodos, chunk_size):
    sub_periodos = periodos[i : i + chunk_size]
    n_sub = len(sub_periodos)

    # Crear figura con grilla de n_sub filas (máximo 3) por 1 columna
    fig, axes = plt.subplots(n_sub, 1, figsize=(8, 3.2 * n_sub))

    # Si es solo 1 sub_periodo en el bloque, axes no es un array, lo envolvemos
    if n_sub == 1:
      axes = [axes]

    for ax, periodo in zip(axes, sub_periodos):
      df_sub = df_panel[df_panel[col_periodo] == periodo].copy()

      # Validar escala 0-1 o 0-100
      max_val = df_sub[col_participacion].max()
      if max_val <= 1.0:
        df_sub[col_participacion] = df_sub[col_participacion] * 100

      # Ordenar de menor a mayor para la visualización de barras
      df_sorted = df_sub.sort_values(by=col_participacion, ascending=True)

      barras = ax.bar(
          np.arange(len(df_sorted)),
          df_sorted[col_participacion].values,
          width=1.0,
          color=color_barras,
          edgecolor='black',
          linewidth=0.6,
          alpha=0.9,
      )

      ax.set_xticks(np.arange(len(df_sorted)))
      ax.set_xticklabels(df_sorted[col_segmento].values)
      ax.set_ylim(0, df_sorted[col_participacion].max() * 1.18)

      # Anotaciones de porcentaje
      for barra, valor in zip(barras, df_sorted[col_participacion]):
        ax.annotate(
            f'{valor:.1f}%',
            (barra.get_x() + barra.get_width() / 2, valor),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=7,
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.22', fc='#E8F8F5', ec='none', alpha=0.9
            ),
        )

      # Formatear el periodo a estilo "Ene '25"
      periodo_formateado = _formatear_periodo_str(periodo)
      titulo_final = f'{titulo_base} \n Periodo: {periodo_formateado}'

      _aplicar_estilo_ejecutivo(
          ax,
          titulo_final,
          ylabel,
          color_titulo=color_titulo,
          rotacion_x=rotacion_x,
      )

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import pandas as pd


def _formatear_periodo_str(periodo_val):
    """Convierte un periodo YYYYMM (numérico o string) a formato 'Ene '25'."""
    p_str = str(int(periodo_val)) if pd.notnull(periodo_val) else ''
    if len(p_str) == 6:
        anio = p_str[2:4]
        mes_num = int(p_str[4:6])
        meses_es = {
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
        mes_str = meses_es.get(mes_num, p_str[4:6])
        return f"{mes_str} '{anio}"
    return p_str


def plot_heatmap_cumplimiento(
    df_panel,
    col_cluster='cluster_comercial',
    col_periodo='periodo',
    col_flag='flg_cumplimiento',
    titulo='Cumplimiento de Materialidad',
    subtitulo='Umbral mínimo de participación: > 3%',
    color_titulo='#4A5568',
    color_cumplio='#5CD699',  # Verde menta
    color_no_cumplio='#FF8B8B',  # Coral suave (rojo)
    color_sin_registro='#E2E8F0',  # Gris claro (celdas faltantes)
):
    """Genera un tablero estilo matriz rectangular con celdas redondeadas,

    título compacto, subtítulo cercano, etiquetas centradas y leyenda
    perfectamente equilibrada al centro inferior.
    """
    # 1. Pivotar el data panel para construir la matriz
    df_pivot = df_panel.pivot(
        index=col_cluster, columns=col_periodo, values=col_flag
    )

    # Ordenar filas y columnas cronológicamente
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)

    clusters = df_pivot.index.tolist()
    periodos_raw = df_pivot.columns.tolist()

    # Formatear periodos
    periodos_labels = [_formatear_periodo_str(p) for p in periodos_raw]
    matriz_datos = df_pivot.values

    colores_map = {
        1: color_cumplio,
        0: color_no_cumplio,
    }

    num_clusters = len(clusters)
    num_periodos = len(periodos_raw)

    # 2. Inicializar figura con proporciones armónicas
    fig_width = max(10, num_periodos * 0.75)
    fig_height = max(4.5, num_clusters * 0.55)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    # Ajuste de márgenes equilibrado (ni mucho espacio ni muy apretado)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.18, right=0.92)

    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    tamano_caja = 0.65
    radio_esquina = 0.15

    # 3. Dibujar celdas
    for i in range(num_clusters):
        for j in range(num_periodos):
            val = matriz_datos[i, j]

            if pd.isna(val):
                color_celda = color_sin_registro
            else:
                color_celda = colores_map.get(int(val), color_sin_registro)

            x = j - tamano_caja / 2
            y = (num_clusters - 1 - i) - tamano_caja / 2

            cuadro_redondeado = FancyBboxPatch(
                (x, y),
                tamano_caja,
                tamano_caja,
                boxstyle=f'round,pad={radio_esquina}',
                facecolor=color_celda,
                edgecolor='none',
            )
            ax.add_patch(cuadro_redondeado)

    # 4. Configurar ejes
    ax.set_xlim(-0.6, num_periodos - 0.4)
    ax.set_ylim(-0.6, num_clusters - 0.4)

    ax.set_xticks(range(num_periodos))
    ax.set_xticklabels(
        periodos_labels, fontsize=10, color='#4A5568', fontweight='normal'
    )
    plt.xticks(rotation=90, ha='center')

    ax.set_yticks(range(num_clusters))
    ax.set_yticklabels(
        clusters[::-1], fontsize=10, color='#4A5568', fontweight='normal'
    )

    # Eliminar bordes y ticks físicos
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_aspect('equal')

    # 5. Título y subtítulo juntos y compactos
    fig.suptitle(
        titulo,
        fontsize=13,
        fontweight='bold',
        color=color_titulo,
        x=0.5,
        y=0.93,
        ha='center',
    )

    if subtitulo:
        fig.text(
            0.5,
            0.87,
            subtitulo,
            fontsize=10,
            fontweight='normal',
            color=color_titulo,
            ha='center',
            va='center',
        )

    # 6. Leyenda centrada horizontalmente con bordes sutiles en los parches
    elementos_leyenda = [
        Patch(
            facecolor=color_cumplio, edgecolor='#38A169', linewidth=0.5, label='Cumple'
        ),
        Patch(
            facecolor=color_no_cumplio, edgecolor='#E53E3E', linewidth=0.5, label='No Cumple'
        ),
        Patch(
            facecolor=color_sin_registro, edgecolor='#CBD5E1', linewidth=0.5, label='Sin Registro'
        ),
    ]

    fig.legend(
        handles=elementos_leyenda,
        loc='lower center',
        bbox_to_anchor=(0.30, -0.04),  # Centrado perfecto en la parte inferior
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handletextpad=0.5,
        columnspacing=2.5,
    )

    plt.show()

def _escalar_serie(s):
  s_clean = s.dropna()
  if not s_clean.empty and s_clean.max() <= 1.0:
    return s * 100, '%'
  return s, ''

def plot_evolutivo_precio_demanda(
    df,
    columna_demanda='efectividad',
    columna_precio='tasa_oferta_ponderada',
    columna_materialidad='desembolsos',
    label_demanda=None,
    label_precio=None,
    titulo=None,
    flg_materialidad=True,
    flg_etiqueta_demanda=False,
    flg_etiqueta_precio=False,
    color_demanda='#27D75A',
    color_precio='#1D4290',
):
  if df.empty:
    print('El DataFrame no contiene registros para graficar')
    return

  sub_df = df.copy()
  sub_df['periodo'] = sub_df['periodo'].astype(int)
  sub_df = sub_df.sort_values('periodo').reset_index(drop=True)

  # Definir etiquetas ejecutivas por defecto si no se proporcionan
  label_demanda = label_demanda or columna_demanda.replace(
      '_', ' '
  ).capitalize()
  label_precio = label_precio or columna_precio.replace('_', ' ').capitalize()

  # Formato de periodo (ej. Ene' 24)
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
  sub_df['periodo_label'] = sub_df['periodo'].apply(
      lambda x: f"{meses[int(x) % 100]}' {str(int(x) // 100)[-2:]}"
  )

  fig, ax1 = plt.subplots(figsize=(8, 4))
  x = np.arange(len(sub_df))

  # Gráfico de barras de fondo (Materialidad) con soporte para n=0
  if flg_materialidad and columna_materialidad in sub_df.columns:
    ax_bar = ax1.twinx()
    ax_bar.set_zorder(0)
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)

    # Rellenamos nulos con 0 para que la barra exista visualmente o mida 0
    valores_mat = sub_df[columna_materialidad].fillna(0)
    barras = ax_bar.bar(
        x, valores_mat, width=0.65, color='lightgray', alpha=0.22, zorder=0
    )

    max_mat = valores_mat.max()
    if max_mat > 0:
      ax_bar.set_ylim(0, max_mat * 1.25)
    else:
      ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])

    for spine in ax_bar.spines.values():
      spine.set_visible(False)

    # Etiqueta n=... para todas las barras (incluyendo ceros o nulos)
    max_val_ref = max_mat if max_mat > 0 else 1
    for barra, valor in zip(barras, valores_mat):
      val_int = int(valor) if pd.notnull(valor) else 0
      ax_bar.text(
          barra.get_x() + barra.get_width() / 2,
          max_val_ref * 0.02,
          f'n={val_int:,}',
          ha='center',
          va='bottom',
          fontsize=6.5,
          color='gray',
          alpha=0.8,
      )

  # Función interna para graficar tramos continuos y puentes discontinuos ante huecos
  def _plot_serie_con_discontinuidades(
      ax, y_vals, color, marker, label, escala_sufijo
  ):
    df_temp = pd.DataFrame(
        {'x': x, 'y': y_vals, 'idx': sub_df['periodo']}
    ).dropna(subset=['y'])

    if df_temp.empty:
      return ax.plot(
          [], [], color=color, linestyle='-', marker=marker, label=label
      )

    df_temp['salto'] = df_temp['x'].diff().fillna(1).gt(1)
    grupo = df_temp['salto'].cumsum()

    # Tramos sólidos continuos
    for _, tramo in df_temp.groupby(grupo):
      ax.plot(
          tramo['x'],
          tramo['y'],
          color=color,
          linestyle='-',
          marker=marker,
          markersize=4,
          linewidth=2,
          zorder=3,
      )

    # Puentes entrecortados entre saltos
    indices_saltos = df_temp.index[df_temp['salto']].tolist()
    for idx in indices_saltos:
      pos_actual = df_temp.index.get_loc(idx)
      idx_anterior = df_temp.index[pos_actual - 1]

      x_prev, y_prev = df_temp.loc[idx_anterior, ['x', 'y']]
      x_curr, y_curr = df_temp.loc[idx, ['x', 'y']]

      ax.plot(
          [x_prev, x_curr],
          [y_prev, y_curr],
          color=color,
          linestyle='--',
          linewidth=1.5,
          zorder=2.5,
      )

    # Objeto dummy limpio para la leyenda
    return ax.plot(
        [], [], color=color, linestyle='-', marker=marker, label=label
    )

  # Serie Demanda
  y_demanda_raw, sufijo_demanda = _escalar_serie(sub_df[columna_demanda])
  linea1 = _plot_serie_con_discontinuidades(
      ax1,
      y_demanda_raw.values,
      color_demanda,
      'o',
      label_demanda,
      sufijo_demanda,
  )

  ax1.set_xticks(x)
  ax1.set_xticklabels(sub_df['periodo_label'])
  ax1.set_xlabel('Periodo', fontsize=9, fontweight='bold')
  ax1.set_ylabel(
      label_demanda, fontsize=9, fontweight='bold', color=color_demanda
  )
  ax1.tick_params(axis='both', labelsize=8, labelcolor='black')
  ax1.tick_params(axis='x', rotation=90)

  # Serie Precio
  ax2 = ax1.twinx()
  y_precio_raw, sufijo_precio = _escalar_serie(sub_df[columna_precio])
  linea2 = _plot_serie_con_discontinuidades(
      ax2, y_precio_raw.values, color_precio, 's', label_precio, sufijo_precio
  )

  ax2.set_ylabel(
      label_precio, fontsize=9, fontweight='bold', color=color_precio
  )
  ax2.tick_params(axis='y', labelsize=8, labelcolor='black')

  # Formato de ejes dinámico según sufijo porcentual
  if sufijo_demanda:
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
    )
  if sufijo_precio:
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
    )

  # Etiquetas Demanda
  if flg_etiqueta_demanda:
    for xi, valor in zip(x, y_demanda_raw.values):
      if pd.notnull(valor):
        ax1.annotate(
            f'{valor:.1f}{sufijo_demanda}',
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
                ec=color_demanda,
                lw=0.5,
                alpha=0.9,
            ),
        )

  # Etiquetas Precio
  if flg_etiqueta_precio:
    for xi, valor in zip(x, y_precio_raw.values):
      if pd.notnull(valor):
        ax2.annotate(
            f'{valor:.1f}{sufijo_precio}',
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
                ec=color_precio,
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
      lineas,
      etiquetas,
      loc='upper left',
      frameon=True,
      facecolor='white',
      edgecolor='lightgray',
      fontsize=8,
  )

  titulo = (
      titulo or f'Evolución de {label_demanda.lower()} y {label_precio.lower()}'
  )
  ax1.set_title(
      titulo,
      fontsize=11,
      fontweight='bold',
      color=color_precio,
      pad=15,
  )

  fig.tight_layout()
  plt.show()

PALETA_15 = [
    '#E37222', '#329262', '#1D4290', '#8B0000', '#E69F00',
    '#0072B2', '#D55E00', '#CC79A7', '#56B4E9', '#009E73',
    '#27D75A', '#332288', '#117733', '#CC6677', '#AA4499'
]

def _formatear_periodos_panel(df, columna_periodo):
    sub_df = df.copy()
    meses = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
             7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
    sub_df['periodo_label'] = sub_df[columna_periodo].apply(
        lambda x: f"{meses[int(x) % 100]}' {str(int(x) // 100)[-2:]}"
    )
    return sub_df

def plot_evolutivo_multiserie_panel(
    df,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica='efectividad',
    label_metrica='Efectividad (%)',
    titulo=None,
    multiplicar_100=True,
    paleta=PALETA_15,
):
  if df.empty:
    print('El DataFrame no contiene registros para graficar')
    return

  sub_df = _formatear_periodos_panel(df, columna_periodo)

  periodos_unicos = sorted(sub_df[columna_periodo].unique())
  mapa_x = {p: i for i, p in enumerate(periodos_unicos)}
  sub_df['x_idx'] = sub_df[columna_periodo].map(mapa_x)

  # Aumentamos un poco la altura para dar espacio a la leyenda inferior
  fig, ax = plt.subplots(figsize=(10, 5.2))
  grupos = sorted(sub_df[columna_grupo].unique())

  for i, grupo in enumerate(grupos):
    color = paleta[i % len(paleta)]
    df_grupo = sub_df[sub_df[columna_grupo] == grupo].sort_values(
        columna_periodo
    )

    y_vals = df_grupo[metrica] * 100 if multiplicar_100 else df_grupo[metrica]

    # Preparamos dataframe temporal del grupo para evaluar huecos usando 'x_idx'
    df_temp = pd.DataFrame(
        {'x': df_grupo['x_idx'], 'y': y_vals}
    ).dropna(subset=['y'])

    if df_temp.empty:
      continue

    # Detectar saltos (si la diferencia de índice x es mayor a 1, hay un hueco temporal)
    df_temp['salto'] = df_temp['x'].diff().fillna(1).gt(1)
    grupo_tramo = df_temp['salto'].cumsum()

    # 1. Dibujar tramos sólidos continuos
    for _, tramo in df_temp.groupby(grupo_tramo):
      ax.plot(
          tramo['x'],
          tramo['y'],
          color=color,
          linestyle='-',
          marker='o',
          markersize=3.5,
          linewidth=1.8,
          zorder=3,
      )

    # 2. Dibujar puentes entrecortados cruzando los vacíos temporales
    indices_saltos = df_temp.index[df_temp['salto']].tolist()
    for idx in indices_saltos:
      pos_actual = df_temp.index.get_loc(idx)
      idx_anterior = df_temp.index[pos_actual - 1]

      x_prev, y_prev = df_temp.loc[idx_anterior, ['x', 'y']]
      x_curr, y_curr = df_temp.loc[idx, ['x', 'y']]

      ax.plot(
          [x_prev, x_curr],
          [y_prev, y_curr],
          color=color,
          linestyle='--',
          linewidth=1.4,
          zorder=2.5,
      )

    # Línea dummy invisible solo para registrar la etiqueta correcta en la leyenda
    ax.plot([], [], color=color, linestyle='-', marker='o', label=str(grupo))

  df_labels = sub_df.drop_duplicates(columna_periodo).sort_values(
      columna_periodo
  )
  ax.set_xticks(df_labels['x_idx'])
  ax.set_xticklabels(df_labels['periodo_label'])
  ax.set_xlabel('Periodo', fontsize=9, fontweight='bold', color='#1D4290')
  ax.set_ylabel(label_metrica, fontsize=9, fontweight='bold', color='#1D4290')
  ax.tick_params(axis='both', labelsize=8, labelcolor='black')
  ax.tick_params(axis='x', rotation=90)

  if multiplicar_100:
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
    )

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')

  # Leyenda abajo, distribuida en columnas automáticas (máx 4 por fila)
  num_cols = min(len(grupos), 4)
  ax.legend(
      loc='upper center',
      bbox_to_anchor=(0.5, -0.22),
      ncol=num_cols,
      frameon=True,
      facecolor='white',
      edgecolor='lightgray',
      fontsize=8,
  )

  titulo = (
      titulo
      or f"Evolución de {metrica.replace('_', ' ').capitalize()} por {columna_grupo.replace('_', ' ').capitalize()}"
  )
  ax.set_title(titulo, fontsize=11, fontweight='bold', color='#1D4290', pad=15)

  # Ajustamos márgenes para evitar que la leyenda se corte al mostrar
  fig.subplots_adjust(bottom=0.25)
  plt.show()
  
def plot_participacion_panel(
    df,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    metrica_materialidad='desembolsos',
    titulo=None,
    paleta=PALETA_15
):
    if df.empty:
        print('El DataFrame no contiene registros para graficar')
        return

    sub_df = _formatear_periodos_panel(df, columna_periodo)
    
    df_pivot = sub_df.pivot(index=columna_periodo, columns=columna_grupo, values=metrica_materialidad).fillna(0)
    df_pct = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

    periodos_unicos = df_pct.index.tolist()
    x = np.arange(len(periodos_unicos))
    
    labels_periodo = sub_df.drop_duplicates(columna_periodo).set_index(columna_periodo).loc[periodos_unicos]['periodo_label'].values

    fig, ax = plt.subplots(figsize=(10, 5.2))
    grupos = df_pct.columns.tolist()

    bottom = np.zeros(len(periodos_unicos))
    for i, grupo in enumerate(grupos):
        color = paleta[i % len(paleta)]
        valores = df_pct[grupo].values

        ax.bar(
            x, valores,
            bottom=bottom,
            width=0.65,
            color=color,
            label=str(grupo),
            edgecolor='white',
            linewidth=0.5
        )
        bottom += valores

    ax.set_xticks(x)
    ax.set_xticklabels(labels_periodo)
    ax.set_xlabel('Periodo', fontsize=9, fontweight='bold', color='#1D4290')
    ax.set_ylabel('Participación (%)', fontsize=9, fontweight='bold', color='#1D4290')
    ax.tick_params(axis='both', labelsize=8, labelcolor='black')
    ax.tick_params(axis='x', rotation=90)

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.0f}%'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')

    # Leyenda abajo en columnas
    num_cols = min(len(grupos), 4)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=num_cols,
        frameon=True,
        facecolor='white',
        edgecolor='lightgray',
        fontsize=8
    )

    titulo = titulo or f'Mix de {metrica_materialidad.replace("_", " ").capitalize()} por {columna_grupo.replace("_", " ").capitalize()} (%)'
    ax.set_title(titulo, fontsize=11, fontweight='bold', color='#1D4290', pad=15)

    fig.subplots_adjust(bottom=0.25)
    plt.show()

def plot_nube_puntos_precio_demanda_interna(
    df,
    columna_precio='tasa_oferta_ponderada',
    columna_demanda='efectividad',
    columna_tamanio=None,  # Opcional: dimensionar tamaño de puntos (monto, n, etc.)
    columna_color=None,  # Opcional: colorear los puntos
    columna_cluster=None,  # Opcional: columna con el nombre/etiqueta de cada clúster
    label_precio=None,
    label_demanda=None,
    titulo=None,
    color_fijo='#27D75A',  # Color por defecto (verde ejecutivo)
    alpha_puntos=0.85,
):
  """Genera una nube de puntos (Scatter Plot) ejecutiva cruzando precio vs

  demanda, con burbujas escaladas, etiquetas abajo sin caja y nota al pie del
  campo de tamaño.
  """
  if df.empty:
    print('El DataFrame no contiene registros para graficar')
    return

  sub_df = df.copy()

  # Definir etiquetas ejecutivas por defecto si no se proporcionan
  label_precio = label_precio or columna_precio.replace('_', ' ').capitalize()
  label_demanda = label_demanda or columna_demanda.replace(
      '_', ' '
  ).capitalize()

  # Detección automática de escala porcentual para los ejes si los valores son 0-1
  def _escalar_serie(s):
    s_clean = s.dropna()
    if not s_clean.empty and s_clean.max() <= 1.0:
      return s * 100, '%'
    return s, ''

  x_vals, sufijo_x = _escalar_serie(sub_df[columna_precio])
  y_vals, sufijo_y = _escalar_serie(sub_df[columna_demanda])

  # Configurar figura bajo estilo ejecutivo (dejamos un pequeño margen inferior extra para la nota)
  fig, ax = plt.subplots(figsize=(9, 5.8), dpi=300)
  fig.patch.set_facecolor('#F8F9FA')
  ax.set_facecolor('#F8F9FA')

  # Procesamiento de tamaños con rango amplio y visible (burbujas grandes)
  s_args = 250  # Tamaño base predeterminado
  if columna_tamanio and columna_tamanio in sub_df.columns:
    tamanio_raw = sub_df[columna_tamanio].fillna(0)
    min_t, max_t = tamanio_raw.min(), tamanio_raw.max()
    if max_t > min_t:
      s_args = 150 + 950 * (tamanio_raw - min_t) / (max_t - min_t)
    else:
      s_args = 400

  # Procesamiento opcional de colores o color fijo
  c_args = color_fijo
  cmap_usado = None
  if columna_color and columna_color in sub_df.columns:
    c_args = sub_df[columna_color]
    cmap_usado = 'Greens'

  # Trazar el scatter plot
  sc = ax.scatter(
      x_vals,
      y_vals,
      s=s_args,
      c=c_args,
      cmap=cmap_usado,
      alpha=alpha_puntos,
      edgecolors='white',
      linewidths=1.2,
      zorder=3,
  )

  # Si se usó una barra de colores continua, agregar colorbar sutil
  if (
      columna_color
      and columna_color in sub_df.columns
      and not pd.api.types.is_categorical_dtype(sub_df[columna_color])
      and not pd.api.types.is_object_dtype(sub_df[columna_color])
  ):
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

  # Añadir etiquetas de texto por debajo de cada punto (limpias, sin caja)
  if columna_cluster and columna_cluster in sub_df.columns:
    for xi, yi, etiqueta in zip(x_vals, y_vals, sub_df[columna_cluster]):
      if pd.notnull(xi) and pd.notnull(yi) and pd.notnull(etiqueta):
        ax.annotate(
            str(etiqueta),
            (xi, yi),
            xytext=(0, -12.5),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=6,
            # fontweight='bold',
            color='#1E293B',
            zorder=4,
        )

  # Configuración y formato de ejes
  ax.set_xlabel(
      f'{label_precio} ({sufijo_x})' if sufijo_x else label_precio,
      fontsize=9,
      fontweight='bold',
      color='#1D4290',
  )
  ax.set_ylabel(
      f'{label_demanda} ({sufijo_y})' if sufijo_y else label_demanda,
      fontsize=9,
      fontweight='bold',
      color='#1D4290',
  )

  ax.tick_params(axis='both', labelsize=8, labelcolor='#1D4290')

  if sufijo_x == '%':
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
    )
  if sufijo_y == '%':
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
    )

  # Limpieza de bordes estilo ejecutivo
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_color('#CBD5E1')
  ax.spines['bottom'].set_color('#CBD5E1')

  ax.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)

  # Título principal
  titulo = (
      titulo
      or f'Relación entre {label_precio.lower()} y {label_demanda.lower()}'
  )
  ax.set_title(
      titulo,
      fontsize=12,
      fontweight='bold',
      color='#1D4290',
      pad=22,
  )

  # Nota al pie indicando la materialidad o campo de tamaño si está activo
  if columna_tamanio:
    nombre_limpio_tam = columna_tamanio.replace('_', ' ').capitalize()
    fig.text(
        0.08,
        0.03,
        f'Nota: El tamaño de las burbujas es proporcional a la materialidad según el campo: {nombre_limpio_tam}.',
        fontsize=75 / 10,
        color='#64748B',
        style='italic',
    )

  # Ajustar márgenes para dar espacio a la nota al pie inferior
  plt.tight_layout(rect=[0, 0.05, 1, 1])
  plt.show()

def plot_efectividad_por_bloques_y_materialidad(
    df,
    columna_cluster='cluster_comercial',
    columna_bloque='categoria_bloque',  # Columna que divide en grupos (ej. Con deuda / Sin deuda)
    metrica_efectividad='efectividad',
    columna_materialidad='n_creditos',
    titulo=None,
):
  """Genera un gráfico ejecutivo con barras de materialidad al fondo,

  puntos de efectividad conectados por líneas y ordenados por bloques y
  efectividad de mayor a menor, usando estrictamente los colores azul y verde
  corporativos.
  """
  if df.empty:
    print('El DataFrame no contiene registros para graficar')
    return

  sub_df = df.copy()

  # Asegurarnos de que la efectividad esté en porcentaje
  if sub_df[metrica_efectividad].max() <= 1.0:
    sub_df['__efect_plot'] = sub_df[metrica_efectividad] * 100
  else:
    sub_df['__efect_plot'] = sub_df[metrica_efectividad]

  # Ordenar dentro de cada bloque por efectividad de mayor a menor
  if columna_bloque in sub_df.columns:
    sub_df = (
        sub_df.sort_values(
            by=[columna_bloque, '__efect_plot'], ascending=[True, False]
        )
        .reset_index(drop=True)
    )
  else:
    sub_df = sub_df.sort_values(by='__efect_plot', ascending=False).reset_index(
        drop=True
    )

  # Configurar figura bajo el estándar ejecutivo corporativo
  fig, ax1 = plt.subplots(figsize=(11, 5.8), dpi=300)
  fig.patch.set_facecolor('#F8F9FA')
  ax1.set_facecolor('#F8F9FA')

  x = np.arange(len(sub_df))

  # Eje secundario para las barras de fondo (Materialidad más atrás: zorder=0)
  ax_bar = ax1.twinx()
  ax_bar.set_zorder(0)
  ax_bar.set_zorder(1)
  ax_bar.patch.set_visible(False)

  valores_mat = sub_df[columna_materialidad].fillna(0)
  max_mat = valores_mat.max() if not valores_mat.empty else 1
  if max_mat == 0:
    max_mat = 1

  ax_bar.set_ylim(0, max_mat * 1.4)
  ax_bar.set_yticks([])

  # Definir los colores corporativos institucionales: Azul y Verde
  bloques_unicos = (
      sub_df[columna_bloque].unique()
      if columna_bloque in sub_df.columns
      else []
  )

  # Bloque 1: Azul corporativo (#1D4290) con barra translúcida suave (#D0E5F2)
  # Bloque 2: Verde corporativo (#27D75A) con barra translúcida suave (#D4EFDF)
  color_b1_barra, color_b1_linea = '#D0E5F2', '#1D4290'
  color_b2_barra, color_b2_linea = '#D4EFDF', '#1B9941'  # Verde institucional ajustado para contraste óptimo

  colores_barras = []
  if len(bloques_unicos) >= 2:
    b1, b2 = bloques_unicos[0], bloques_unicos[1]
    for b in sub_df[columna_bloque]:
      colores_barras.append(color_b1_barra if b == b1 else color_b2_barra)
  else:
    colores_barras = ['#D0E5F2'] * len(sub_df)

  barras = ax_bar.bar(
      x,
      valores_mat,
      width=0.6,
      color=colores_barras,
      alpha=0.5,
      edgecolor='white',
      zorder=0,
  )

  # Etiquetas de materialidad en entero sobre las barras
  for barra, valor in zip(barras, valores_mat):
    val_int = int(round(valor)) if pd.notnull(valor) else 0
    ax_bar.text(
        barra.get_x() + barra.get_width() / 2,
        max_mat * 0.02,
        f'n={val_int:,}',
        ha='center',
        va='bottom',
        fontsize=6.5,
        color='#64748B',
        alpha=0.9,
    )

  for spine in ax_bar.spines.values():
    spine.set_visible(False)

  # Configurar eje principal de efectividad (zorder superior)
  ax1.set_zorder(3)
  ax1.patch.set_visible(False)

  y_efectividad = sub_df['__efect_plot']

  # Dibujar líneas conectando los puntos y puntos agrupados por color del bloque correspondiente
  if len(bloques_unicos) >= 2:
    for b in bloques_unicos:
      mask = sub_df[columna_bloque] == b
      if mask.any():
        x_sub = x[mask]
        y_sub = y_efectividad[mask]
        col_actual = color_b1_linea if b == b1 else color_b2_linea

        # Línea del bloque
        ax1.plot(
            x_sub,
            y_sub,
            color=col_actual,
            linestyle='-',
            linewidth=1.6,
            alpha=0.9,
            zorder=3,
        )
        # Puntos del bloque
        ax1.scatter(
            x_sub,
            y_sub,
            color=col_actual,
            s=75,
            zorder=4,
            edgecolors='white',
            linewidths=1.2,
        )
  else:
    ax1.plot(
        x,
        y_efectividad,
        color=color_b1_linea,
        linestyle='-',
        linewidth=1.6,
        alpha=0.9,
        zorder=3,
    )
    ax1.scatter(
        x,
        y_efectividad,
        color=color_b1_linea,
        s=75,
        zorder=4,
        edgecolors='white',
        linewidths=1.2,
    )

  # Etiquetas de porcentaje encima de cada punto con el color de su respectivo bloque
  for xi, valor, b in zip(
      x,
      y_efectividad,
      sub_df[columna_bloque] if len(bloques_unicos) >= 2 else ['único'] * len(x),
  ):
    if pd.notnull(valor):
      col_txt = color_b1_linea if b == bloques_unicos[0] else color_b2_linea
      ax1.annotate(
          f'{valor:.1f}%',
          (xi, valor),
          xytext=(0, 9),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=7,
          fontweight='bold',
          color=col_txt,
      )

  # Línea divisoria vertical y títulos superiores de los bloques con sus colores coordinados
  if len(bloques_unicos) >= 2:
    idx_cambio = (sub_df[columna_bloque] != bloques_unicos[0]).idxmax() - 0.5
    ax1.axvline(
        x=idx_cambio,
        color='#94A3B8',
        linestyle=':',
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
    )

    pos_b1 = (sub_df[columna_bloque] == bloques_unicos[0]).sum() / 2 - 0.5
    pos_b2 = (
        sub_df[columna_bloque] == bloques_unicos[0]
    ).sum() + (sub_df[columna_bloque] == bloques_unicos[1]).sum() / 2 - 0.5

    ax1.text(
        pos_b1,
        ax1.get_ylim()[1] * 0.93,
        str(bloques_unicos[0]),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color=color_b1_linea,
    )
    ax1.text(
        pos_b2,
        ax1.get_ylim()[1] * 0.93,
        str(bloques_unicos[1]),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color=color_b2_linea,
    )

  # Configuración de ejes
  ax1.set_xticks(x)
  ax1.set_xticklabels(
      sub_df[columna_cluster], rotation=45, ha='right', fontsize=8
  )
  ax1.set_ylabel(
      'Efectividad (%)', fontsize=9, fontweight='bold', color='#1D4290'
  )
  ax1.tick_params(axis='both', labelsize=8, labelcolor='#1D4290')
  ax1.yaxis.set_major_formatter(
      plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
  )

  max_y = y_efectividad.max() if not y_efectividad.empty else 100
  ax1.set_ylim(0, max(max_y * 1.25, 30))

  # Limpieza de bordes estilo ejecutivo
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.spines['left'].set_color('#CBD5E1')
  ax1.spines['bottom'].set_color('#CBD5E1')
  ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray', zorder=1)

  # Título principal
  titulo = titulo or 'Efectividad por Clúster Comercial y Materialidad'
  ax1.set_title(
      titulo,
      fontsize=12,
      fontweight='bold',
      color='#1D4290',
      pad=20,
  )

  plt.tight_layout()
  plt.show()

def plot_efectividad_y_tasa_por_bloques(
    df,
    columna_cluster='cluster_comercial',
    columna_bloque='categoria_bloque',  # Columna que divide en grupos (ej. Con deuda / Sin deuda)
    metrica_efectividad='efectividad',
    columna_tasa='tasa_desembolsada',  # Ahora las barras muestran tasa en lugar de materialidad
    titulo=None,
):
  """Genera un gráfico ejecutivo con barras de fondo representando la tasa,

  y puntos de efectividad conectados por líneas ordenados por bloques y
  efectividad de mayor a menor, usando azul y verde corporativos.
  """
  if df.empty:
    print('El DataFrame no contiene registros para graficar')
    return

  sub_df = df.copy()

  # Asegurarnos de que la efectividad esté en porcentaje
  if sub_df[metrica_efectividad].max() <= 1.0:
    sub_df['__efect_plot'] = sub_df[metrica_efectividad] * 100
  else:
    sub_df['__efect_plot'] = sub_df[metrica_efectividad]

  # Detección automática de escala porcentual para la tasa si los valores son 0-1
  if sub_df[columna_tasa].max() <= 1.0:
    sub_df['__tasa_plot'] = sub_df[columna_tasa] * 100
    sufijo_tasa = '%'
  else:
    sub_df['__tasa_plot'] = sub_df[columna_tasa]
    sufijo_tasa = '%'  # o '' dependiendo de tu formato

  # Ordenar dentro de cada bloque por efectividad de mayor a menor
  if columna_bloque in sub_df.columns:
    sub_df = (
        sub_df.sort_values(
            by=[columna_bloque, '__efect_plot'], ascending=[True, False]
        )
        .reset_index(drop=True)
    )
  else:
    sub_df = sub_df.sort_values(by='__efect_plot', ascending=False).reset_index(
        drop=True
    )

  # Configurar figura bajo el estándar ejecutivo corporativo
  fig, ax1 = plt.subplots(figsize=(11, 5.8), dpi=300)
  fig.patch.set_facecolor('#F8F9FA')
  ax1.set_facecolor('#F8F9FA')

  x = np.arange(len(sub_df))

  # Eje secundario para las barras de fondo (Tasa más atrás: zorder=0)
  ax_bar = ax1.twinx()
  ax_bar.set_zorder(0)
  ax_bar.set_zorder(1)
  ax_bar.patch.set_visible(False)

  valores_tasa = sub_df['__tasa_plot'].fillna(0)
  max_tasa = valores_tasa.max() if not valores_tasa.empty else 1
  if max_tasa == 0:
    max_tasa = 1

  ax_bar.set_ylim(0, max_tasa * 1.4)
  ax_bar.set_yticks([])

  # Definir los colores corporativos institucionales: Azul y Verde
  bloques_unicos = (
      sub_df[columna_bloque].unique()
      if columna_bloque in sub_df.columns
      else []
  )

  color_b1_barra, color_b1_linea = '#D0E5F2', '#1D4290'  # Azul corporativo
  color_b2_barra, color_b2_linea = (
      '#D4EFDF',
      '#1B9941',
  )  # Verde institucional

  colores_barras = []
  if len(bloques_unicos) >= 2:
    b1, b2 = bloques_unicos[0], bloques_unicos[1]
    for b in sub_df[columna_bloque]:
      colores_barras.append(color_b1_barra if b == b1 else color_b2_barra)
  else:
    colores_barras = ['#D0E5F2'] * len(sub_df)

  barras = ax_bar.bar(
      x,
      valores_tasa,
      width=0.6,
      color=colores_barras,
      alpha=0.5,
      edgecolor='white',
      zorder=0,
  )

  # Etiquetas de tasa sobre las barras (ej. 17.5%)
  for barra, valor in zip(barras, valores_tasa):
    if pd.notnull(valor):
      ax_bar.text(
          barra.get_x() + barra.get_width() / 2,
          max_tasa * 0.02,
          f'{valor:.1f}{sufijo_tasa}',
          ha='center',
          va='bottom',
          fontsize=6.5,
          color='#64748B',
          alpha=0.9,
      )

  for spine in ax_bar.spines.values():
    spine.set_visible(False)

  # Configurar eje principal de efectividad (zorder superior)
  ax1.set_zorder(3)
  ax1.patch.set_visible(False)

  y_efectividad = sub_df['__efect_plot']

  # Dibujar líneas conectando los puntos y puntos agrupados por color del bloque
  if len(bloques_unicos) >= 2:
    for b in bloques_unicos:
      mask = sub_df[columna_bloque] == b
      if mask.any():
        x_sub = x[mask]
        y_sub = y_efectividad[mask]
        col_actual = color_b1_linea if b == b1 else color_b2_linea

        # Línea del bloque
        ax1.plot(
            x_sub,
            y_sub,
            color=col_actual,
            linestyle='-',
            linewidth=1.6,
            alpha=0.9,
            zorder=3,
        )
        # Puntos del bloque
        ax1.scatter(
            x_sub,
            y_sub,
            color=col_actual,
            s=75,
            zorder=4,
            edgecolors='white',
            linewidths=1.2,
        )
  else:
    ax1.plot(
        x,
        y_efectividad,
        color=color_b1_linea,
        linestyle='-',
        linewidth=1.6,
        alpha=0.9,
        zorder=3,
    )
    ax1.scatter(
        x,
        y_efectividad,
        color=color_b1_linea,
        s=75,
        zorder=4,
        edgecolors='white',
        linewidths=1.2,
    )

  # Etiquetas de porcentaje de efectividad encima de cada punto
  for xi, valor, b in zip(
      x,
      y_efectividad,
      sub_df[columna_bloque] if len(bloques_unicos) >= 2 else ['único'] * len(x),
  ):
    if pd.notnull(valor):
      col_txt = color_b1_linea if b == bloques_unicos[0] else color_b2_linea
      ax1.annotate(
          f'{valor:.1f}%',
          (xi, valor),
          xytext=(0, 9),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=7,
          fontweight='bold',
          color=col_txt,
      )

  # Línea divisoria vertical y títulos superiores de los bloques
  if len(bloques_unicos) >= 2:
    idx_cambio = (sub_df[columna_bloque] != bloques_unicos[0]).idxmax() - 0.5
    ax1.axvline(
        x=idx_cambio,
        color='#94A3B8',
        linestyle=':',
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
    )

    pos_b1 = (sub_df[columna_bloque] == bloques_unicos[0]).sum() / 2 - 0.5
    pos_b2 = (
        sub_df[columna_bloque] == bloques_unicos[0]
    ).sum() + (sub_df[columna_bloque] == bloques_unicos[1]).sum() / 2 - 0.5

    ax1.text(
        pos_b1,
        ax1.get_ylim()[1] * 0.93,
        str(bloques_unicos[0]),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color=color_b1_linea,
    )
    ax1.text(
        pos_b2,
        ax1.get_ylim()[1] * 0.93,
        str(bloques_unicos[1]),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color=color_b2_linea,
    )

  # Configuración de ejes
  ax1.set_xticks(x)
  ax1.set_xticklabels(
      sub_df[columna_cluster], rotation=45, ha='right', fontsize=8
  )
  ax1.set_ylabel(
      'Efectividad (%)', fontsize=9, fontweight='bold', color='#1D4290'
  )
  ax1.tick_params(axis='both', labelsize=8, labelcolor='#1D4290')
  ax1.yaxis.set_major_formatter(
      plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
  )

  max_y = y_efectividad.max() if not y_efectividad.empty else 100
  ax1.set_ylim(0, max(max_y * 1.25, 30))

  # Limpieza de bordes estilo ejecutivo
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.spines['left'].set_color('#CBD5E1')
  ax1.spines['bottom'].set_color('#CBD5E1')
  ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray', zorder=1)

  # Título principal
  titulo = titulo or 'Efectividad y Tasa por Clúster Comercial'
  ax1.set_title(
      titulo,
      fontsize=12,
      fontweight='bold',
      color='#1D4290',
      pad=20,
  )

  plt.tight_layout()
  plt.show()

def _aplicar_estilo_ejecutivo(
    ax, titulo, ylabel, color_titulo='#1D4290', rotacion_x=90
):
  ax.set_title(
      titulo, fontsize=11, fontweight='bold', color=color_titulo, pad=15
  )
  ax.set_xlabel('')
  ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', color=color_titulo)
  ax.tick_params(
      axis='x', rotation=rotacion_x, labelsize=8, labelcolor='black'
  )
  ax.tick_params(axis='y', labelsize=8, labelcolor='black')
  ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')

  for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)


def plot_completitud_segmento(
    df,
    campo_cluster='cluster_comercial',
    campo_completitud='completitud',
    titulo='Completitud por Segmento',
    ylabel='Porcentaje (%)',
    color_barras='#27D75A',
    color_titulo='#1D4290',
    rotacion_x=90,
):
  # Hacemos una copia para no alterar el DataFrame original
  df_plot = df.copy()

  # Validación automática de escala (si el valor máximo es <= 1.0, se convierte a porcentaje)
  max_val = df_plot[campo_completitud].max()
  if max_val <= 1.0:
    df_plot[campo_completitud] = df_plot[campo_completitud] * 100

  # Ordenar de menor a mayor completitud
  df_sorted = df_plot.sort_values(by=campo_completitud, ascending=True)

  fig, ax = plt.subplots(figsize=(8, 4))

  barras = ax.bar(
      np.arange(len(df_sorted)),
      df_sorted[campo_completitud].values,
      width=1.0,
      color=color_barras,
      edgecolor='black',
      linewidth=0.6,
      alpha=0.9,
  )

  ax.set_xticks(np.arange(len(df_sorted)))
  ax.set_xticklabels(df_sorted[campo_cluster].values)

  ax.set_ylim(0, df_sorted[campo_completitud].max() * 1.18)

  for barra, valor in zip(barras, df_sorted[campo_completitud]):
    ax.annotate(
        f'{valor:.1f}%',
        (barra.get_x() + barra.get_width() / 2, valor),
        xytext=(0, 6),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=7.5,
        color='black',
        bbox=dict(
            boxstyle='round,pad=0.22', fc='#E8F8F5', ec='none', alpha=0.9
        ),
    )

  # Aplicar tu estilo ejecutivo
  _aplicar_estilo_ejecutivo(
      ax,
      titulo,
      ylabel,
      color_titulo=color_titulo,
      rotacion_x=rotacion_x,
  )

  plt.tight_layout()
  plt.show()


def plot_completitud_por_bloques_y_materialidad(
    df,
    columna_cluster='cluster_comercial',
    columna_bloque='categoria_bloque',  # Columna que divide en grupos (ej. Con deuda / Sin deuda)
    metrica_completitud='completitud',
    columna_materialidad='registros',
    titulo=None,
):
  """Genera un gráfico ejecutivo con barras de materialidad al fondo,

  puntos de completitud conectados por líneas y ordenados por bloques y
  completitud de mayor a menor, usando estrictamente los colores azul y verde
  corporativos.
  """
  if df.empty:
    print('El DataFrame no contiene registros para graficar')
    return

  sub_df = df.copy()

  # Asegurarnos de que la completitud esté en porcentaje
  if sub_df[metrica_completitud].max() <= 1.0:
    sub_df['__compl_plot'] = sub_df[metrica_completitud] * 100
  else:
    sub_df['__compl_plot'] = sub_df[metrica_completitud]

  # Ordenar dentro de cada bloque por completitud de mayor a menor
  if columna_bloque in sub_df.columns:
    sub_df = (
        sub_df.sort_values(
            by=[columna_bloque, '__compl_plot'], ascending=[True, False]
        )
        .reset_index(drop=True)
    )
  else:
    sub_df = sub_df.sort_values(by='__compl_plot', ascending=False).reset_index(
        drop=True
    )

  # Configurar figura bajo el estándar ejecutivo corporativo
  fig, ax1 = plt.subplots(figsize=(11, 5.8), dpi=300)
  fig.patch.set_facecolor('#F8F9FA')
  ax1.set_facecolor('#F8F9FA')

  x = np.arange(len(sub_df))

  # Eje secundario para las barras de fondo (Materialidad más atrás: zorder=0)
  ax_bar = ax1.twinx()
  ax_bar.set_zorder(0)
  ax_bar.set_zorder(1)
  ax_bar.patch.set_visible(False)

  valores_mat = sub_df[columna_materialidad].fillna(0)
  max_mat = valores_mat.max() if not valores_mat.empty else 1
  if max_mat == 0:
    max_mat = 1

  ax_bar.set_ylim(0, max_mat * 1.4)
  ax_bar.set_yticks([])

  # Definir los colores corporativos institucionales: Azul y Verde
  bloques_unicos = (
      sub_df[columna_bloque].unique()
      if columna_bloque in sub_df.columns
      else []
  )

  # Bloque 1: Azul corporativo (#1D4290) con barra translúcida suave (#D0E5F2)
  # Bloque 2: Verde corporativo (#27D75A) con barra translúcida suave (#D4EFDF)
  color_b1_barra, color_b1_linea = '#D0E5F2', '#1D4290'
  color_b2_barra, color_b2_linea = '#D4EFDF', '#1B9941'  # Verde institucional ajustado para contraste óptimo

  colores_barras = []
  if len(bloques_unicos) >= 2:
    b1, b2 = bloques_unicos[0], bloques_unicos[1]
    for b in sub_df[columna_bloque]:
      colores_barras.append(color_b1_barra if b == b1 else color_b2_barra)
  else:
    colores_barras = ['#D0E5F2'] * len(sub_df)

  barras = ax_bar.bar(
      x,
      valores_mat,
      width=0.6,
      color=colores_barras,
      alpha=0.5,
      edgecolor='white',
      zorder=0,
  )

  # Etiquetas de materialidad en entero sobre las barras
  for barra, valor in zip(barras, valores_mat):
    val_int = int(round(valor)) if pd.notnull(valor) else 0
    ax_bar.text(
        barra.get_x() + barra.get_width() / 2,
        max_mat * 0.02,
        f'n={val_int:,}',
        ha='center',
        va='bottom',
        fontsize=6.5,
        color='#64748B',
        alpha=0.9,
    )

  for spine in ax_bar.spines.values():
    spine.set_visible(False)

  # Configurar eje principal de completitud (zorder superior)
  ax1.set_zorder(3)
  ax1.patch.set_visible(False)

  y_completitud = sub_df['__compl_plot']

  # Dibujar líneas conectando los puntos y puntos agrupados por color del bloque correspondiente
  if len(bloques_unicos) >= 2:
    for b in bloques_unicos:
      mask = sub_df[columna_bloque] == b
      if mask.any():
        x_sub = x[mask]
        y_sub = y_completitud[mask]
        col_actual = color_b1_linea if b == b1 else color_b2_linea

        # Línea del bloque
        ax1.plot(
            x_sub,
            y_sub,
            color=col_actual,
            linestyle='-',
            linewidth=1.6,
            alpha=0.9,
            zorder=3,
        )
        # Puntos del bloque
        ax1.scatter(
            x_sub,
            y_sub,
            color=col_actual,
            s=75,
            zorder=4,
            edgecolors='white',
            linewidths=1.2,
        )
  else:
    ax1.plot(
        x,
        y_completitud,
        color=color_b1_linea,
        linestyle='-',
        linewidth=1.6,
        alpha=0.9,
        zorder=3,
    )
    ax1.scatter(
        x,
        y_completitud,
        color=color_b1_linea,
        s=75,
        zorder=4,
        edgecolors='white',
        linewidths=1.2,
    )

  # Etiquetas de porcentaje encima de cada punto con el color de su respectivo bloque
  for xi, valor, b in zip(
      x,
      y_completitud,
      sub_df[columna_bloque] if len(bloques_unicos) >= 2 else ['único'] * len(x),
  ):
    if pd.notnull(valor):
      col_txt = color_b1_linea if b == bloques_unicos[0] else color_b2_linea
      ax1.annotate(
          f'{valor:.1f}%',
          (xi, valor),
          xytext=(0, 9),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=7,
          fontweight='bold',
          color=col_txt,
      )

  # Línea divisoria vertical y títulos superiores de los bloques con sus colores coordinados
  if len(bloques_unicos) >= 2:
    idx_cambio = (sub_df[columna_bloque] != bloques_unicos[0]).idxmax() - 0.5
    ax1.axvline(
        x=idx_cambio,
        color='#94A3B8',
        linestyle=':',
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
    )

    pos_b1 = (sub_df[columna_bloque] == bloques_unicos[0]).sum() / 2 - 0.5
    pos_b2 = (
        sub_df[columna_bloque] == bloques_unicos[0]
    ).sum() + (sub_df[columna_bloque] == bloques_unicos[1]).sum() / 2 - 0.5

    ax1.text(
        pos_b1,
        ax1.get_ylim()[1] * 0.93,
        str(bloques_unicos[0]),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color=color_b1_linea,
    )
    ax1.text(
        pos_b2,
        ax1.get_ylim()[1] * 0.93,
        str(bloques_unicos[1]),
        ha='center',
        fontsize=9,
        fontweight='bold',
        color=color_b2_linea,
    )

  # Configuración de ejes
  ax1.set_xticks(x)
  ax1.set_xticklabels(
      sub_df[columna_cluster], rotation=45, ha='right', fontsize=8
  )
  ax1.set_ylabel(
      'Completitud (%)', fontsize=9, fontweight='bold', color='#1D4290'
  )
  ax1.tick_params(axis='both', labelsize=8, labelcolor='#1D4290')
  ax1.yaxis.set_major_formatter(
      plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
  )

  max_y = y_completitud.max() if not y_completitud.empty else 100
  ax1.set_ylim(0, max(max_y * 1.25, 30))

  # Limpieza de bordes estilo ejecutivo
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.spines['left'].set_color('#CBD5E1')
  ax1.spines['bottom'].set_color('#CBD5E1')
  ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray', zorder=1)

  # Título principal
  titulo = titulo or 'Completitud de Datos por Clúster Comercial y Materialidad'
  ax1.set_title(
      titulo,
      fontsize=12,
      fontweight='bold',
      color='#1D4290',
      pad=20,
  )

  plt.tight_layout()
  plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _formatear_periodo(val):
  """Convierte formato YYYYMM (ej. 202501) a Ene' 25"""
  val_str = str(val).strip()
  if len(val_str) == 6 and val_str.isdigit():
    anio = val_str[2:4]
    mes_num = int(val_str[4:6])
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
    return f"{meses.get(mes_num, 'Mes')}' {anio}"
  return val_str


# =====================================================================
# 1. GRÁFICO DE BARRAS SIMPLE POR PERIODO (CRONOLÓGICO Y CENTRADO)
# =====================================================================
def plot_completitud_periodo(
    df,
    campo_periodo='periodo',
    campo_completitud='completitud',
    campo_conteo='registros',
    titulo='Completitud por Periodo',
    ylabel='Porcentaje (%)',
    color_barras='#27D75A',
    color_titulo='#1D4290',
    rotacion_x=90,
):
  df_plot = df.copy()

  if df_plot[campo_completitud].max() <= 1.0:
    df_plot[campo_completitud] = df_plot[campo_completitud] * 100

  # Ordenamiento cronológico estricto
  df_sorted = df_plot.sort_values(by=campo_periodo, ascending=True)

  fig, ax = plt.subplots(figsize=(9, 4.5))

  barras = ax.bar(
      np.arange(len(df_sorted)),
      df_sorted[campo_completitud].values,
      width=0.65,
      color=color_barras,
      edgecolor='black',
      linewidth=0.6,
      alpha=0.9,
  )

  etiquetas_x = [
      f"{_formatear_periodo(row[campo_periodo])}\n(n={int(row[campo_conteo]):,})"
      for _, row in df_sorted.iterrows()
  ]

  ax.set_xticks(np.arange(len(df_sorted)))
  # CORREGIDO: ha='center' para alinear exactamente con el centro de las barras
  ax.set_xticklabels(etiquetas_x, rotation=rotacion_x, ha='center')

  ax.set_ylim(0, df_sorted[campo_completitud].max() * 1.22)

  for barra, valor in zip(barras, df_sorted[campo_completitud]):
    ax.annotate(
        f'{valor:.1f}%',
        (barra.get_x() + barra.get_width() / 2, valor),
        xytext=(0, 6),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=7.5,
        color='black',
        bbox=dict(
            boxstyle='round,pad=0.22', fc='#E8F8F5', ec='none', alpha=0.9
        ),
    )

  ax.set_title(
      titulo, fontsize=11, fontweight='bold', color=color_titulo, pad=15
  )
  ax.set_xlabel('')
  ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', color=color_titulo)
  ax.tick_params(
      axis='x', rotation=rotacion_x, labelsize=8, labelcolor='black'
  )
  ax.tick_params(axis='y', labelsize=8, labelcolor='black')
  ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')

  for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)

  plt.tight_layout()
  plt.show()


# =====================================================================
# 2. GRÁFICO DUAL DE LÍNEAS Y MATERIALIDAD (CRONOLÓGICO Y CENTRADO)
# =====================================================================
def plot_completitud_por_periodo_y_materialidad(
    df,
    columna_periodo='periodo',
    metrica_completitud='completitud',
    columna_materialidad='registros',
    titulo=None,
    rotacion_x=90,
):
  if df.empty:
    print('El DataFrame no contiene registros para graficar')
    return

  sub_df = df.copy()

  if sub_df[metrica_completitud].max() <= 1.0:
    sub_df['__compl_plot'] = sub_df[metrica_completitud] * 100
  else:
    sub_df['__compl_plot'] = sub_df[metrica_completitud]

  # Ordenamiento cronológico estricto
  sub_df = sub_df.sort_values(by=columna_periodo, ascending=True).reset_index(
      drop=True
  )

  fig, ax1 = plt.subplots(figsize=(10, 5.2), dpi=300)
  fig.patch.set_facecolor('#F8F9FA')
  ax1.set_facecolor('#F8F9FA')

  x = np.arange(len(sub_df))

  ax_bar = ax1.twinx()
  ax_bar.set_zorder(1)
  ax_bar.patch.set_visible(False)

  valores_mat = sub_df[columna_materialidad].fillna(0)
  max_mat = valores_mat.max() if not valores_mat.empty else 1
  if max_mat == 0:
    max_mat = 1

  ax_bar.set_ylim(0, max_mat * 1.4)
  ax_bar.set_yticks([])

  color_barra = '#D0E5F2'
  color_linea = '#1D4290'

  barras = ax_bar.bar(
      x,
      valores_mat,
      width=0.55,
      color=color_barra,
      alpha=0.6,
      edgecolor='white',
      zorder=0,
  )

  for barra, valor in zip(barras, valores_mat):
    val_int = int(round(valor)) if pd.notnull(valor) else 0
    ax_bar.text(
        barra.get_x() + barra.get_width() / 2,
        max_mat * 0.02,
        f'n={val_int:,}',
        ha='center',
        va='bottom',
        fontsize=7,
        color='#64748B',
        alpha=0.9,
    )

  for spine in ax_bar.spines.values():
    spine.set_visible(False)

  ax1.set_zorder(3)
  ax1.patch.set_visible(False)

  y_completitud = sub_df['__compl_plot']

  ax1.plot(
      x,
      y_completitud,
      color=color_linea,
      linestyle='-',
      linewidth=1.8,
      marker='o',
      markersize=6,
      markerfacecolor=color_linea,
      markeredgecolor='white',
      markeredgewidth=1.2,
      zorder=3,
  )

  for xi, valor in zip(x, y_completitud):
    if pd.notnull(valor):
      ax1.annotate(
          f'{valor:.1f}%',
          (xi, valor),
          xytext=(0, 9),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=7.5,
          fontweight='bold',
          color=color_linea,
      )

  etiquetas_x = [_formatear_periodo(p) for p in sub_df[columna_periodo]]

  ax1.set_xticks(x)
  # CORREGIDO: ha='center' para asegurar la simetría exacta con las líneas y puntos
  ax1.set_xticklabels(etiquetas_x, rotation=rotacion_x, ha='center', fontsize=8)

  ax1.set_ylabel(
      'Completitud (%)', fontsize=9, fontweight='bold', color='#1D4290'
  )
  ax1.tick_params(axis='y', labelsize=8, labelcolor='#1D4290')
  ax1.yaxis.set_major_formatter(
      plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
  )

  max_y = y_completitud.max() if not y_completitud.empty else 100
  ax1.set_ylim(0, max(max_y * 1.25, 30))

  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.spines['left'].set_color('#CBD5E1')
  ax1.spines['bottom'].set_color('#CBD5E1')
  ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray', zorder=2)

  titulo = titulo or 'Evolución de Completitud por Periodo y Volumen'
  ax1.set_title(
      titulo,
      fontsize=12,
      fontweight='bold',
      color='#1D4290',
      pad=20,
  )

  plt.tight_layout()
  plt.show()

def plot_pareto_instituciones(
    df,
    columna_institucion='cod_instit_financiera',
    columna_registros='registros',
    umbral=0.8,
    titulo=(
        'Diagrama de Pareto: Concentración de Registros por Entidad (Sin Filtro'
        ' por Entidad Financiera)'
    ),
    rotacion_x=90,
):
  """Genera un gráfico de Pareto dual completo con barras verdes, línea azul,

  etiquetas de ejes claras y líneas guía exactas formando la escuadra
  desde el borde izquierdo hasta el punto de quiebre.
  """
  if df.empty:
    print('El DataFrame no contiene registros para graficar.')
    return

  # Preparamos y ordenamos los datos de mayor a menor
  df_plot = df.copy()
  if columna_registros in df_plot.columns:
    df_plot = df_plot.sort_values(
        by=columna_registros, ascending=False
    ).reset_index(drop=True)

  if 'proporcion' not in df_plot.columns:
    total_reg = df_plot[columna_registros].sum()
    df_plot['proporcion'] = df_plot[columna_registros] / total_reg

  df_plot['cumsum_prop'] = df_plot['proporcion'].cumsum()

  x = np.arange(len(df_plot))
  y_registros = df_plot[columna_registros].values
  y_cumsum = df_plot['cumsum_prop'].values * 100  # En porcentaje

  # Figura y Ejes Duales
  fig, ax1 = plt.subplots(figsize=(11, 6.0), dpi=300)
  fig.patch.set_facecolor('#F8F9FA')
  ax1.set_facecolor('#F8F9FA')

  # Fijar límites explícitos en X para alinear el borde izquierdo de forma exacta (-0.5)
  ax1.set_xlim(-0.5, len(x) - 0.5)

  # Eje Derecho: Barras de Registros (n)
  ax_bar = ax1.twinx()
  ax_bar.set_zorder(1)
  ax_bar.patch.set_visible(False)

  max_reg = y_registros.max() if not df_plot.empty else 1
  ax_bar.set_ylim(0, max_reg * 1.35)

  color_barra = '#27AE60'  # Verde institucional
  barras = ax_bar.bar(
      x,
      y_registros,
      width=0.6,
      color=color_barra,
      alpha=0.65,
      edgecolor='white',
      zorder=1,
  )

  # Formato del Eje Derecho
  ax_bar.set_ylabel(
      'Número de Registros (n)',
      fontsize=9,
      fontweight='bold',
      color='#27AE60',
      labelpad=10,
  )
  ax_bar.tick_params(axis='y', labelsize=8, labelcolor='#27AE60')
  ax_bar.yaxis.set_major_formatter(
      plt.FuncFormatter(lambda val, _: f'{int(val):,}')
  )
  ax_bar.spines['top'].set_visible(False)
  ax_bar.spines['left'].set_visible(False)
  ax_bar.spines['right'].set_color('#CBD5E1')
  ax_bar.spines['bottom'].set_visible(False)

  # Etiqueta de conteo n sobre cada barra
  for barra, valor in zip(barras, y_registros):
    val_int = int(round(valor)) if pd.notnull(valor) else 0
    ax_bar.text(
        barra.get_x() + barra.get_width() / 2,
        valor + (max_reg * 0.015),
        f'{val_int:,}',
        ha='center',
        va='bottom',
        fontsize=7,
        color='#2C3E50',
    )

  # Eje Izquierdo: Curva de Pareto Acumulada (%)
  ax1.set_zorder(3)
  ax1.patch.set_visible(False)

  color_linea = '#1D4290'  # Azul institucional
  ax1.plot(
      x,
      y_cumsum,
      color=color_linea,
      linestyle='-',
      linewidth=2,
      marker='o',
      markersize=4.5,
      markerfacecolor='white',
      markeredgecolor=color_linea,
      markeredgewidth=1.5,
      zorder=3,
  )

  ax1.set_ylim(0, 108)
  ax1.set_ylabel(
      'Porcentaje Acumulado (%)', fontsize=9, fontweight='bold', color=color_linea
  )
  ax1.tick_params(axis='y', labelsize=8, labelcolor=color_linea)
  ax1.yaxis.set_major_formatter(
      plt.FuncFormatter(lambda val, _: f'{val:.0f}%')
  )

  # Encontrar el índice donde cruza o supera el umbral
  umbral_pct = umbral * 100 if umbral <= 1.0 else umbral
  idx_umbral = np.argmax(y_cumsum >= umbral_pct)

  if len(y_cumsum) > 0 and y_cumsum[-1] >= umbral_pct:
    x_val = x[idx_umbral]
    y_val = y_cumsum[idx_umbral]

    # Líneas guía exactas: horizontal desde el borde izquierdo (-0.5) y vertical desde el eje X (0)
    ax1.plot(
        [-0.5, x_val],
        [y_val, y_val],
        color='#64748B',
        linestyle='--',
        linewidth=1.2,
        zorder=2,
    )
    ax1.plot(
        [x_val, x_val],
        [0, y_val],
        color='#64748B',
        linestyle='--',
        linewidth=1.2,
        zorder=2,
    )

    # Anotación del punto de corte Pareto
    ax1.annotate(
        f'Umbral {umbral_pct:.0f}% (n={idx_umbral+1} entidades)',
        xy=(x_val, y_val),
        xytext=(x_val + 0.8, y_val - 10),
        arrowprops=dict(facecolor='#1D4290', shrink=0.05, width=1, headwidth=5),
        fontsize=8,
        fontweight='bold',
        color='#1D4290',
        bbox=dict(
            boxstyle='round,pad=0.3',
            fc='white',
            ec='#1D4290',
            alpha=0.9,
            linewidth=0.8,
        ),
    )

  # Configuración del Eje X y su Etiqueta
  entidades = df_plot[columna_institucion].astype(str).tolist()
  ax1.set_xticks(x)
  ax1.set_xticklabels(entidades, rotation=rotacion_x, ha='center', fontsize=8)
  ax1.set_xlabel(
      'Código de Institución Financiera',
      fontsize=9,
      fontweight='bold',
      color='#334155',
      labelpad=10,
  )

  # Estética general de los ejes
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.spines['left'].set_color('#CBD5E1')
  ax1.spines['bottom'].set_color('#CBD5E1')
  ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray', zorder=0)

  ax1.set_title(
      titulo, fontsize=11, fontweight='bold', color=color_linea, pad=20
  )

  plt.tight_layout()
  plt.show()

PALETA_15 = [
    '#E37222', '#329262', '#1D4290', '#8B0000', '#E69F00'
    '#0072B2', '#D55E00', '#CC79A7', '#56B4E9', '#009E73',
    '#27D75A', '#332288', '#117733', '#CC6677', '#AA4499'
]

def _formatear_periodos_panel(df, columna_periodo):
    sub_df = df.copy()
    meses = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
             7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
    sub_df['periodo_label'] = sub_df[columna_periodo].apply(
        lambda x: f"{meses[int(x) % 100]}' {str(int(x) // 100)[-2:]}"
    )
    return sub_df

def plot_evolutivo_multiserie_con_barras_apiladas(
    df,
    columna_periodo='periodo',
    columna_grupo='cluster_comercial',
    label_grupo='Cluster',
    metrica='tasa_desembolsada',
    label_metrica='Tasa Desembolsada',
    columna_materialidad='delta_deuda_final',
    modo_materialidad='unidad_monetaria',  # 'porcentaje', 'unidad_registros', 'unidad_monetaria'
    titulo=None,
    multiplicar_100='auto',                # 'auto', True o False
    paleta=PALETA_15,
):
    if df.empty:
        print('El DataFrame no contiene registros para graficar')
        return

    sub_df = _formatear_periodos_panel(df, columna_periodo)

    periodos_unicos = sorted(sub_df[columna_periodo].unique())
    mapa_x = {p: i for i, p in enumerate(periodos_unicos)}
    sub_df['x_idx'] = sub_df[columna_periodo].map(mapa_x)

    # Figura adaptada para leyenda inferior espaciosa
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax2 = ax.twinx()

    grupos = sorted(sub_df[columna_grupo].unique())

    pivot_mat = sub_df.pivot_table(
        index='x_idx', 
        columns=columna_grupo, 
        values=columna_materialidad, 
        aggfunc='sum', 
        fill_value=0
    ).reindex(range(len(periodos_unicos)), fill_value=0)

    if modo_materialidad == 'porcentaje':
        pivot_mat = pivot_mat.div(pivot_mat.sum(axis=1), axis=0).fillna(0) * 100
        label_mat_leg = f'Materialidad (%)'
    elif modo_materialidad == 'unidad_monetaria':
        pivot_mat = pivot_mat / 1e6  # Dividir entre 1 millón
        label_mat_leg = f'Materialidad ({columna_materialidad} en Millones)'
    else:  # 'unidad_registros'
        label_mat_leg = f'Materialidad ({columna_materialidad} Absolutos)'

    bottom = np.zeros(len(periodos_unicos))
    for i, grupo in enumerate(grupos):
        color = paleta[i % len(paleta)]
        valores_barra = pivot_mat[grupo].values
        
        ax2.bar(
            pivot_mat.index,
            valores_barra,
            bottom=bottom,
            color=color,
            alpha=0.20,  # Sombreado de fondo
            edgecolor='white',
            linewidth=0.5,
            width=0.6,
            zorder=1
        )
        bottom += valores_barra

    ax2.set_ylabel(label_mat_leg, fontsize=8, color='gray')
    ax2.tick_params(axis='y', labelsize=8, labelcolor='gray')
    ax2.grid(False)

    sample_vals = sub_df[metrica].dropna()
    if multiplicar_100 == 'auto':
        # Si la métrica viene en formato decimal (ej: 0.15), la convertimos a porcentaje (15%)
        multiplicar_100 = True if (not sample_vals.empty and sample_vals.max() <= 1.0) else False

    for i, grupo in enumerate(grupos):
        color = paleta[i % len(paleta)]
        df_grupo = sub_df[sub_df[columna_grupo] == grupo].sort_values(
            columna_periodo
        )

        y_vals = df_grupo[metrica] * 100 if multiplicar_100 else df_grupo[metrica]

        df_temp = pd.DataFrame(
            {'x': df_grupo['x_idx'], 'y': y_vals}
        ).dropna(subset=['y'])

        if df_temp.empty:
            continue

        df_temp['salto'] = df_temp['x'].diff().fillna(1).gt(1)
        grupo_tramo = df_temp['salto'].cumsum()

        # Tramos sólidos continuos
        for _, tramo in df_temp.groupby(grupo_tramo):
            ax.plot(
                tramo['x'],
                tramo['y'],
                color=color,
                linestyle='-',
                marker='o',
                markersize=3.5,
                linewidth=1.8,
                zorder=3,
            )

        # Puentes entrecortados cruzando vacíos
        indices_saltos = df_temp.index[df_temp['salto']].tolist()
        for idx in indices_saltos:
            pos_actual = df_temp.index.get_loc(idx)
            idx_anterior = df_temp.index[pos_actual - 1]

            x_prev, y_prev = df_temp.loc[idx_anterior, ['x', 'y']]
            x_curr, y_curr = df_temp.loc[idx, ['x', 'y']]

            ax.plot(
                [x_prev, x_curr],
                [y_prev, y_curr],
                color=color,
                linestyle='--',
                linewidth=1.4,
                zorder=2.5,
            )

        # Línea dummy para la leyenda inferior
        ax.plot([], [], color=color, linestyle='-', marker='o', label=str(grupo))

    df_labels = sub_df.drop_duplicates(columna_periodo).sort_values(
        columna_periodo
    )
    ax.set_xticks(df_labels['x_idx'])
    ax.set_xticklabels(df_labels['periodo_label'])
    ax.set_xlabel('Periodo', fontsize=9, fontweight='bold', color='#1D4290')
    
    final_label_metrica = f"{label_metrica} (%)" if multiplicar_100 else label_metrica
    ax.set_ylabel(final_label_metrica, fontsize=9, fontweight='bold', color='#1D4290')
    
    ax.tick_params(axis='both', labelsize=8, labelcolor='black')
    ax.tick_params(axis='x', rotation=90)

    if multiplicar_100:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda val, _: f'{val:.1f}%')
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    titulo = (
        titulo
        or f"Evolución de {label_metrica} con Materialidad por {columna_grupo.replace('_', ' ').capitalize()}"
    )
    ax.set_title(titulo, fontsize=11, fontweight='bold', color='#1D4290', pad=15)

    # Leyenda abajo, distribuida en columnas (máx 4 por fila)
    num_cols = min(len(grupos), 4)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=num_cols,
        frameon=True,
        facecolor='white',
        edgecolor='lightgray',
        fontsize=8,
        title=label_grupo
    )

    # Ajustamos márgenes para dar espacio a la leyenda inferior
    fig.subplots_adjust(bottom=0.32, right=0.90)
    plt.show()

