import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Calidad del Aire CDMX",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS GLOBALES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label { color: #7a8499 !important; font-size: 0.78rem; }

/* KPI cards */
.kpi-card {
    background: #1a1d2e;
    border: 1px solid #252840;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; font-weight: 600; color: #7eb8f7; }
.kpi-label { font-size: 0.72rem; color: #7a8499; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }
.kpi-delta { font-size: 0.8rem; margin-top: 0.15rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 2px; background: #0f1117; border-bottom: 1px solid #1e2130; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7a8499;
    border-radius: 6px 6px 0 0;
    padding: 0.6rem 1.2rem;
    font-size: 0.85rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] { background: #1a1d2e !important; color: #7eb8f7 !important; border-bottom: 2px solid #7eb8f7; }

/* Section headers */
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.2rem 0 0.5rem;
    border-left: 3px solid #7eb8f7;
    padding-left: 0.6rem;
}

/* Insight boxes */
.insight-box {
    background: #131625;
    border-left: 3px solid #7eb8f7;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.84rem;
    color: #a0aec0;
    margin: 0.8rem 0;
    line-height: 1.6;
}

/* Main background */
.main .block-container { background: #0b0e18; padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

CONTAMINANTES = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM25', 'SO2']
UNIDADES = {'CO': 'ppm', 'NO': 'ppb', 'NO2': 'ppb', 'NOX': 'ppb',
            'O3': 'ppb', 'PM10': 'µg/m³', 'PM25': 'µg/m³', 'SO2': 'ppb'}
COLORES_ESTACION = {'Invierno': '#4393c3', 'Primavera': '#4dac26',
                    'Verano':  '#d73027', 'Otoño':    '#f1a340'}
PALETA = px.colors.qualitative.Plotly

# ─────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv('rama_2023_05.csv', parse_dates=['fecha'], dayfirst=True)
    df['anio']     = df['fecha'].dt.year
    df['mes']      = df['fecha'].dt.month
    df['anio_mes'] = df['fecha'].dt.to_period('M').astype(str)
    df['estacion_anio'] = pd.cut(
        df['mes'], bins=[0, 3, 6, 9, 12],
        labels=['Invierno', 'Primavera', 'Verano', 'Otoño']
    ).astype(str)
    return df

@st.cache_data
def calcular_matriz_mensual(df):
    M = df.groupby('anio_mes')[CONTAMINANTES].mean()
    meta = df.groupby('anio_mes').agg(
        anio=('anio', 'first'),
        mes=('mes', 'first'),
        estacion=('estacion_anio', 'first')
    ).reset_index()
    return M, meta

@st.cache_data
def calcular_pca(M_std_arr):
    pca = PCA()
    scores = pca.fit_transform(M_std_arr)
    return scores, pca.components_, pca.explained_variance_ratio_

@st.cache_data
def calcular_mds(M_std_arr):
    dist_matrix = squareform(pdist(M_std_arr, metric='euclidean'))
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=42, normalized_stress='auto', n_init=1)
    coords = mds.fit_transform(dist_matrix)
    return coords, mds.stress_, dist_matrix

@st.cache_data
def calcular_sammon(dist_matrix, coords_init):
    def sammon_stress(coords_flat, dist_orig):
        n = dist_orig.shape[0]
        coords = coords_flat.reshape(n, 2)
        dist_proj = squareform(pdist(coords, metric='euclidean'))
        mask = dist_orig > 1e-10
        num = np.sum(((dist_orig[mask] - dist_proj[mask])**2) / dist_orig[mask])
        den = np.sum(dist_orig[mask])
        return num / den
    res = minimize(sammon_stress, coords_init.flatten(), args=(dist_matrix,),
                   method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-8})
    return res.x.reshape(-1, 2), res.fun

@st.cache_data
def calcular_espectro(serie_arr):
    n = len(serie_arr)
    t = np.arange(n)
    coef = np.polyfit(t, serie_arr, 1)
    tendencia = np.polyval(coef, t)
    serie_dt  = serie_arr - tendencia
    ventana   = np.hanning(n)
    fft_vals  = np.fft.rfft(serie_dt * ventana)
    freqs     = np.fft.rfftfreq(n, d=1.0)
    potencia  = (np.abs(fft_vals) ** 2) / n
    periodos  = np.where(freqs > 0, 1.0 / freqs, np.inf)
    return freqs, periodos, potencia, tendencia

df = cargar_datos()
M, meta = calcular_matriz_mensual(df)
scaler   = StandardScaler()
M_std    = scaler.fit_transform(M)
M_std_df = pd.DataFrame(M_std, index=M.index, columns=M.columns)

# ─────────────────────────────────────────────
# SIDEBAR — FILTROS GLOBALES
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ Calidad del Aire CDMX")
    st.markdown("<div style='color:#4a5568;font-size:0.75rem;margin-bottom:1.2rem;'>SEDEMA / SIMAT · 2015–2023</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-title'>Periodo</div>", unsafe_allow_html=True)
    anios_disp = sorted(df['anio'].unique())
    anio_rango = st.slider("Años", int(anios_disp[0]), int(anios_disp[-1]),
                            (int(anios_disp[0]), int(anios_disp[-1])))

    st.markdown("<div class='section-title'>Contaminantes</div>", unsafe_allow_html=True)
    conts_sel = st.multiselect("Seleccionar", CONTAMINANTES, default=CONTAMINANTES)
    if not conts_sel:
        conts_sel = CONTAMINANTES

    st.markdown("<div class='section-title'>Estaciones del año</div>", unsafe_allow_html=True)
    estaciones_sel = st.multiselect(
        "Filtrar",
        ['Invierno', 'Primavera', 'Verano', 'Otoño'],
        default=['Invierno', 'Primavera', 'Verano', 'Otoño']
    )
    if not estaciones_sel:
        estaciones_sel = ['Invierno', 'Primavera', 'Verano', 'Otoño']

    st.divider()
    st.markdown("<div style='color:#4a5568;font-size:0.7rem;'>Proyecto Final · Analítica y Visualización de Datos · ESCOM IPN</div>", unsafe_allow_html=True)

# Aplicar filtros
df_f = df[(df['anio'] >= anio_rango[0]) & (df['anio'] <= anio_rango[1]) &
          (df['estacion_anio'].isin(estaciones_sel))].copy()
meta_f = meta[(meta['anio'] >= anio_rango[0]) & (meta['anio'] <= anio_rango[1]) &
              (meta['estacion'].isin(estaciones_sel))].copy()
idx_f  = meta_f.index.tolist()
M_f    = M.iloc[idx_f] if idx_f else M
M_std_f = scaler.fit_transform(M_f) if len(M_f) > 2 else M_std

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.markdown("# 🌫️ Calidad del Aire — Ciudad de México")
st.markdown(f"<div style='color:#7a8499;font-size:0.85rem;margin-bottom:1.5rem;'>Red Automática de Monitoreo Atmosférico (RAMA) · SEDEMA / SIMAT · Periodo: {anio_rango[0]}–{anio_rango[1]}</div>", unsafe_allow_html=True)
st.info("""
**Fuente:** Red Automática de Monitoreo Atmosférico (RAMA)

**Institución:** Secretaría del Medio Ambiente (SEDEMA)

**Periodo de estudio:** 2015–2023

**Variables analizadas:** CO, NO, NO₂, NOX, O₃, PM10, PM2.5 y SO₂
""")
# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
st.markdown("## Panorama General")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Periodo",
        f"{df_f['anio'].min()}-{df_f['anio'].max()}"
    )
with col2:
    st.metric(
        "Registros",
        f"{len(df_f):,}"
    )
with col3:
    st.metric(
        "Meses",
        len(M_f)
    )
with col4:
    st.metric(
        "Contaminantes",
        len(CONTAMINANTES)
    )
    
st.markdown("## Indicadores Principales")

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "O₃ promedio",
        f"{df_f['O3'].mean():.1f} ppb"
    )
with col6:
    st.metric(
        "PM10 promedio",
        f"{df_f['PM10'].mean():.1f} µg/m³"
    )
with col7:
    st.metric(
        "PM2.5 promedio",
        f"{df_f['PM25'].mean():.1f} µg/m³"
    )
with col8:
    st.metric(
        "NO₂ promedio",
        f"{df_f['NO2'].mean():.1f} ppb"
    )
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Exploración y Preprocesamiento",
    "🌀  Análisis Espectral",
    "🔬  Análisis Multivariado y Correlación"
])

# ══════════════════════════════════════════════
# TAB 1 — EXPLORACIÓN Y PREPROCESAMIENTO
# ══════════════════════════════════════════════
with tab1:

    # ── Serie temporal ──
    st.markdown("<div class='section-title'>Serie temporal diaria</div>", unsafe_allow_html=True)
    cont_serie = st.selectbox("Contaminante", conts_sel, key='serie_cont')
    col_s1, col_s2 = st.columns([3, 1])

    with col_s1:
        fig_serie = go.Figure()
        fig_serie.add_trace(go.Scatter(
            x=df_f['fecha'], y=df_f[cont_serie],
            mode='lines', line=dict(color='#7eb8f7', width=1),
            opacity=0.6, name='Diario',
            hovertemplate='%{x|%d %b %Y}<br>%{y:.3f} ' + UNIDADES[cont_serie]
        ))
        # Media móvil 30 días
        ma30 = df_f.set_index('fecha')[cont_serie].rolling(30).mean()
        fig_serie.add_trace(go.Scatter(
            x=ma30.index, y=ma30.values,
            mode='lines', line=dict(color='#f1c40f', width=2),
            name='Media móvil 30d'
        ))
        # Marcar outliers IQR
        q1, q3 = df_f[cont_serie].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask_out = (df_f[cont_serie] < q1 - 1.5*iqr) | (df_f[cont_serie] > q3 + 1.5*iqr)
        fig_serie.add_trace(go.Scatter(
            x=df_f.loc[mask_out, 'fecha'], y=df_f.loc[mask_out, cont_serie],
            mode='markers', marker=dict(color='#e05252', size=4, symbol='circle'),
            name=f'Outliers IQR ({mask_out.sum()})'
        ))
        fig_serie.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=340, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation='h', y=-0.15),
            xaxis=dict(showgrid=True, gridcolor='#1e2130'),
            yaxis=dict(showgrid=True, gridcolor='#1e2130',
                       title=f'{cont_serie} ({UNIDADES[cont_serie]})')
        )
        st.plotly_chart(fig_serie, use_container_width=True)

    with col_s2:
        n_out = mask_out.sum()
        pct_out = mask_out.mean()*100
        st.markdown(f"""
        <div class='insight-box'>
        <b>Outliers detectados</b><br>
        Método IQR (factor 1.5)<br><br>
        <span style='font-family:monospace;font-size:1.1rem;color:#7eb8f7;'>{n_out}</span> días<br>
        <span style='color:#7a8499;'>{pct_out:.1f}% del total</span><br><br>
        Estos valores corresponden a eventos de alta concentración reales — no se eliminan, se etiquetan como contingencias ambientales.
        </div>
        """, unsafe_allow_html=True)

    # ── Distribuciones KDE ──
    st.markdown("<div class='section-title'>Distribuciones de densidad (KDE)</div>", unsafe_allow_html=True)
    conts_kde = st.multiselect("Contaminantes a comparar", conts_sel,
                                default=conts_sel[:4], key='kde_sel')
    if conts_kde:
        fig_kde = go.Figure()
        for i, cont in enumerate(conts_kde):
            vals = df_f[cont].dropna().values
            kde  = stats.gaussian_kde(vals)
            xs   = np.linspace(vals.min(), vals.max(), 200)
            fig_kde.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode='lines',
                name=f'{cont} ({UNIDADES[cont]})',
                line=dict(color=PALETA[i % len(PALETA)], width=2),
                fill='tozeroy', fillcolor=f'rgba({int(PALETA[i%len(PALETA)][1:3],16)},'
                                          f'{int(PALETA[i%len(PALETA)][3:5],16)},'
                                          f'{int(PALETA[i%len(PALETA)][5:7],16)},0.08)'
            ))
        fig_kde.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=320, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title='Concentración', yaxis_title='Densidad',
            legend=dict(orientation='h', y=-0.2),
            xaxis=dict(showgrid=True, gridcolor='#1e2130'),
            yaxis=dict(showgrid=True, gridcolor='#1e2130')
        )
        st.plotly_chart(fig_kde, use_container_width=True)

    # ── Estacionalidad ──
    st.markdown("<div class='section-title'>Estacionalidad mensual promedio</div>", unsafe_allow_html=True)
    cont_est = st.selectbox("Contaminante", conts_sel, key='est_cont')
    perfil_mes = df_f.groupby('mes')[cont_est].mean().reset_index()
    nombres_mes = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
                   7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
    perfil_mes['mes_str'] = perfil_mes['mes'].map(nombres_mes)

    fig_est = go.Figure()
    fig_est.add_trace(go.Bar(
        x=perfil_mes['mes_str'], y=perfil_mes[cont_est],
        marker_color='#7eb8f7', marker_opacity=0.8, name='Promedio mensual'
    ))
    fig_est.add_trace(go.Scatter(
        x=perfil_mes['mes_str'], y=perfil_mes[cont_est],
        mode='lines+markers', line=dict(color='#f1c40f', width=2),
        marker=dict(size=6), name='Tendencia'
    ))
    fig_est.update_layout(
        template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title=f'{cont_est} ({UNIDADES[cont_est]})',
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1e2130'),
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig_est, use_container_width=True)

    # ── Heatmap calendario ──
    st.markdown("<div class='section-title'>Heatmap año × mes</div>", unsafe_allow_html=True)
    cont_heat = st.selectbox("Contaminante", conts_sel, key='heat_cont')
    pivot_heat = df_f.groupby(['anio', 'mes'])[cont_heat].mean().unstack('mes').round(3)
    pivot_heat.columns = [nombres_mes[m] for m in pivot_heat.columns]

    fig_heat = px.imshow(
        pivot_heat, color_continuous_scale='RdYlGn_r',
        labels=dict(color=f'{cont_heat} ({UNIDADES[cont_heat]})'),
        aspect='auto'
    )
    fig_heat.update_layout(
        template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title='Mes', yaxis_title='Año'
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>Lectura del heatmap:</b> Cada celda es el promedio del contaminante en ese año-mes.
    Colores rojos indican meses con alta concentración (mayor riesgo para la salud),
    verdes indican meses más limpios. Los patrones verticales revelan estacionalidad;
    los patrones horizontales revelan tendencias interanuales.
    </div>
    """, unsafe_allow_html=True)

    # ── Boxplots comparativos ──
    st.markdown("<div class='section-title'>Distribución por estación del año</div>", unsafe_allow_html=True)
    cont_box = st.selectbox("Contaminante", conts_sel, key='box_cont')
    fig_box = px.box(
        df_f, x='estacion_anio', y=cont_box,
        color='estacion_anio',
        color_discrete_map=COLORES_ESTACION,
        category_orders={'estacion_anio': ['Invierno','Primavera','Verano','Otoño']},
        points='outliers'
    )
    fig_box.update_layout(
        template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        xaxis_title='Estación del año',
        yaxis_title=f'{cont_box} ({UNIDADES[cont_box]})',
        yaxis=dict(showgrid=True, gridcolor='#1e2130')
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — ANÁLISIS ESPECTRAL
# ══════════════════════════════════════════════
with tab2:

    st.markdown("<div class='section-title'>Espectro de potencia — FFT</div>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        cont_fft = st.selectbox("Contaminante", conts_sel, key='fft_cont')
    with col_f2:
        max_periodo = st.slider("Período máximo a mostrar (días)", 30, 400, 400, key='fft_max')

    serie = df_f[cont_fft].values
    freqs, periodos, potencia, tendencia = calcular_espectro(serie)

    # Serie temporal + tendencia
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Serie temporal y tendencia lineal**")
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=df_f['fecha'], y=serie,
            mode='lines', line=dict(color='#7eb8f7', width=1),
            opacity=0.5, name='Original'
        ))
        t_num = np.arange(len(serie))
        fig_t.add_trace(go.Scatter(
            x=df_f['fecha'], y=tendencia,
            mode='lines', line=dict(color='#e05252', width=2, dash='dash'),
            name='Tendencia lineal'
        ))
        fig_t.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation='h', y=-0.2),
            yaxis=dict(title=f'{cont_fft} ({UNIDADES[cont_fft]})',
                       showgrid=True, gridcolor='#1e2130'),
            xaxis=dict(showgrid=True, gridcolor='#1e2130')
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col_t2:
        st.markdown("**Espectro de potencia (FFT)**")
        mask = (freqs > 0) & (periodos <= max_periodo)
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(
            x=periodos[mask], y=potencia[mask],
            mode='lines', line=dict(color='#7eb8f7', width=1.5),
            fill='tozeroy', fillcolor='rgba(126,184,247,0.1)',
            name='Potencia', hovertemplate='Período: %{x:.1f} días<br>Potencia: %{y:.2f}'
        ))
        # Marcar ciclos clave
        for p_ref, label, col_ref in [(365,'Anual','#e05252'), (182,'Semianual','#f1a340'), (7,'Semanal','#52e07e')]:
            if p_ref <= max_periodo:
                fig_sp.add_vline(x=p_ref, line_dash='dash', line_color=col_ref,
                                  annotation_text=label, annotation_position='top',
                                  annotation_font_color=col_ref, annotation_font_size=10)
        fig_sp.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title='Período (días)', yaxis_title='Potencia espectral',
            xaxis=dict(showgrid=True, gridcolor='#1e2130'),
            yaxis=dict(showgrid=True, gridcolor='#1e2130')
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    # Tabla de frecuencias dominantes
    st.markdown("<div class='section-title'>Top frecuencias dominantes</div>", unsafe_allow_html=True)
    mask2 = freqs > 0
    top_idx = np.argsort(potencia[mask2])[::-1][:8]
    freqs_top  = freqs[mask2][top_idx]
    per_top    = periodos[mask2][top_idx]
    pot_top    = potencia[mask2][top_idx]

    def clasificar_ciclo(p):
        if 340 <= p <= 390: return '📅 Anual'
        if 160 <= p <= 200: return '📅 Semianual'
        if 6 <= p <= 8:     return '📅 Semanal'
        return '—'

    tabla_freq = pd.DataFrame({
        'Período (días)': per_top.round(1),
        'Frecuencia (1/día)': freqs_top.round(5),
        'Potencia': pot_top.round(2),
        'Ciclo': [clasificar_ciclo(p) for p in per_top]
    })
    st.dataframe(tabla_freq, use_container_width=True, hide_index=True)

    # Reconstrucción espectral
    st.markdown("<div class='section-title'>Reconstrucción con armónicos dominantes</div>", unsafe_allow_html=True)
    k_armonicos = st.slider("Número de armónicos (k)", 2, 30, 5, key='k_arm')

    def reconstruir(serie_arr, k):
        n = len(serie_arr)
        t = np.arange(n)
        coef = np.polyfit(t, serie_arr, 1)
        tend = np.polyval(coef, t)
        serie_dt = serie_arr - tend
        fft_v = np.fft.rfft(serie_dt)
        pot_v = np.abs(fft_v)**2
        top_k = np.argsort(pot_v)[::-1][:k]
        fft_fil = np.zeros_like(fft_v)
        fft_fil[top_k] = fft_v[top_k]
        return np.fft.irfft(fft_fil, n=n) + tend

    rec = reconstruir(serie, k_armonicos)

    fig_rec = go.Figure()
    fig_rec.add_trace(go.Scatter(
        x=df_f['fecha'], y=serie, mode='lines',
        line=dict(color='#7eb8f7', width=1), opacity=0.4, name='Original'
    ))
    fig_rec.add_trace(go.Scatter(
        x=df_f['fecha'], y=rec, mode='lines',
        line=dict(color='#f1a340', width=2), name=f'Reconstruida (k={k_armonicos})'
    ))
    # Error de reconstrucción
    rmse = np.sqrt(np.mean((serie - rec)**2))
    fig_rec.update_layout(
        template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation='h', y=-0.2),
        title=dict(text=f'RMSE de reconstrucción: {rmse:.4f} {UNIDADES[cont_fft]}',
                   font=dict(size=12, color='#7a8499'), x=0),
        yaxis=dict(title=f'{cont_fft} ({UNIDADES[cont_fft]})',
                   showgrid=True, gridcolor='#1e2130'),
        xaxis=dict(showgrid=True, gridcolor='#1e2130')
    )
    st.plotly_chart(fig_rec, use_container_width=True)

    st.markdown(f"""
    <div class='insight-box'>
    <b>Ciclos dominantes en {cont_fft}:</b> La FFT descompone la serie en sus frecuencias
    constitutivas. El pico anual (~365 días) confirma la estacionalidad climática.
    El ciclo semanal (~7 días) refleja patrones de movilidad — menor tráfico los fines de semana.
    Con solo k={k_armonicos} armónicos se captura la estructura principal, filtrando el ruido diario.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — ANÁLISIS MULTIVARIADO Y CORRELACIÓN
# ══════════════════════════════════════════════
with tab3:

    # ── PCA ──
    st.markdown("<div class='section-title'>Análisis de Componentes Principales (PCA)</div>", unsafe_allow_html=True)

    scores, loadings, varianza = calcular_pca(tuple(map(tuple, M_std_f)))

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        # Scree plot
        acum = np.cumsum(varianza) * 100
        fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scree.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(varianza))],
            y=varianza * 100, name='Varianza (%)',
            marker_color='#7eb8f7', opacity=0.8
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(acum))],
            y=acum, mode='lines+markers', name='Acumulada (%)',
            line=dict(color='#f1a340', width=2), marker=dict(size=6)
        ), secondary_y=True)
        fig_scree.add_hline(y=90, line_dash='dash', line_color='#e05252',
                             annotation_text='90%', secondary_y=True)
        fig_scree.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation='h', y=-0.2),
            yaxis=dict(title='Varianza explicada (%)', showgrid=True, gridcolor='#1e2130'),
            yaxis2=dict(title='Acumulada (%)')
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    with col_p2:
        # Heatmap de loadings
        load_df = pd.DataFrame(
            loadings[:4],
            index=[f'PC{i+1} ({varianza[i]*100:.1f}%)' for i in range(4)],
            columns=CONTAMINANTES
        ).round(3)
        fig_load = px.imshow(
            load_df, color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1, text_auto='.2f', aspect='auto'
        )
        fig_load.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title='Loading')
        )
        st.plotly_chart(fig_load, use_container_width=True)

    # Biplot interactivo
    st.markdown("**Biplot PCA — PC1 vs PC2**")
    n_scores = min(len(scores), len(meta_f))
    meta_plot = meta_f.iloc[:n_scores].copy()
    meta_plot['PC1'] = scores[:n_scores, 0]
    meta_plot['PC2'] = scores[:n_scores, 1]
    meta_plot['anio_mes_str'] = M_f.index[:n_scores]

    fig_biplot = go.Figure()
    for est, color in COLORES_ESTACION.items():
        mask_e = meta_plot['estacion'] == est
        if mask_e.any():
            fig_biplot.add_trace(go.Scatter(
                x=meta_plot.loc[mask_e, 'PC1'],
                y=meta_plot.loc[mask_e, 'PC2'],
                mode='markers', name=est,
                marker=dict(color=color, size=9, opacity=0.85,
                             line=dict(color='white', width=0.5)),
                text=meta_plot.loc[mask_e, 'anio_mes_str'],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}'
            ))
    # Flechas de loadings
    escala = 3.0
    for j, cont in enumerate(CONTAMINANTES):
        fig_biplot.add_annotation(
            x=loadings[0, j]*escala, y=loadings[1, j]*escala,
            ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
            arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor='#e0e0e0',
            font=dict(color='#e0e0e0', size=10),
            text=f'<b>{cont}</b>',
            showarrow=True
        )
    fig_biplot.add_hline(y=0, line_dash='dash', line_color='#333', line_width=1)
    fig_biplot.add_vline(x=0, line_dash='dash', line_color='#333', line_width=1)
    fig_biplot.update_layout(
        template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(title='Estación', orientation='h', y=-0.15),
        xaxis=dict(title=f'PC1 ({varianza[0]*100:.1f}% var.)',
                   showgrid=True, gridcolor='#1e2130'),
        yaxis=dict(title=f'PC2 ({varianza[1]*100:.1f}% var.)',
                   showgrid=True, gridcolor='#1e2130')
    )
    st.plotly_chart(fig_biplot, use_container_width=True)

    n_comp_90 = int(np.argmax(np.cumsum(varianza) >= 0.90) + 1)
    st.markdown(f"""
    <div class='insight-box'>
    <b>PCA:</b> PC1 ({varianza[0]*100:.1f}%) y PC2 ({varianza[1]*100:.1f}%) acumulan
    el {(varianza[0]+varianza[1])*100:.1f}% de la varianza total.
    Se necesitan {n_comp_90} componentes para superar el 90%.
    La separación por estaciones confirma que el patrón estacional es la principal
    fuente de variabilidad en la contaminación de la CDMX.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── MDS + Sammon ──
    st.markdown("<div class='section-title'>Escalamiento Multidimensional (MDS) y Mapeo de Sammon</div>", unsafe_allow_html=True)

    with st.spinner('Calculando MDS y Sammon...'):
        coords_mds, stress_mds, dist_mat = calcular_mds(tuple(map(tuple, M_std_f)))
        coords_sammon, stress_sammon     = calcular_sammon(dist_mat, coords_mds)

    col_m1, col_m2 = st.columns(2)

    for col, coords, titulo, stress_val in zip(
        [col_m1, col_m2],
        [coords_mds, coords_sammon],
        ['MDS Métrico', 'Mapeo de Sammon'],
        [stress_mds, stress_sammon]
    ):
        n_c = min(len(coords), len(meta_f))
        mp  = meta_f.iloc[:n_c].copy()
        mp['D1'] = coords[:n_c, 0]
        mp['D2'] = coords[:n_c, 1]
        mp['anio_mes_str'] = M_f.index[:n_c]

        fig_m = go.Figure()
        for est, color in COLORES_ESTACION.items():
            mask_e = mp['estacion'] == est
            if mask_e.any():
                fig_m.add_trace(go.Scatter(
                    x=mp.loc[mask_e, 'D1'], y=mp.loc[mask_e, 'D2'],
                    mode='markers', name=est,
                    marker=dict(color=color, size=9, opacity=0.85,
                                 line=dict(color='white', width=0.5)),
                    text=mp.loc[mask_e, 'anio_mes_str'],
                    hovertemplate='<b>%{text}</b><br>D1: %{x:.2f}<br>D2: %{y:.2f}'
                ))
        fig_m.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=340, margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text=f'{titulo} · Stress = {stress_val:.4f}',
                       font=dict(size=11, color='#7a8499')),
            legend=dict(orientation='h', y=-0.18),
            xaxis=dict(title='Dimensión 1', showgrid=True, gridcolor='#1e2130'),
            yaxis=dict(title='Dimensión 2', showgrid=True, gridcolor='#1e2130')
        )
        col.plotly_chart(fig_m, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>MDS vs Sammon:</b> Ambos proyectan los perfiles mensuales en 2D.
    MDS minimiza el error global de distancias; Sammon pondera más los pares cercanos,
    preservando mejor la estructura local. Si los clusters son más compactos en Sammon,
    significa que los meses similares son <i>más</i> similares entre sí de lo que
    sugiere una vista global.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Correlación ──
    st.markdown("<div class='section-title'>Estimación de correlación</div>", unsafe_allow_html=True)

    corr_matrix = df_f[conts_sel].corr(method='pearson')
    n_obs = len(df_f)

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_masked = corr_matrix.copy().astype(float)
        corr_masked[mask_tri] = np.nan

        fig_corr = px.imshow(
            corr_masked, color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1, text_auto='.2f', aspect='auto'
        )
        fig_corr.update_layout(
            template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title='r Pearson')
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_c2:
        # Top correlaciones con p-values
        pares = []
        for i, c1 in enumerate(conts_sel):
            for j, c2 in enumerate(conts_sel):
                if i < j:
                    r, p = stats.pearsonr(df_f[c1], df_f[c2])
                    pares.append({'Par': f'{c1} – {c2}', 'r': round(r, 4),
                                   'p-value': f'{p:.2e}',
                                   'Sig.': '✓' if p < 0.05 else '✗'})
        pares_df = pd.DataFrame(pares).sort_values('r', key=abs, ascending=False)
        st.markdown("**Tabla de correlaciones (ordenada por |r|)**")
        st.dataframe(pares_df, use_container_width=True, hide_index=True, height=340)

    # Correlación cruzada interactiva
    st.markdown("<div class='section-title'>Correlación cruzada con desfase temporal</div>", unsafe_allow_html=True)
    col_cc1, col_cc2, col_cc3 = st.columns(3)
    with col_cc1:
        cont_x = st.selectbox("Contaminante X", conts_sel, index=conts_sel.index('NOX') if 'NOX' in conts_sel else 0, key='ccx')
    with col_cc2:
        cont_y = st.selectbox("Contaminante Y", conts_sel, index=conts_sel.index('O3') if 'O3' in conts_sel else 1, key='ccy')
    with col_cc3:
        max_lag = st.slider("Desfase máximo (días)", 7, 60, 30, key='lag')

    x_n = (df_f[cont_x] - df_f[cont_x].mean()) / df_f[cont_x].std()
    y_n = (df_f[cont_y] - df_f[cont_y].mean()) / df_f[cont_y].std()
    n_cc = len(x_n)
    lags = np.arange(-max_lag, max_lag+1)
    xcorr = [
        x_n.iloc[max(0,-lag):n_cc-max(0,lag)].values @
        y_n.iloc[max(0,lag):n_cc-max(0,-lag)].values / n_cc
        for lag in lags
    ]
    conf_95 = 1.96 / np.sqrt(n_cc)
    lag_max = lags[np.argmax(np.abs(xcorr))]

    fig_cc = go.Figure()
    colores_bars = ['#e05252' if abs(v) > conf_95 else '#4a5568' for v in xcorr]
    fig_cc.add_trace(go.Bar(
        x=lags, y=xcorr, marker_color=colores_bars,
        name='Correlación cruzada',
        hovertemplate='Lag: %{x}d<br>Corr: %{y:.4f}'
    ))
    fig_cc.add_hline(y=conf_95,  line_dash='dash', line_color='#f1a340',
                      annotation_text='IC 95%', annotation_font_size=10)
    fig_cc.add_hline(y=-conf_95, line_dash='dash', line_color='#f1a340')
    fig_cc.add_hline(y=0, line_color='#555', line_width=1)
    fig_cc.update_layout(
        template='plotly_dark', paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=f'Desfase (días)  [negativo = {cont_x} precede a {cont_y}]',
        yaxis_title='Correlación cruzada',
        xaxis=dict(showgrid=True, gridcolor='#1e2130'),
        yaxis=dict(showgrid=True, gridcolor='#1e2130'),
        title=dict(text=f'Desfase de mayor correlación: {lag_max} días',
                   font=dict(size=11, color='#7a8499'), x=0)
    )
    st.plotly_chart(fig_cc, use_container_width=True)

    sig_count = sum(1 for v in xcorr if abs(v) > conf_95)
    st.markdown(f"""
    <div class='insight-box'>
    <b>{cont_x} → {cont_y}:</b> El desfase de mayor correlación es <b>{lag_max} días</b>.
    {'Un lag negativo indica que ' + cont_x + ' precede a ' + cont_y + ' — consistente con la química fotoquímica donde los precursores NOX generan O3 con cierto retraso temporal.' if lag_max < 0 else 'Un lag positivo indica que ' + cont_y + ' precede a ' + cont_x + '.'}
    Las barras rojas ({sig_count} de {len(lags)}) superan las bandas de confianza al 95% — correlación estadísticamente significativa.
    </div>
    """, unsafe_allow_html=True)

