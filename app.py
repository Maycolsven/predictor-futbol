"""Predictor de fútbol europeo — interfaz en Streamlit.

Uso: streamlit run app.py
"""
import html
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, registro
from src.poisson import LINEAS
from src.prediccion import Predictor

st.set_page_config(page_title="Predictor Fútbol", page_icon=":material/sports_soccer:", layout="wide",
                   initial_sidebar_state="collapsed")

NOMBRE_LIGA = {c: n for c, (n, _, _) in config.LIGAS.items()} | {"CL": "Champions League"}
COLOR = {"L": "#22c55e", "E": "#6b7280", "V": "#ef4444"}
PCT = st.column_config.NumberColumn(format="%.0f%%")
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
ETIQUETA_MODELO = {"Elo": "Elo", "Poisson": "Poisson", "ML": "Machine learning",
                   "Final": "Final (stacking, lo que usa la app)", "Base": "Base (siempre la frecuencia)",
                   "Casas": "Casas de apuestas", "Sin lesionados": "Final sin ajuste por lesionados",
                   "Final (con cuotas)": "Final (mismos partidos)"}
EN_NUBE = os.environ.get("HOME", "").startswith("/home/adminuser")  # Streamlit Community Cloud
CUOTAS = ("cuota_l", "cuota_e", "cuota_v", "cuota_o25", "cuota_u25")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption, button, input {font-family: 'Inter', sans-serif !important;}
.stAppDeployButton, footer {display: none;}
.block-container {padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1100px;}
h1, h2, h3, h4 {letter-spacing: -0.01em;}
.encabezado {display: flex; justify-content: space-between; align-items: baseline; padding: 4px 0 10px 0;
             border-bottom: 1px solid #1f2b45; margin-bottom: 8px;}
.encabezado .titulo {font-size: 1.35rem; font-weight: 700;}
.encabezado .estado {font-size: 0.78rem; color: #94a3b8;}
.seccion {font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8;
          margin: 14px 0 6px 0;}
.leyenda {font-size: 0.8rem; color: #94a3b8; margin-bottom: 10px;}
.punto {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin: 0 4px 0 10px; vertical-align: middle;}
.tarjeta {background: #151f33; border: 1px solid #1f2b45; border-radius: 12px; padding: 12px 14px 10px 14px;
          margin-bottom: 6px;}
.tarjeta .meta {font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px; display: flex; justify-content: space-between;}
.tarjeta .meta .liga {color: #cbd5e1; font-weight: 500;}
.tarjeta .equipos {display: flex; justify-content: space-between; align-items: center; font-weight: 600;
                   font-size: 1rem; margin-bottom: 8px;}
.tarjeta .equipos .vs {color: #475569; font-weight: 400; font-size: 0.75rem; padding: 0 8px;}
.tarjeta .equipos .visita {text-align: right;}
.barra {display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #0b1220;}
.barra span {display: block; height: 100%;}
.probs {display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 4px; color: #94a3b8;
        font-variant-numeric: tabular-nums;}
.tarjeta .pie {display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.82rem; color: #cbd5e1;
               font-variant-numeric: tabular-nums;}
.tarjeta .pie .fav {font-weight: 600; color: #e5e7eb;}
.tarjeta .pie .fav .punto {margin-left: 0;}
.marcador {display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 12px;
           background: #151f33; border: 1px solid #1f2b45; border-radius: 12px; padding: 16px 18px; margin: 4px 0 10px 0;}
.marcador .nombre {font-size: 1.15rem; font-weight: 600;}
.marcador .pct {font-size: 2rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums;}
.marcador .centro {text-align: center; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;}
.marcador .centro .empate {font-size: 1.25rem; font-weight: 600; color: #cbd5e1; letter-spacing: 0;}
.marcador .visita {text-align: right;}
.chips {display: flex; gap: 4px; margin: 4px 0 8px 0;}
.chip {display: inline-flex; width: 26px; height: 26px; border-radius: 6px; align-items: center; justify-content: center;
       font-size: 0.72rem; font-weight: 700; color: #0b1220;}
.chip.g {background: #22c55e;} .chip.e {background: #94a3b8;} .chip.p {background: #ef4444;}
.linea {font-size: 0.82rem; color: #cbd5e1; margin: 2px 0;}
.linea b {display: inline-block; width: 14px; font-weight: 700;}
.linea b.g {color: #22c55e;} .linea b.e {color: #94a3b8;} .linea b.p {color: #ef4444;}
div[data-testid="stButton"] > button {border-radius: 8px; font-weight: 500;}
div[data-testid="stMetric"] {background: #151f33; border: 1px solid #1f2b45; border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetricLabel"] {color: #94a3b8;}
</style>
""", unsafe_allow_html=True)


def seccion(texto):
    st.markdown(f'<div class="seccion">{texto}</div>', unsafe_allow_html=True)


def leyenda_html():
    return ('<div class="leyenda">Barra de probabilidades:'
            + "".join(f'<span class="punto" style="background:{COLOR[k]}"></span>{t}'
                      for k, t in (("L", "gana el local"), ("E", "empate"), ("V", "gana la visita")))
            + ' &nbsp;·&nbsp; <b>+2.5</b> = probabilidad de 3 goles o más &nbsp;·&nbsp; último número = marcador más probable</div>')


# ---------------------------------------------------------------- datos

def version_datos():
    """Cambia cada vez que se corre actualizar.py, así la caché se renueva sola."""
    ruta = config.DATA / "modelo.json"
    return ruta.stat().st_mtime if ruta.exists() else 0.0


@st.cache_resource
def cargar_predictor(version):  # sin "_": Streamlit ignora los parámetros que empiezan con "_"
    try:
        return Predictor()
    except (FileNotFoundError, KeyError):
        return None


@st.cache_data
def cargar_datos(version):
    leer = lambda n, **kw: pd.read_csv(config.DATA / n, **kw)  # noqa: E731
    partidos = leer("partidos.csv", parse_dates=["fecha"])
    historial = leer("historial_elo.csv", parse_dates=["fecha"])
    fixtures = leer("fixtures.csv", parse_dates=["fecha"]) if (config.DATA / "fixtures.csv").exists() else pd.DataFrame()
    reg = leer("predicciones.csv", parse_dates=["fecha"]) if registro.RUTA.exists() else pd.DataFrame()
    return partidos, historial, fixtures, reg


@st.cache_data
def predecir_fixtures(_pred, version, ligas, dias):
    """Predicción de todos los próximos partidos (se calcula una vez por filtro)."""
    hoy = pd.Timestamp.today().normalize()
    fx = fixtures[fixtures["competicion"].isin(ligas) & (fixtures["fecha"] >= hoy)
                  & (fixtures["fecha"] < hoy + pd.Timedelta(days=dias))]
    salida = []
    for r in fx.itertuples():
        if not (_pred.conoce(r.local) and _pred.conoce(r.visita)):
            continue
        cuotas = {c: getattr(r, c, np.nan) for c in CUOTAS}
        salida.append((r._asdict(), _pred.partido(r.local, r.visita, r.competicion, fecha=r.fecha, cuotas=cuotas)))
    return salida


# ---------------------------------------------------------------- piezas visuales

def favorito(p, local, visita):
    """(texto, clave de color): 'Real Madrid 58%' o 'Parejo' si nadie pasa de 45% con ventaja clara."""
    pl, pe, pv = p["L"], p["E"], p["V"]
    if max(pl, pv) < 0.45 and abs(pl - pv) < 0.08:
        return "Parejo", "E"
    return (f"{local} {pl:.0%}", "L") if pl > pv else (f"{visita} {pv:.0%}", "V")


def favorito_html(p, local, visita):
    texto, k = favorito(p, local, visita)
    return f'<span class="fav"><span class="punto" style="background:{COLOR[k]}"></span>{html.escape(texto)}</span>'


def barra_html(p):
    return ('<div class="barra">' + "".join(
        f'<span style="width:{p[k] * 100:.1f}%;background:{COLOR[k]}"></span>' for k in "LEV") + "</div>"
        f'<div class="probs"><span>{p["L"]:.0%}</span><span>{p["E"]:.0%}</span><span>{p["V"]:.0%}</span></div>')


def tarjeta_html(r, p):
    gl, gv, _ = p["top"][0]
    hora = "" if pd.isna(r["hora"]) else f" · {r['hora']}"
    return f"""
<div class="tarjeta">
  <div class="meta"><span>{DIAS_SEMANA[r['fecha'].weekday()]} {r['fecha']:%d/%m}{hora}</span>
       <span class="liga">{NOMBRE_LIGA[r['competicion']]}</span></div>
  <div class="equipos"><span class="local">{html.escape(r['local'])}</span><span class="vs">vs</span>
       <span class="visita">{html.escape(r['visita'])}</span></div>
  {barra_html(p['1x2'])}
  <div class="pie">{favorito_html(p['1x2'], r['local'], r['visita'])}
       <span>+2.5 &nbsp;{p['over']['2.5']:.0%}</span><span>{gl}-{gv}</span></div>
</div>"""


def marcador_html(local, visita, p):
    pl, pe, pv = p["1x2"]["L"], p["1x2"]["E"], p["1x2"]["V"]
    return f"""
<div class="marcador">
  <div><div class="nombre">{html.escape(local)}</div><div class="pct" style="color:{COLOR['L']}">{pl:.0%}</div></div>
  <div class="centro">empate<div class="empate">{pe:.0%}</div></div>
  <div class="visita"><div class="nombre">{html.escape(visita)}</div><div class="pct" style="color:{COLOR['V']}">{pv:.0%}</div></div>
</div>"""


def forma_html(partidos, equipo, n=5):
    """Fichas G/E/P de los últimos partidos + una línea por partido."""
    p = partidos[(partidos["local"] == equipo) | (partidos["visita"] == equipo)].tail(n).iloc[::-1]
    chips, lineas = [], []
    for r in p.itertuples():
        es_local = r.local == equipo
        gf, gc = (r.gl, r.gv) if es_local else (r.gv, r.gl)
        k = "g" if gf > gc else "e" if gf == gc else "p"
        rival = r.visita if es_local else r.local
        chips.append(f'<span class="chip {k}">{k.upper()}</span>')
        lineas.append(f'<div class="linea"><b class="{k}">{k.upper()}</b> {gf}-{gc} {"vs" if es_local else "en"} '
                      f'{html.escape(rival)} <span style="color:#64748b">· {NOMBRE_LIGA.get(r.competicion, r.competicion)}'
                      f' · {r.fecha:%d/%m/%y}</span></div>')
    return f'<div class="chips">{"".join(chips)}</div>' + "".join(lineas)


def mapa_marcadores(matriz, local, visita):
    m = np.array(matriz) * 100
    goles = [str(i) for i in range(m.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=m, x=goles, y=goles, colorscale=[[0, "#151f33"], [1, "#22c55e"]], showscale=False,
        text=[[f"{v:.1f}%" for v in fila] for fila in m], texttemplate="%{text}",
        hovertemplate=f"{local} %{{y}} - %{{x}} {visita}: %{{z:.1f}}%<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title=f"Goles de {visita}", side="top"),
                      yaxis=dict(title=f"Goles de {local}", autorange="reversed"))
    return fig


def grafico_elo(historial, local, visita):
    desde = historial["fecha"].max() - pd.DateOffset(years=4)
    fig = go.Figure()
    for equipo, color in ((local, COLOR["L"]), (visita, COLOR["V"])):
        h = historial[(historial["equipo"] == equipo) & (historial["fecha"] >= desde)]
        fig.add_scatter(x=h["fecha"], y=h["elo"], name=equipo, mode="lines", line=dict(color=color, width=2))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1))
    return fig


def grafico_calibracion(cal, eje_x):
    cal = pd.DataFrame(cal)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfecto")
    fig.add_scatter(x=cal["pred"], y=cal["real"], mode="markers+lines", name="Modelo", line=dict(color=COLOR["L"]),
                    marker=dict(size=[max(6, min(22, n / 25)) for n in cal["n"]]),
                    text=[f"{n} partidos" for n in cal["n"]])
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title=eje_x, tickformat=".0%"), yaxis=dict(title="Lo que pasó de verdad", tickformat=".0%"))
    return fig


def tabla_metricas(d, con_brier=True):
    df = pd.DataFrame([{"Modelo": ETIQUETA_MODELO.get(n, n), "Log-loss ↓": m["log_loss"], "Brier ↓": m["brier"],
                        "Acierto": m["acierto"] * 100, "Partidos": m["partidos"]} for n, m in d.items()])
    if not con_brier:
        df = df.drop(columns="Brier ↓")
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Log-loss ↓": st.column_config.NumberColumn(format="%.4f"),
                                "Brier ↓": st.column_config.NumberColumn(format="%.4f"),
                                "Acierto": st.column_config.NumberColumn(format="%.1f%%")})


def tabla_lesionados(imp):
    j = imp["jugadores"]
    return pd.DataFrame({"Jugador": j["jugador"], "Posición": j["posicion"], "Lesión": j["lesion"],
                         "Vuelve": j["hasta"].dt.strftime("%d/%m/%y").fillna("?"),
                         "Valor (M€)": j["valor"] / 1e6})


def mostrar_detalle(p, local, visita, cuotas=None, clave=""):
    """Todo el detalle de un partido (se usa en la ventana de cada tarjeta y en Enfrentamiento)."""
    st.markdown(marcador_html(local, visita, p), unsafe_allow_html=True)
    pe, pp, pm = p["elo_1x2"], p["poisson_1x2"], p["ml_1x2"]
    el, ev_ = p["elo_ajustado"]
    fmt = lambda d: f"{d['L']:.0%} / {d['E']:.0%} / {d['V']:.0%}"  # noqa: E731
    texto = (f"**Elo** {el:.0f} vs {ev_:.0f} → {fmt(pe)} · **Poisson** → {fmt(pp)} · **ML** → {fmt(pm)} · "
             f"Cuota justa: 1 = {1 / p['1x2']['L']:.2f} · X = {1 / p['1x2']['E']:.2f} · 2 = {1 / p['1x2']['V']:.2f}")
    if cuotas and all(pd.notna(cuotas.get(c)) for c in ("cuota_l", "cuota_e", "cuota_v")):
        inv = [1 / cuotas["cuota_l"], 1 / cuotas["cuota_e"], 1 / cuotas["cuota_v"]]
        c = [x / sum(inv) for x in inv]
        texto += f" · **Casas** → {c[0]:.0%} / {c[1]:.0%} / {c[2]:.0%}"
    if p["competicion"] == "CL":
        texto += " · cruce entre ligas (se calcula como partido de Champions)"
    st.caption(texto)

    seccion("Goles")
    ll, lv = p["goles"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(f"Esperados {local}", f"{ll:.2f}")
    g2.metric(f"Esperados {visita}", f"{lv:.2f}")
    g3.metric("Más de 2.5 goles", f"{p['over']['2.5']:.0%}")
    g4.metric("Ambos marcan", f"{p['btts']:.0%}")
    izq, der = st.columns([3, 2])
    with izq:
        st.plotly_chart(mapa_marcadores(p["matriz"], local, visita), width="stretch",
                        config={"displayModeBar": False}, key=f"mapa{clave}")
    with der:
        st.dataframe(pd.DataFrame([{"Marcador": f"{a}-{b}", "Prob.": pr * 100} for a, b, pr in p["top"]]),
                     hide_index=True, width="stretch", column_config={"Prob.": PCT})
        st.dataframe(pd.DataFrame([{"Línea": f"{linea} goles", "Más de": p["over"][linea] * 100,
                                    "Menos de": (1 - p["over"][linea]) * 100} for linea in LINEAS]),
                     hide_index=True, width="stretch", column_config={"Más de": PCT, "Menos de": PCT})

    seccion("Lesionados")
    b1, b2 = st.columns(2)
    for col, equipo, imp in ((b1, local, p["bajas"]["L"]), (b2, visita, p["bajas"]["V"])):
        with col:
            if imp is None:
                st.caption(f"{equipo}: sin datos de Transfermarkt")
            elif len(imp["jugadores"]):
                st.caption(f"{equipo}: falta el **{imp['pct']:.0%}** del valor del plantel ({imp['delta_elo']:+.0f} Elo)")
                st.dataframe(tabla_lesionados(imp), hide_index=True, width="stretch",
                             column_config={"Valor (M€)": st.column_config.NumberColumn(format="%.1f")})
            else:
                st.caption(f"{equipo}: sin lesionados")

    seccion("Forma reciente")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"**{local}**")
        st.markdown(forma_html(partidos, local), unsafe_allow_html=True)
    with f2:
        st.markdown(f"**{visita}**")
        st.markdown(forma_html(partidos, visita), unsafe_allow_html=True)
    seccion("Evolución del Elo (últimos 4 años)")
    st.plotly_chart(grafico_elo(historial, local, visita), width="stretch", key=f"elo{clave}")


@st.dialog("Detalle del partido", width="large")
def ventana_detalle(r, p):
    cuotas = {c: r.get(c, np.nan) for c in CUOTAS}
    mostrar_detalle(p, r["local"], r["visita"], cuotas, clave="dlg")


# ---------------------------------------------------------------- barra lateral y carga

with st.sidebar:
    st.markdown("**Predictor Fútbol**")
    if not EN_NUBE and st.button("Actualizar datos", width="stretch",
                                 help="Baja resultados, próximos partidos y lesionados, y reajusta los modelos (~3 min)"):
        with st.spinner("Actualizando…"):
            r = subprocess.run([sys.executable, "actualizar.py", "--rapido"], cwd=config.RAIZ,
                               capture_output=True, text=True)
        if r.returncode == 0:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Listo")
        else:
            st.error("Falló la actualización")
            st.code(r.stderr[-2000:])

pred = cargar_predictor(version_datos())
if pred is None:
    st.warning("Todavía no hay datos. Corre `python actualizar.py` en la carpeta del proyecto.")
    st.stop()
partidos, historial, fixtures, reg = cargar_datos(version_datos())
info = pred.info

with st.sidebar:
    st.caption(f"{len(partidos):,} partidos desde {partidos['fecha'].min():%Y} · 16 ligas + Champions")
    st.caption("Modelos: Elo, Poisson (Dixon-Coles), machine learning con cuotas y lesionados, combinados con stacking.")
    if EN_NUBE:
        st.caption("Se actualiza solo todos los días a la 1:00 a.m. (Perú).")

actualizado = pd.Timestamp(info["actualizado"])
st.markdown(f'<div class="encabezado"><span class="titulo">Predictor Fútbol</span>'
            f'<span class="estado">Datos al {actualizado:%d/%m/%Y %H:%M} · Lesionados {pred.fecha_bajas or "sin datos"}</span></div>',
            unsafe_allow_html=True)

tab_prox, tab_vs, tab_rank, tab_hist, tab_eval = st.tabs(
    ["Partidos", "Enfrentamiento", "Ranking Elo", "Historial", "Evaluación"])


# ---------------------------------------------------------------- partidos

with tab_prox:
    if fixtures.empty:
        st.info("No hay próximos partidos cargados. Actualiza los datos.")
    else:
        disponibles = [c for c in list(config.LIGAS) + ["CL"] if c in set(fixtures["competicion"])]
        c1, c2 = st.columns([4, 1])
        elegidas = c1.multiselect("Ligas", disponibles, format_func=NOMBRE_LIGA.get,
                                  default=[c for c in disponibles if c in config.LIGAS_PRINCIPALES + ["CL"]])
        dias = c2.selectbox("Próximos", [3, 7, 14], index=1, format_func=lambda d: f"{d} días")
        lista = predecir_fixtures(pred, version_datos(), tuple(elegidas), dias)
        if not lista:
            st.info("No hay partidos para ese filtro.")
        else:
            st.markdown(leyenda_html(), unsafe_allow_html=True)
            columnas = st.columns(2)
            for i, (r, p) in enumerate(lista):
                with columnas[i % 2]:
                    st.markdown(tarjeta_html(r, p), unsafe_allow_html=True)
                    if st.button("Ver detalle", key=f"det{i}", width="stretch"):
                        ventana_detalle(r, p)


# ---------------------------------------------------------------- enfrentamiento

with tab_vs:
    tabla = pred.tabla
    activos = tabla[tabla["activo"]].copy()
    activos["etiqueta"] = activos["equipo"] + "  ·  " + activos["liga"].map(NOMBRE_LIGA).fillna("Otros")
    opciones = activos.sort_values("equipo")["equipo"].tolist()
    etiqueta = dict(zip(activos["equipo"], activos["etiqueta"]))
    c1, c2, c3 = st.columns([5, 5, 2])
    local = c1.selectbox("Local", opciones, index=opciones.index("Real Madrid") if "Real Madrid" in opciones else 0,
                         format_func=etiqueta.get)
    visita = c2.selectbox("Visita", opciones, index=opciones.index("Man City") if "Man City" in opciones else 1,
                          format_func=etiqueta.get)
    neutral = c3.checkbox("Cancha neutral", help="Por ejemplo, la final de Champions")
    con_bajas = c3.checkbox("Con lesionados", value=True)
    if local == visita:
        st.warning("Elige dos equipos distintos.")
    else:
        p = pred.partido(local, visita, neutral=neutral, con_bajas=con_bajas)
        mostrar_detalle(p, local, visita, clave="vs")
        st.caption("Para un cruce inventado no hay cuotas, así que el modelo usa solo Elo, goles, forma y lesionados.")


# ---------------------------------------------------------------- ranking

with tab_rank:
    c1, c2 = st.columns([3, 1])
    ligas_rank = c1.multiselect("Filtrar por liga", list(config.LIGAS) + ["CL"], format_func=NOMBRE_LIGA.get,
                                placeholder="Todas")
    solo_activos = c2.checkbox("Solo equipos activos", value=True)
    r = pred.tabla.copy()
    if solo_activos:
        r = r[r["activo"]]
    if ligas_rank:
        r = r[r["liga"].isin(ligas_rank)]
    r = r.reset_index(drop=True)
    r.index += 1
    st.dataframe(pd.DataFrame({"Equipo": r["equipo"], "Liga": r["liga"].map(NOMBRE_LIGA).fillna("Otros (solo Champions)"),
                               "Elo": r["elo"].round(0).astype(int)}), width="stretch", height=600)


# ---------------------------------------------------------------- historial

with tab_hist:
    st.markdown("Cada actualización guarda la predicción de los partidos de la semana **antes** de que se jueguen. "
                "Cuando llega el resultado, se compara. Así sabemos cómo le va al modelo en vivo, y si el "
                "ajuste por lesionados ayuda de verdad.")
    if reg.empty:
        st.info("Todavía no hay predicciones registradas.")
    else:
        res = registro.resumen(reg)
        if res is None:
            st.info(f"{len(reg)} predicciones registradas, ninguna resuelta aún.")
        else:
            st.markdown(f"**{res['partidos']} partidos resueltos** ({res['desde']} – {res['hasta']}) · "
                        f"{res['pendientes']} pendientes")
            seccion("Quién gana (1X2)")
            tabla_metricas(res["1x2"])
            k1, k2, k3 = st.columns(3)
            k1.metric("Más de 2.5 · log-loss", f"{res['over25']['Final']['log_loss']:.4f}",
                      help=f"Casas: {res['over25']['Casas']['log_loss']:.4f}" if "Casas" in res["over25"] else None)
            k2.metric("Ambos marcan · acierto", f"{res['btts']['Final']['acierto']:.0%}")
            k3.metric("Marcador exacto", f"{res['marcador_acierto']:.0%}")
            pm = pd.DataFrame(res["por_mes"])
            fig = go.Figure(go.Bar(x=pm["mes"], y=pm["acierto"] * 100, text=pm["partidos"].astype(str) + " partidos",
                                   marker_color=COLOR["L"]))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title="% acierto 1X2"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        seccion("Últimas predicciones")
        ult = reg.sort_values("fecha", ascending=False).head(60).copy()
        ult["Pronóstico"] = [favorito({"L": a, "E": b, "V": c}, l, v)[0]
                             for a, b, c, l, v in zip(ult["pL"], ult["pE"], ult["pV"], ult["local"], ult["visita"])]
        ult["Resultado"] = [("" if pd.isna(g) else f"{int(g)}-{int(h)}") for g, h in zip(ult["gl"], ult["gv"])]
        ult["Acierto"] = [("pendiente" if pd.isna(r_) else ("sí" if "LEV"[int(np.argmax([a, b, c]))] == r_ else "no"))
                          for a, b, c, r_ in zip(ult["pL"], ult["pE"], ult["pV"], ult["resultado"])]
        st.dataframe(ult[["fecha", "local", "visita", "Pronóstico", "marcador", "Resultado", "Acierto"]]
                     .rename(columns={"fecha": "Fecha", "local": "Local", "visita": "Visita", "marcador": "Marcador prob."}),
                     hide_index=True, width="stretch", column_config={"Fecha": st.column_config.DateColumn(format="DD/MM")})


# ---------------------------------------------------------------- evaluación

with tab_eval:
    ev = info["evaluacion"]
    temps = ev["temporadas"]
    nombre_t = lambda t: f"20{t[:2]}/{t[2:]}"  # noqa: E731
    st.markdown(f"**Examen rotativo:** cada una de las temporadas {', '.join(map(nombre_t, temps))} se predijo "
                f"con modelos que solo vieron las anteriores, mes a mes como si fuera en vivo. "
                f"En total {ev['partidos']:,} partidos de ligas principales + Champions. "
                "El **stacking** (cómo se combinan Elo, Poisson y ML) se entrena siempre con la temporada anterior.")
    st.caption("El ajuste por lesionados no entra en este examen (no hay historial de bajas); se mide en la pestaña Historial.")
    vista = st.radio("Ver", ["Promedio"] + [nombre_t(t) for t in temps], horizontal=True)
    e = ev if vista == "Promedio" else ev["por_temporada"][temps[[nombre_t(t) for t in temps].index(vista)]]

    seccion("Quién gana (1X2)")
    tabla_metricas(e["1x2"])
    st.caption("Solo los partidos de liga que tienen cuotas:")
    tabla_metricas(e["1x2_con_cuotas"])
    seccion("Goles")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Más de 2.5 goles")
        tabla_metricas(e["over25"], con_brier=False)
        st.caption("Solo partidos con cuotas:")
        tabla_metricas(e["over25_con_cuotas"], con_brier=False)
    with g2:
        st.caption("Ambos marcan")
        tabla_metricas(e["btts"], con_brier=False)
        st.metric("Acierto del marcador exacto", f"{e['marcador']['acierto']:.1%}",
                  help="Adivinar el marcador exacto es muy difícil: 10–13% ya es un buen número.")
    st.markdown("""
- **Log-loss**: mide qué tan buenas son las *probabilidades*. Mientras más bajo, mejor.
- **Acierto**: % de veces que la opción más probable fue la que pasó. En 1X2, 50–55% ya es muy bueno.
- **Base**: un "modelo tonto" que siempre dice lo mismo. Hay que ganarle sí o sí.
- **Casas de apuestas**: la vara más alta. Como el ML usa sus cuotas como una variable más, lo normal es quedar muy cerca.
""")
    seccion("Calibración")
    st.caption("Si el modelo dice 60%, ¿pasa ~60% de las veces? Mientras más cerca de la diagonal, mejor.")
    k1, k2 = st.columns(2)
    k1.plotly_chart(grafico_calibracion(e["calibracion"], "Prob. que dio el modelo (gana el local)"), width="stretch")
    k2.plotly_chart(grafico_calibracion(e["calibracion_over25"], "Prob. que dio el modelo (más de 2.5)"), width="stretch")
    if "importancia" in ev:
        seccion("Qué mira el modelo de ML")
        imp = pd.DataFrame(ev["importancia"][:15], columns=["Variable", "Importancia"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["Importancia"], y=imp["Variable"], orientation="h", marker_color=COLOR["L"]))
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Cuánto empeora el log-loss si se desordena"))
        st.plotly_chart(fig, width="stretch")
