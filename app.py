"""Predictor de fútbol europeo — interfaz en Streamlit.

Uso: streamlit run app.py
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, registro
from src.poisson import LINEAS
from src.prediccion import Predictor

st.set_page_config(page_title="Predictor Fútbol", page_icon="⚽", layout="wide")

NOMBRE_LIGA = {c: n for c, (n, _, _) in config.LIGAS.items()} | {"CL": "Champions League"}
COLOR = {"L": "#2e7d32", "E": "#9e9e9e", "V": "#c62828"}
PCT = st.column_config.NumberColumn(format="%.0f%%")
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
ETIQUETA_MODELO = {"Elo": "Elo (fase 1)", "Poisson": "Poisson (fase 2)", "ML": "Machine learning (fase 3)",
                   "Final": "⭐ Final (stacking, lo que usa la app)", "Base": "Base (siempre la frecuencia)",
                   "Casas": "Casas de apuestas", "Sin lesionados": "Final sin ajuste por lesionados",
                   "Final (con cuotas)": "⭐ Final (mismos partidos)"}


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
        cuotas = {c: getattr(r, c, np.nan) for c in ("cuota_l", "cuota_e", "cuota_v", "cuota_o25", "cuota_u25")}
        salida.append((r._asdict(), _pred.partido(r.local, r.visita, r.competicion, fecha=r.fecha, cuotas=cuotas)))
    return salida


# ---------------------------------------------------------------- piezas visuales

def favorito(p, local, visita):
    """'Real Madrid 58%' o 'Parejo' si nadie pasa de 45% con ventaja clara."""
    pl, pe, pv = p["L"], p["E"], p["V"]
    if max(pl, pv) < 0.45 and abs(pl - pv) < 0.08:
        return f"⚖️ Parejo ({pl:.0%} / {pe:.0%} / {pv:.0%})"
    return f"🏠 {local} {pl:.0%}" if pl > pv else f"✈️ {visita} {pv:.0%}"


def forma(partidos, equipo, n=5):
    p = partidos[(partidos["local"] == equipo) | (partidos["visita"] == equipo)].tail(n).iloc[::-1]
    filas = []
    for r in p.itertuples():
        es_local = r.local == equipo
        gf, gc = (r.gl, r.gv) if es_local else (r.gv, r.gl)
        icono = "✅" if gf > gc else "➖" if gf == gc else "❌"
        rival = r.visita if es_local else r.local
        filas.append(f"{icono} {gf}-{gc} {'vs' if es_local else 'en'} {rival} · "
                     f"{NOMBRE_LIGA.get(r.competicion, r.competicion)} · {r.fecha:%d/%m/%y}")
    return filas


def barra_probs(p, local, visita):
    fig = go.Figure()
    for clave, etiqueta in (("L", local), ("E", "Empate"), ("V", visita)):
        fig.add_bar(y=[""], x=[p[clave] * 100], orientation="h", name=etiqueta,
                    marker_color=COLOR[clave], text=f"{etiqueta}<br><b>{p[clave]:.0%}</b>",
                    textposition="inside", insidetextanchor="middle", hoverinfo="skip")
    fig.update_layout(barmode="stack", height=90, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      xaxis=dict(visible=False, range=[0, 100]), yaxis=dict(visible=False))
    return fig


def mapa_marcadores(matriz, local, visita):
    m = np.array(matriz) * 100
    goles = [str(i) for i in range(m.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=m, x=goles, y=goles, colorscale="Blues", showscale=False,
        text=[[f"{v:.1f}%" for v in fila] for fila in m], texttemplate="%{text}",
        hovertemplate=f"{local} %{{y}} - %{{x}} {visita}: %{{z:.1f}}%<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(title=f"Goles de {visita}", side="top"),
                      yaxis=dict(title=f"Goles de {local}", autorange="reversed"))
    return fig


def grafico_elo(historial, local, visita):
    desde = historial["fecha"].max() - pd.DateOffset(years=4)
    fig = go.Figure()
    for equipo, color in ((local, "#1565c0"), (visita, "#ef6c00")):
        h = historial[(historial["equipo"] == equipo) & (historial["fecha"] >= desde)]
        fig.add_scatter(x=h["fecha"], y=h["elo"], name=equipo, mode="lines", line=dict(color=color, width=2))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    return fig


def grafico_calibracion(cal, eje_x):
    cal = pd.DataFrame(cal)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfecto")
    fig.add_scatter(x=cal["pred"], y=cal["real"], mode="markers+lines", name="Modelo",
                    marker=dict(size=[max(6, min(22, n / 25)) for n in cal["n"]]),
                    text=[f"{n} partidos" for n in cal["n"]])
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1),
                      xaxis=dict(title=eje_x, tickformat=".0%"),
                      yaxis=dict(title="Lo que pasó de verdad", tickformat=".0%"))
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
    """Todo el detalle de un partido (se usa en la lista y en Enfrentamiento)."""
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Gana {local}", f"{p['1x2']['L']:.0%}")
    m2.metric("Empate", f"{p['1x2']['E']:.0%}")
    m3.metric(f"Gana {visita}", f"{p['1x2']['V']:.0%}")
    st.plotly_chart(barra_probs(p["1x2"], local, visita), width="stretch", config={"displayModeBar": False},
                    key=f"barra{clave}")
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

    st.markdown("**⚽ Goles**")
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

    st.markdown("**🚑 Lesionados**")
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
                st.caption(f"{equipo}: sin lesionados 🎉")

    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"**Últimos 5 · {local}**")
        st.markdown("\n".join(f"- {x}" for x in forma(partidos, local)))
    with f2:
        st.markdown(f"**Últimos 5 · {visita}**")
        st.markdown("\n".join(f"- {x}" for x in forma(partidos, visita)))
    st.plotly_chart(grafico_elo(historial, local, visita), width="stretch", key=f"elo{clave}")


# ---------------------------------------------------------------- barra lateral y carga

with st.sidebar:
    st.title("⚽ Predictor")
    if st.button("🔄 Actualizar datos", width="stretch",
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
    st.caption(f"Actualizado: {info['actualizado'].replace('T', ' ')}")
    st.caption(f"Lesionados: {pred.fecha_bajas or 'sin datos'}")
    st.caption(f"{len(partidos):,} partidos desde {partidos['fecha'].min():%Y}")
    st.caption("Elo + Poisson + ML (con cuotas) + lesionados, combinados con stacking")

tab_prox, tab_vs, tab_rank, tab_hist, tab_eval = st.tabs(
    ["📅 Partidos", "⚔️ Enfrentamiento", "🏆 Ranking Elo", "📈 Historial", "📊 ¿Qué tan bueno es?"])


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
            st.caption("Haz clic en un partido para ver el detalle debajo. **+2.5** = probabilidad de 3 goles o más.")
            filas = []
            for r, p in lista:
                gl, gv, _ = p["top"][0]
                filas.append({"Fecha": f"{DIAS_SEMANA[r['fecha'].weekday()][:3]} {r['fecha']:%d/%m}",
                              "Hora": "" if pd.isna(r["hora"]) else r["hora"],
                              "Liga": NOMBRE_LIGA[r["competicion"]],
                              "Partido": f"{r['local']} vs {r['visita']}",
                              "Pronóstico": favorito(p["1x2"], r["local"], r["visita"]),
                              "+2.5": p["over"]["2.5"] * 100, "Marcador": f"{gl}-{gv}"})
            evento = st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch",
                                  height=min(600, 38 + 35 * len(filas)), on_select="rerun",
                                  selection_mode="single-row", column_config={"+2.5": PCT})
            sel = evento.selection.rows
            if sel:
                r, p = lista[sel[0]]
                st.subheader(f"{r['local']} vs {r['visita']} · {NOMBRE_LIGA[r['competicion']]} · {r['fecha']:%d/%m}")
                cuotas = {c: r.get(c, np.nan) for c in ("cuota_l", "cuota_e", "cuota_v")}
                mostrar_detalle(p, r["local"], r["visita"], cuotas, clave="fx")
            else:
                st.info("👆 Selecciona un partido de la lista para ver el detalle.")


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
        st.caption("Sin cuotas: para un cruce inventado el modelo no las tiene, así que usa solo Elo, goles, forma y lesionados.")
        mostrar_detalle(p, local, visita, clave="vs")


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
            st.caption("¿Quién gana? (1X2)")
            tabla_metricas(res["1x2"])
            k1, k2, k3 = st.columns(3)
            k1.metric("Más de 2.5 · log-loss", f"{res['over25']['Final']['log_loss']:.4f}",
                      help=f"Casas: {res['over25']['Casas']['log_loss']:.4f}" if "Casas" in res["over25"] else None)
            k2.metric("Ambos marcan · acierto", f"{res['btts']['Final']['acierto']:.0%}")
            k3.metric("Marcador exacto", f"{res['marcador_acierto']:.0%}")
            pm = pd.DataFrame(res["por_mes"])
            fig = go.Figure(go.Bar(x=pm["mes"], y=pm["acierto"] * 100, text=pm["partidos"].astype(str) + " partidos",
                                   marker_color="#1565c0"))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title="% acierto 1X2"))
            st.plotly_chart(fig, width="stretch")
        st.caption("Últimas predicciones")
        ult = reg.sort_values("fecha", ascending=False).head(60).copy()
        ult["Pronóstico"] = [favorito({"L": a, "E": b, "V": c}, l, v)
                             for a, b, c, l, v in zip(ult["pL"], ult["pE"], ult["pV"], ult["local"], ult["visita"])]
        ult["Resultado"] = [("" if pd.isna(g) else f"{int(g)}-{int(h)}") for g, h in zip(ult["gl"], ult["gv"])]
        ult["✓"] = [("" if pd.isna(r_) else ("✅" if "LEV"[int(np.argmax([a, b, c]))] == r_ else "❌"))
                    for a, b, c, r_ in zip(ult["pL"], ult["pE"], ult["pV"], ult["resultado"])]
        st.dataframe(ult[["fecha", "local", "visita", "Pronóstico", "marcador", "Resultado", "✓"]]
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

    st.subheader("¿Quién gana? (1X2)")
    tabla_metricas(e["1x2"])
    st.caption("Solo los partidos de liga que tienen cuotas:")
    tabla_metricas(e["1x2_con_cuotas"])
    st.subheader("Goles")
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
    st.subheader("Calibración")
    st.caption("Si el modelo dice 60%, ¿pasa ~60% de las veces? Mientras más cerca de la diagonal, mejor.")
    k1, k2 = st.columns(2)
    k1.plotly_chart(grafico_calibracion(e["calibracion"], "Prob. que dio el modelo (gana el local)"), width="stretch")
    k2.plotly_chart(grafico_calibracion(e["calibracion_over25"], "Prob. que dio el modelo (más de 2.5)"), width="stretch")
    if "importancia" in ev:
        st.subheader("¿Qué mira el modelo de ML?")
        imp = pd.DataFrame(ev["importancia"][:15], columns=["Variable", "Importancia"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["Importancia"], y=imp["Variable"], orientation="h", marker_color="#1565c0"))
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(title="Cuánto empeora el log-loss si se desordena"))
        st.plotly_chart(fig, width="stretch")
