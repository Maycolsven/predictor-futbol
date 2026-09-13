"""Fase 5: registro de predicciones y comparación con lo que pasó.

Cada vez que se actualiza, se guarda la predicción de los partidos de los
próximos días. Solo se guarda la PRIMERA vez que se ve un partido (así queda
una predicción hecha antes del partido, sin trampa). Cuando llega el resultado,
se cruza y queda en el historial.
"""
import numpy as np
import pandas as pd

from . import config, evaluacion

RUTA = config.DATA / "predicciones.csv"
CLAVE = ["fecha", "local", "visita"]


def registrar(pred, fixtures, ahora=None):
    """Agrega al registro los partidos nuevos de los próximos DIAS_REGISTRO días."""
    ahora = pd.Timestamp(ahora) if ahora is not None else pd.Timestamp.now()
    hoy = ahora.normalize()
    previo = pd.read_csv(RUTA, parse_dates=["fecha"]) if RUTA.exists() else pd.DataFrame(columns=CLAVE)
    vistos = set(zip(previo["fecha"], previo["local"], previo["visita"]))
    fx = fixtures[(fixtures["fecha"] >= hoy) & (fixtures["fecha"] < hoy + pd.Timedelta(days=config.DIAS_REGISTRO))]

    filas = []
    for r in fx.itertuples():
        if (r.fecha, r.local, r.visita) in vistos or not (pred.conoce(r.local) and pred.conoce(r.visita)):
            continue
        cuotas = {c: getattr(r, c, np.nan) for c in ("cuota_l", "cuota_e", "cuota_v", "cuota_o25", "cuota_u25")}
        con = pred.partido(r.local, r.visita, r.competicion, fecha=r.fecha, cuotas=cuotas)
        sin = pred.partido(r.local, r.visita, r.competicion, fecha=r.fecha, cuotas=cuotas, con_bajas=False)
        gl, gv, _ = con["top"][0]
        filas.append({
            "fecha": r.fecha, "local": r.local, "visita": r.visita, "competicion": r.competicion,
            "registrado": ahora.strftime("%Y-%m-%d %H:%M"),
            "pL": con["1x2"]["L"], "pE": con["1x2"]["E"], "pV": con["1x2"]["V"],
            "over25": con["over"]["2.5"], "btts": con["btts"], "marcador": f"{gl}-{gv}",
            "sin_pL": sin["1x2"]["L"], "sin_pE": sin["1x2"]["E"], "sin_pV": sin["1x2"]["V"],
            "bajas_l": (con["bajas"]["L"] or {}).get("pct", np.nan),
            "bajas_v": (con["bajas"]["V"] or {}).get("pct", np.nan),
            **cuotas,
        })
    if not filas:
        nuevo = previo
    elif previo.empty:
        nuevo = pd.DataFrame(filas)
    else:
        nuevo = pd.concat([previo, pd.DataFrame(filas)], ignore_index=True)
    nuevo.to_csv(RUTA, index=False)
    return len(filas)


def resolver(partidos):
    """Cruza el registro con los resultados ya jugados."""
    if not RUTA.exists():
        return pd.DataFrame()
    reg = pd.read_csv(RUTA, parse_dates=["fecha"])
    reg = reg.drop(columns=[c for c in ("gl", "gv", "resultado") if c in reg.columns])
    res = partidos[CLAVE + ["gl", "gv", "resultado"]].drop_duplicates(CLAVE)
    reg = reg.merge(res, on=CLAVE, how="left")
    reg.to_csv(RUTA, index=False)
    return reg


def resumen(reg):
    """Métricas del registro sobre los partidos ya jugados: final, sin bajas y casas."""
    r = reg.dropna(subset=["resultado"]).copy()
    if r.empty:
        return None
    y = r["resultado"].to_numpy()
    out = {"partidos": int(len(r)), "pendientes": int(reg["resultado"].isna().sum()),
           "desde": r["fecha"].min().strftime("%d/%m/%Y"), "hasta": r["fecha"].max().strftime("%d/%m/%Y"),
           "1x2": {"Final": evaluacion.metricas(r[["pL", "pE", "pV"]].to_numpy(), y),
                   "Sin lesionados": evaluacion.metricas(r[["sin_pL", "sin_pE", "sin_pV"]].to_numpy(), y)}}
    mask, p_c = evaluacion.probs_casas(r, ["cuota_l", "cuota_e", "cuota_v"])
    if mask.any():
        out["1x2"]["Casas"] = evaluacion.metricas(p_c, y[mask])
        out["1x2"]["Final (con cuotas)"] = evaluacion.metricas(r.loc[mask, ["pL", "pE", "pV"]].to_numpy(), y[mask])
    y_over = (r["gl"] + r["gv"] > 2.5).to_numpy()
    out["over25"] = {"Final": evaluacion.metricas_si_no(r["over25"], y_over)}
    mask, p_c = evaluacion.probs_casas(r, ["cuota_o25", "cuota_u25"])
    if mask.any():
        out["over25"]["Casas"] = evaluacion.metricas_si_no(p_c[:, 0], y_over[mask])
    out["btts"] = {"Final": evaluacion.metricas_si_no(r["btts"], ((r["gl"] > 0) & (r["gv"] > 0)).to_numpy())}
    out["marcador_acierto"] = float((r["marcador"] == r["gl"].astype(int).astype(str) + "-" + r["gv"].astype(int).astype(str)).mean())
    r["mes"] = r["fecha"].dt.to_period("M").astype(str)
    r["acierto"] = (r[["pL", "pE", "pV"]].to_numpy().argmax(axis=1) == pd.Series(y).map({"L": 0, "E": 1, "V": 2}).to_numpy())
    out["por_mes"] = r.groupby("mes").agg(partidos=("acierto", "size"), acierto=("acierto", "mean")).reset_index().to_dict("records")
    return out
