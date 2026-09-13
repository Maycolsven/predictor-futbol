"""Variables (features) que mira el modelo de machine learning.

Todo se calcula con datos de ANTES del partido. La idea: después de cada partido
cada equipo tiene un "estado" (puntos de los últimos 5, tiros al arco, racha de
Elo…). Para predecir un partido se usa el estado que tenía cada equipo justo
antes de jugarlo.
"""
import numpy as np
import pandas as pd

VENTANAS = [("pts", 5), ("pts", 10), ("gf", 5), ("gc", 5), ("gf", 10), ("gc", 10),
            ("tap_f", 5), ("tap_c", 5), ("tiros_f", 10), ("tiros_c", 10)]
ESTADO = [f"{c}{v}" for c, v in VENTANAS] + ["elo_tend"]
POR_EQUIPO = ESTADO + ["descanso"]
CUOTAS = ["cuota_l", "cuota_e", "cuota_v", "cuota_o25", "cuota_u25"]
CASAS = ["casas_L", "casas_E", "casas_V", "casas_o25"]  # lo que implican las cuotas (NaN si no hay)
FEATURES = (["elo_dif", "elo_prom", "lam_l", "lam_v", "neutral", "nivel"]
            + [f"l_{c}" for c in POR_EQUIPO] + [f"v_{c}" for c in POR_EQUIPO]
            + ["dif_pts5", "dif_tap5"] + CASAS)
DESCANSO_TIPICO = 7.0  # para enfrentamientos sin fecha

_NOMBRES = {
    "elo_dif": "Diferencia de Elo", "elo_prom": "Nivel de los equipos (Elo promedio)",
    "lam_l": "Goles esperados del local (Poisson)", "lam_v": "Goles esperados de la visita (Poisson)",
    "neutral": "Cancha neutral", "nivel": "Competición (Champions / 1.ª / 2.ª)",
    "pts5": "puntos (últ. 5)", "pts10": "puntos (últ. 10)", "gf5": "goles a favor (últ. 5)",
    "gc5": "goles en contra (últ. 5)", "gf10": "goles a favor (últ. 10)", "gc10": "goles en contra (últ. 10)",
    "tap_f5": "tiros al arco a favor (últ. 5)", "tap_c5": "tiros al arco en contra (últ. 5)",
    "tiros_f10": "tiros a favor (últ. 10)", "tiros_c10": "tiros en contra (últ. 10)",
    "elo_tend": "racha de Elo (últ. 5)", "descanso": "días de descanso",
    "dif_pts5": "Diferencia de puntos (últ. 5)", "dif_tap5": "Diferencia de dominio en tiros al arco (últ. 5)",
    "casas_L": "Casas: prob. local", "casas_E": "Casas: prob. empate", "casas_V": "Casas: prob. visita",
    "casas_o25": "Casas: prob. más de 2.5",
}


def probs_casas(df):
    """Probabilidades implícitas en las cuotas (sin el margen). NaN donde no hay cuotas."""
    inv = 1 / df[["cuota_l", "cuota_e", "cuota_v"]].astype(float)
    p = inv.div(inv.sum(axis=1), axis=0)
    o, u = 1 / df["cuota_o25"].astype(float), 1 / df["cuota_u25"].astype(float)
    return pd.DataFrame({"casas_L": p["cuota_l"], "casas_E": p["cuota_e"], "casas_V": p["cuota_v"],
                         "casas_o25": o / (o + u)}, index=df.index)


def nombre(col):
    """Nombre legible de una feature, para mostrar en la app."""
    if col[:2] in ("l_", "v_"):
        return ("Local: " if col[0] == "l" else "Visita: ") + _NOMBRES[col[2:]]
    return _NOMBRES[col]


def estados(partidos):
    """Una fila por equipo y partido, con su estado DESPUÉS de ese partido."""
    n = len(partidos)
    p = partidos
    base = pd.DataFrame({
        "partido": np.r_[p.index, p.index],
        "equipo": np.r_[p["local"].to_numpy(), p["visita"].to_numpy()],
        "fecha": np.r_[p["fecha"].to_numpy(), p["fecha"].to_numpy()],
        "gf": np.r_[p["gl"].to_numpy(), p["gv"].to_numpy()].astype(float),
        "gc": np.r_[p["gv"].to_numpy(), p["gl"].to_numpy()].astype(float),
        "tiros_f": np.r_[p["tiros_l"].to_numpy(), p["tiros_v"].to_numpy()].astype(float),
        "tiros_c": np.r_[p["tiros_v"].to_numpy(), p["tiros_l"].to_numpy()].astype(float),
        "tap_f": np.r_[p["tap_l"].to_numpy(), p["tap_v"].to_numpy()].astype(float),
        "tap_c": np.r_[p["tap_v"].to_numpy(), p["tap_l"].to_numpy()].astype(float),
        "elo_post": np.r_[p["elo_l_post"].to_numpy(), p["elo_v_post"].to_numpy()],
        "es_local": np.r_[np.ones(n, bool), np.zeros(n, bool)],
    })
    base["pts"] = np.where(base["gf"] > base["gc"], 3.0, np.where(base["gf"] == base["gc"], 1.0, 0.0))
    base = base.sort_values(["equipo", "fecha", "partido"], kind="stable")
    g = base.groupby("equipo", sort=False)
    for col, v in VENTANAS:
        base[f"{col}{v}"] = g[col].rolling(v, min_periods=1).mean().reset_index(level=0, drop=True)
    base["elo_tend"] = base["elo_post"] - g["elo_post"].shift(5)
    return base


def _completar(f):
    f = f.copy()
    f["elo_dif"] = f["elo_l"] - f["elo_v"]
    f["elo_prom"] = (f["elo_l"] + f["elo_v"]) / 2
    f["neutral"] = f["neutral"].astype(float)
    f["dif_pts5"] = f["l_pts5"] - f["v_pts5"]
    f["dif_tap5"] = (f["l_tap_f5"] - f["l_tap_c5"]) - (f["v_tap_f5"] - f["v_tap_c5"])
    f = f.join(probs_casas(f))
    return f[FEATURES].astype(float)


def historicas(partidos, lam):
    """Features de cada partido jugado, con el estado previo de cada equipo."""
    est = estados(partidos)
    previo = est.groupby("equipo", sort=False)[ESTADO + ["fecha"]].shift(1)
    previo["descanso"] = (est["fecha"] - previo["fecha"]).dt.days.clip(upper=30)
    previo["partido"], previo["es_local"] = est["partido"], est["es_local"]
    loc = previo[previo["es_local"]].set_index("partido")[POR_EQUIPO].add_prefix("l_")
    vis = previo[~previo["es_local"]].set_index("partido")[POR_EQUIPO].add_prefix("v_")
    f = partidos[["elo_l", "elo_v", "neutral", "nivel"] + CUOTAS].join(loc).join(vis).join(lam[["lam_l", "lam_v"]])
    return _completar(f)


def estado_actual(partidos):
    """Estado de cada equipo después de su último partido (para predecir los que vienen)."""
    est = estados(partidos)
    return est.groupby("equipo", sort=False).tail(1).set_index("equipo")[ESTADO + ["fecha"]]


def vector(estado, local, visita, elo_l, elo_v, lam_l, lam_v, neutral, nivel, fecha=None, cuotas=None):
    """Features de un partido futuro o hipotético. `cuotas` = dict con cuota_l… (opcional)."""
    fila = {"elo_l": elo_l, "elo_v": elo_v, "neutral": float(neutral), "nivel": nivel,
            "lam_l": lam_l, "lam_v": lam_v}
    for c in CUOTAS:
        fila[c] = float((cuotas or {}).get(c) or np.nan)
    for pref, equipo in (("l_", local), ("v_", visita)):
        e = estado.loc[equipo] if equipo in estado.index else None
        for c in ESTADO:
            fila[pref + c] = e[c] if e is not None else np.nan
        if e is None:
            fila[pref + "descanso"] = np.nan
        elif fecha is None:
            fila[pref + "descanso"] = DESCANSO_TIPICO
        else:
            fila[pref + "descanso"] = float(np.clip((pd.Timestamp(fecha) - e["fecha"]).days, 0, 30))
    return _completar(pd.DataFrame([fila]))
