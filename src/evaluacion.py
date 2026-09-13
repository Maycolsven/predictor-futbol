"""Métricas, 'stacking' (cómo se combinan los modelos) y examen final."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import ml

MODELOS_1X2 = ("Elo", "Poisson", "ML")
MODELOS_SI_NO = ("Poisson", "ML")


# ---------------------------------------------------------------- métricas

def metricas(probs, resultados):
    """Para 1X2: log-loss (menor = mejor), Brier (menor = mejor) y % de acierto."""
    y = pd.Series(resultados).map({"L": 0, "E": 1, "V": 2}).to_numpy()
    probs = np.asarray(probs)
    return {
        "log_loss": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(((probs - np.eye(3)[y]) ** 2).sum(axis=1).mean()),
        "acierto": float((probs.argmax(axis=1) == y).mean()),
        "partidos": int(len(y)),
    }


def metricas_si_no(p, y):
    """Para mercados de sí/no (más de 2.5, ambos marcan)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return {
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "acierto": float(((p > 0.5) == (y == 1)).mean()),
        "partidos": int(len(y)),
    }


def probs_casas(df, columnas):
    """Probabilidades implícitas en las cuotas, quitando el margen de la casa."""
    mask = df[columnas].notna().all(axis=1).to_numpy()
    inv = 1 / df.loc[mask, columnas].to_numpy(dtype=float)
    return mask, inv / inv.sum(axis=1, keepdims=True)


def frecuencias(entreno):
    goles = entreno["gl"] + entreno["gv"]
    res = entreno["resultado"]
    return {"L": float((res == "L").mean()), "E": float((res == "E").mean()), "V": float((res == "V").mean()),
            "over25": float((goles > 2.5).mean()), "btts": float(((entreno["gl"] > 0) & (entreno["gv"] > 0)).mean())}


def calibracion(p, real):
    tramos = pd.cut(p, bins=np.linspace(0, 1, 11))
    return (pd.DataFrame({"tramo": tramos, "pred": p, "real": real})
            .groupby("tramo", observed=True).agg(pred=("pred", "mean"), real=("real", "mean"), n=("real", "size"))
            .reset_index(drop=True).to_dict(orient="records"))


# ---------------------------------------------------------------- stacking

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _matriz(preds, modelos):
    """Una columna por (modelo, opción) con el logit de su probabilidad."""
    cols = []
    for m in modelos:
        p = np.asarray(preds[m], dtype=float)
        cols.append(_logit(p) if p.ndim == 2 else _logit(p)[:, None])
    return np.hstack(cols)


def ajustar_stack(preds, y, mercado):
    """Aprende cuánto confiar en cada modelo (y de paso calibra las probabilidades).

    Analogía: tres expertos opinan; una regresión logística aprende, viendo una
    temporada completa, a quién hacerle más caso y cuánto exagera cada uno.
    """
    modelos = MODELOS_1X2 if mercado == "1x2" else MODELOS_SI_NO
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(_matriz(preds, modelos), y)
    return {"lr": lr, "modelos": modelos}


def predecir_stack(stack, preds):
    p = stack["lr"].predict_proba(_matriz(preds, stack["modelos"]))
    return p if p.shape[1] == 3 else p[:, 1]


# ---------------------------------------------------------------- examen

def evaluar_temporada(prueba, preds, matrices, base):
    """Examen de una temporada. `preds[mercado][modelo]` = probabilidades fuera de muestra."""
    res = prueba["resultado"].to_numpy()
    gl, gv = prueba["gl"].to_numpy(), prueba["gv"].to_numpy()
    y_over, y_btts = (gl + gv) > 2.5, (gl > 0) & (gv > 0)
    n = len(prueba)
    ev = {"partidos": n}

    ev["1x2"] = {m: metricas(p, res) for m, p in preds["1x2"].items()}
    ev["1x2"]["Base"] = metricas(np.tile([base["L"], base["E"], base["V"]], (n, 1)), res)
    mask, p_c = probs_casas(prueba, ["cuota_l", "cuota_e", "cuota_v"])
    ev["1x2_con_cuotas"] = {m: metricas(p[mask], res[mask]) for m, p in preds["1x2"].items() if m in ("ML", "Final")}
    ev["1x2_con_cuotas"]["Casas"] = metricas(p_c, res[mask])

    ev["over25"] = {m: metricas_si_no(p, y_over) for m, p in preds["over25"].items()}
    ev["over25"]["Base"] = metricas_si_no(np.full(n, base["over25"]), y_over)
    mask, p_c = probs_casas(prueba, ["cuota_o25", "cuota_u25"])
    ev["over25_con_cuotas"] = {m: metricas_si_no(p[mask], y_over[mask]) for m, p in preds["over25"].items()
                               if m in ("ML", "Final")}
    ev["over25_con_cuotas"]["Casas"] = metricas_si_no(p_c[:, 0], y_over[mask])

    ev["btts"] = {m: metricas_si_no(p, y_btts) for m, p in preds["btts"].items()}
    ev["btts"]["Base"] = metricas_si_no(np.full(n, base["btts"]), y_btts)

    tope = matrices.shape[1] - 1
    p_real = matrices[np.arange(n), np.minimum(gl, tope), np.minimum(gv, tope)]
    top = matrices.reshape(n, -1).argmax(axis=1)
    ev["marcador"] = {"log_loss": float(-np.log(np.clip(p_real, 1e-12, 1)).mean()),
                      "acierto": float(((top // (tope + 1) == gl) & (top % (tope + 1) == gv)).mean()), "partidos": n}
    ev["calibracion"] = calibracion(preds["1x2"]["Final"][:, 0], res == "L")
    ev["calibracion_over25"] = calibracion(preds["over25"]["Final"], y_over)
    return ev


def promediar(evaluaciones):
    """Promedio (ponderado por partidos) de varias temporadas, con la misma forma que una sola."""
    temps = list(evaluaciones)
    prom = {"partidos": sum(evaluaciones[t]["partidos"] for t in temps), "temporadas": temps}
    for clave in ("1x2", "1x2_con_cuotas", "over25", "over25_con_cuotas", "btts"):
        prom[clave] = {}
        for modelo in evaluaciones[temps[-1]][clave]:
            filas = [evaluaciones[t][clave][modelo] for t in temps if modelo in evaluaciones[t][clave]]
            n = sum(f["partidos"] for f in filas)
            prom[clave][modelo] = {k: sum(f[k] * f["partidos"] for f in filas) / n for k in ("log_loss", "brier", "acierto")}
            prom[clave][modelo]["partidos"] = n
    filas = [evaluaciones[t]["marcador"] for t in temps]
    n = sum(f["partidos"] for f in filas)
    prom["marcador"] = {k: sum(f[k] * f["partidos"] for f in filas) / n for k in ("log_loss", "acierto")}
    prom["marcador"]["partidos"] = n
    prom["calibracion"] = evaluaciones[temps[-1]]["calibracion"]
    prom["calibracion_over25"] = evaluaciones[temps[-1]]["calibracion_over25"]
    return prom


def objetivo(df, mercado):
    return ml.objetivo(df, mercado)
