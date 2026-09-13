"""De Elo a probabilidades 1X2 con un modelo 'logit ordenado'.

Analogía: la diferencia de Elo es como la diferencia de estatura entre dos
boxeadores. El modelo aprende de miles de partidos cuánta ventaja da cada
punto de diferencia, y dónde están los "cortes" entre perder, empatar y ganar.
Así el empate sale más probable cuando los equipos están parejos.
"""
import numpy as np
from scipy.optimize import minimize

from . import config, elo
from .evaluacion import metricas


def _sigmoide(x):
    return 1 / (1 + np.exp(-x))


def _probs(params, dif, neutral):
    b, h, c1, d = params
    z = b * dif / 100 + h * (1 - neutral)
    c2 = c1 + np.exp(d)
    p_v = _sigmoide(c1 - z)
    p_ve = _sigmoide(c2 - z)
    return np.column_stack([1 - p_ve, p_ve - p_v, p_v])  # local, empate, visita


def ajustar(df):
    dif = (df["elo_l"] - df["elo_v"]).to_numpy()
    neutral = df["neutral"].astype(float).to_numpy()
    y = df["resultado"].map({"L": 0, "E": 1, "V": 2}).to_numpy()

    def nll(params):
        p = _probs(params, dif, neutral)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()

    res = minimize(nll, x0=[0.5, 0.3, -1.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
    return [float(v) for v in res.x]


def predecir(params, elo_local, elo_visita, neutral=False):
    p = _probs(params, np.array([elo_local - elo_visita]), np.array([float(neutral)]))[0]
    return {"L": float(p[0]), "E": float(p[1]), "V": float(p[2])}


def separar(partidos, temporada):
    """Entrenamiento = temporadas anteriores; prueba = ligas principales + Champions de `temporada`."""
    validas = config.temporadas()[config.TEMPORADAS_CALENTAMIENTO:]
    entreno = partidos[partidos["temporada"].isin(validas) & (partidos["temporada"] < temporada)]
    prueba = partidos[(partidos["temporada"] == temporada)
                      & partidos["competicion"].isin(config.LIGAS_PRINCIPALES + ["CL"])]
    return entreno, prueba


def probs_fuera_de_muestra(con_elo, temporada):
    """Probabilidades Elo para `temporada` con un modelo que no la vio."""
    entreno, prueba = separar(con_elo, temporada)
    params = ajustar(entreno)
    prueba = prueba.copy()
    prueba[["elo_pL", "elo_pE", "elo_pV"]] = _probs(
        params, (prueba["elo_l"] - prueba["elo_v"]).to_numpy(), prueba["neutral"].astype(float).to_numpy())
    return prueba, entreno


def buscar_parametros_elo(partidos, ks=(8, 10, 12, 15, 20), ventajas=(50, 65, 80)):
    """Prueba combinaciones de K y ventaja local en la temporada de validación."""
    mejor = None
    for k in ks:
        for v in ventajas:
            con_elo, _, _ = elo.calcular(partidos, k=k, ventaja_local=v)
            prueba, _ = probs_fuera_de_muestra(con_elo, config.TEMPORADA_VALIDACION)
            ll = metricas(prueba[["elo_pL", "elo_pE", "elo_pV"]].to_numpy(), prueba["resultado"])["log_loss"]
            print(f"   K={k:2d}  ventaja={v:2d}  log-loss={ll:.4f}")
            if mejor is None or ll < mejor[0]:
                mejor = (ll, k, v)
    return {"k": mejor[1], "ventaja_local": mejor[2]}


def entrenar_final(con_elo):
    """Modelo definitivo: usa todas las temporadas (menos el calentamiento)."""
    validas = config.temporadas()[config.TEMPORADAS_CALENTAMIENTO:]
    return ajustar(con_elo[con_elo["temporada"].isin(validas)])
