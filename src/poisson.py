"""Modelo de goles: Poisson con corrección Dixon-Coles.

Analogía: cada equipo tiene un "cañón" (ataque) y un "muro" (defensa). Los
goles que se esperan del local salen de su cañón contra el muro del rival,
más un empujón por jugar en casa. Con esos goles esperados, la distribución
de Poisson dice qué tan probable es cada marcador: 0-0, 1-0, 2-1…

Dixon-Coles corrige un defecto conocido: el Poisson puro se equivoca con los
marcadores bajos (0-0, 1-0, 0-1, 1-1).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.stats import poisson as dist_poisson
from sklearn.linear_model import PoissonRegressor

from . import config

VENTANA_DIAS = 3 * 365   # solo se usan los últimos 3 años
MAX_GOLES = 10           # la matriz de marcadores va de 0 a 10 goles por equipo
LINEAS = ("0.5", "1.5", "2.5", "3.5", "4.5")


# ---------------------------------------------------------------- ajuste

def _filas(df):
    """Cada partido da 2 filas: los goles que mete el local y los que mete la visita."""
    ataca = np.concatenate([df["local"].to_numpy(), df["visita"].to_numpy()])
    defiende = np.concatenate([df["visita"].to_numpy(), df["local"].to_numpy()])
    en_casa = np.concatenate([1.0 - df["neutral"].astype(float).to_numpy(), np.zeros(len(df))])
    comp = np.concatenate([df["competicion"].to_numpy()] * 2)
    return ataca, defiende, en_casa, comp


def _diseno(ataca, defiende, en_casa, comp, equipos, ligas):
    """Matriz dispersa: [ataque de cada equipo | defensa de cada equipo | liga | en casa]."""
    n, ne, nl = len(ataca), len(equipos), len(ligas)
    idx_e, idx_l = pd.Index(equipos), pd.Index(ligas)
    ia, idf, il = idx_e.get_indexer(ataca), idx_e.get_indexer(defiende), idx_l.get_indexer(comp)
    r = np.arange(n)
    filas = np.concatenate([r[ia >= 0], r[idf >= 0], r[il >= 0], r])
    cols = np.concatenate([ia[ia >= 0], ne + idf[idf >= 0], 2 * ne + il[il >= 0], np.full(n, 2 * ne + nl)])
    unos = (ia >= 0).sum() + (idf >= 0).sum() + (il >= 0).sum()
    vals = np.concatenate([np.ones(unos), en_casa])
    return sparse.csr_matrix((vals, (filas, cols)), shape=(n, 2 * ne + nl + 1))


def ajustar(partidos, fecha, vida_media, alpha):
    """Calcula ataque y defensa de cada equipo con los partidos ANTERIORES a `fecha`.

    Los recientes pesan más: uno de hace `vida_media` días pesa la mitad.
    `alpha` frena a los equipos con pocos partidos para que no tengan valores extremos.
    """
    fecha = pd.Timestamp(fecha)
    datos = partidos[(partidos["fecha"] < fecha)
                     & (partidos["fecha"] >= fecha - pd.Timedelta(days=VENTANA_DIAS))]
    ataca, defiende, en_casa, comp = _filas(datos)
    y = np.concatenate([datos["gl"].to_numpy(), datos["gv"].to_numpy()]).astype(float)
    dias = (fecha - datos["fecha"]).dt.days.to_numpy()
    peso = np.tile(0.5 ** (dias / vida_media), 2)
    equipos, ligas = np.unique(ataca), np.unique(comp)

    X = _diseno(ataca, defiende, en_casa, comp, equipos, ligas)
    reg = PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-6).fit(X, y, sample_weight=peso)
    ne, nl, c = len(equipos), len(ligas), reg.coef_
    return {
        "intercepto": float(reg.intercept_),
        "casa": float(c[-1]),
        "ataque": {str(e): float(v) for e, v in zip(equipos, c[:ne])},
        "defensa": {str(e): float(v) for e, v in zip(equipos, c[ne:2 * ne])},
        "liga": {str(l): float(v) for l, v in zip(ligas, c[2 * ne:2 * ne + nl])},
    }


def lambdas(params, locales, visitas, competiciones, neutrales):
    """Goles esperados de local y visita. Un equipo desconocido cuenta como promedio."""
    at, de, li = params["ataque"], params["defensa"], params["liga"]
    base = params["intercepto"] + np.array([li.get(c, 0.0) for c in competiciones])
    casa = params["casa"] * (1 - np.asarray(neutrales, dtype=float))
    lam_l = np.exp(base + casa + np.array([at.get(e, 0.0) for e in locales])
                   + np.array([de.get(e, 0.0) for e in visitas]))
    lam_v = np.exp(base + np.array([at.get(e, 0.0) for e in visitas])
                   + np.array([de.get(e, 0.0) for e in locales]))
    return lam_l, lam_v


# ---------------------------------------------------------------- marcadores

def _tau(gl, gv, ll, lv, rho):
    """Corrección Dixon-Coles para los marcadores 0-0, 0-1, 1-0 y 1-1."""
    t = np.ones_like(ll, dtype=float)
    t = np.where((gl == 0) & (gv == 0), 1 - ll * lv * rho, t)
    t = np.where((gl == 0) & (gv == 1), 1 + ll * rho, t)
    t = np.where((gl == 1) & (gv == 0), 1 + lv * rho, t)
    t = np.where((gl == 1) & (gv == 1), 1 - rho, t)
    return np.clip(t, 1e-10, None)


def matrices(lam_l, lam_v, rho, n=MAX_GOLES):
    """Probabilidad de cada marcador. Forma (partidos, goles local, goles visita)."""
    lam_l, lam_v = np.asarray(lam_l, dtype=float), np.asarray(lam_v, dtype=float)
    g = np.arange(n + 1)
    m = dist_poisson.pmf(g[None, :], lam_l[:, None])[:, :, None] * dist_poisson.pmf(g[None, :], lam_v[:, None])[:, None, :]
    m[:, 0, 0] *= 1 - lam_l * lam_v * rho
    m[:, 0, 1] *= 1 + lam_l * rho
    m[:, 1, 0] *= 1 + lam_v * rho
    m[:, 1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)


def mercados_vec(m):
    """1X2, más/menos de X goles y ambos marcan, para muchas matrices a la vez."""
    g = np.arange(m.shape[1])
    total = g[:, None] + g[None, :]
    return {
        "1x2": np.column_stack([(m * (g[:, None] > g[None, :])).sum(axis=(1, 2)),
                                np.trace(m, axis1=1, axis2=2),
                                (m * (g[:, None] < g[None, :])).sum(axis=(1, 2))]),
        "over": {linea: (m * (total > float(linea))).sum(axis=(1, 2)) for linea in LINEAS},
        "btts": m[:, 1:, 1:].sum(axis=(1, 2)),
    }


def mercados(lam_l, lam_v, rho):
    """Todo lo que se puede decir de un partido a partir de sus goles esperados."""
    m = matrices([lam_l], [lam_v], rho)[0]
    mk = mercados_vec(m[None])
    orden = np.argsort(m, axis=None)[::-1][:5]
    return {
        "1x2": {"L": float(mk["1x2"][0, 0]), "E": float(mk["1x2"][0, 1]), "V": float(mk["1x2"][0, 2])},
        "goles": (float(lam_l), float(lam_v)),
        "over": {linea: float(v[0]) for linea, v in mk["over"].items()},
        "btts": float(mk["btts"][0]),
        "top": [(int(i // m.shape[1]), int(i % m.shape[1]), float(m.flat[i])) for i in orden],
        "matriz": m[:6, :6].tolist(),
    }


# ---------------------------------------------------------------- evaluación y parámetros

def _prueba(partidos, temporada):
    return partidos[(partidos["temporada"] == temporada)
                    & partidos["competicion"].isin(config.LIGAS_PRINCIPALES + ["CL"])]


def backtest(partidos, temporada, vida_media, alpha, cada_dias=30):
    """Predice una temporada como si fuera en vivo: reajusta cada `cada_dias` solo con el pasado."""
    prueba = _prueba(partidos, temporada)
    lam = pd.DataFrame(index=prueba.index, columns=["lam_l", "lam_v"], dtype=float)
    paso = pd.Timedelta(days=cada_dias)
    d, fin = prueba["fecha"].min().normalize(), prueba["fecha"].max()
    while d <= fin:
        bloque = prueba[(prueba["fecha"] >= d) & (prueba["fecha"] < d + paso)]
        if len(bloque):
            params = ajustar(partidos, d, vida_media, alpha)
            ll, lv = lambdas(params, bloque["local"], bloque["visita"], bloque["competicion"], bloque["neutral"])
            lam.loc[bloque.index, "lam_l"] = ll
            lam.loc[bloque.index, "lam_v"] = lv
        d += paso
    return lam


def lambdas_historicos(partidos, vida_media, alpha, desde="2014-07-01", cache=None):
    """Goles esperados 'en vivo' para cada partido desde `desde`.

    Cada mes se reajusta el modelo solo con partidos anteriores, así el ML puede usar
    los goles esperados como dato sin hacer trampa. Se guarda en `cache` para no
    recalcular los meses viejos (se recalculan siempre los 2 últimos).
    """
    claves = ["fecha", "local", "visita"]
    lam = pd.DataFrame(index=partidos.index, columns=["lam_l", "lam_v"], dtype=float)
    hechos = set()
    if cache is not None and cache.exists():
        g = pd.read_pickle(cache)
        if len(g) and g["vida_media"].iat[0] == vida_media and g["alpha"].iat[0] == alpha:
            g = g[g["mes"] < g["mes"].max() - pd.DateOffset(months=1)]
            hechos = set(g["mes"])
            m = partidos[claves].reset_index().merge(g, on=claves, how="inner")
            lam.loc[m["index"], ["lam_l", "lam_v"]] = m[["lam_l", "lam_v"]].to_numpy()

    meses = pd.date_range(pd.Timestamp(desde), partidos["fecha"].max(), freq="MS")
    pendientes = [m for m in meses if m not in hechos]
    for i, mes in enumerate(pendientes, 1):
        bloque = partidos[(partidos["fecha"] >= mes) & (partidos["fecha"] < mes + pd.DateOffset(months=1))]
        if bloque.empty:
            continue
        params = ajustar(partidos, mes, vida_media, alpha)
        ll, lv = lambdas(params, bloque["local"], bloque["visita"], bloque["competicion"], bloque["neutral"])
        lam.loc[bloque.index, "lam_l"] = ll
        lam.loc[bloque.index, "lam_v"] = lv
        if i % 12 == 0:
            print(f"   … {i}/{len(pendientes)} meses")

    if cache is not None:
        out = partidos[claves].join(lam).dropna(subset=["lam_l"])
        out["mes"] = out["fecha"].dt.to_period("M").dt.to_timestamp()
        out["vida_media"], out["alpha"] = vida_media, alpha
        out.to_pickle(cache)
    return lam


def nll_marcador(lam_l, lam_v, gl, gv, rho=0.0):
    """Qué tan sorprendido quedó el modelo con el marcador real (menor = mejor)."""
    lam_l, lam_v, gl, gv = (np.asarray(x, dtype=float) for x in (lam_l, lam_v, gl, gv))
    return float(-(dist_poisson.logpmf(gl, lam_l) + dist_poisson.logpmf(gv, lam_v)
                   + np.log(_tau(gl, gv, lam_l, lam_v, rho))).mean())


def ajustar_rho(lam_l, lam_v, gl, gv):
    res = minimize_scalar(lambda r: nll_marcador(lam_l, lam_v, gl, gv, r), bounds=(-0.3, 0.3), method="bounded")
    return float(res.x)


def buscar_parametros(partidos, vidas=(240, 480, 720, 1000), alphas=(1e-4, 3e-4, 1e-3)):
    """Elige vida media y regularización mirando la temporada de validación."""
    prueba = _prueba(partidos, config.TEMPORADA_VALIDACION)
    mejor = None
    for v in vidas:
        for a in alphas:
            lam = backtest(partidos, config.TEMPORADA_VALIDACION, v, a)
            nll = nll_marcador(lam["lam_l"], lam["lam_v"], prueba["gl"], prueba["gv"])
            print(f"   vida media={v:3d} días  alpha={a:.4f}  log-loss marcador={nll:.4f}")
            if mejor is None or nll < mejor[0]:
                mejor = (nll, v, a)
    return {"vida_media": mejor[1], "alpha": mejor[2]}
