"""Rating Elo de clubes europeos.

Idea: cada equipo tiene un puntaje. Antes de cada partido se calcula cuánto
"debería" sacar cada uno; después se compara con lo que pasó de verdad y se
mueven los puntos. Ganarle a un equipo fuerte da más puntos que ganarle a uno
débil, y golear da un poco más que ganar por la mínima.
"""
import pandas as pd

from . import config


def esperado(elo_local, elo_visita, ventaja_local):
    """Puntaje esperado del local (0 a 1) según la fórmula clásica del Elo."""
    return 1 / (1 + 10 ** (-(elo_local + ventaja_local - elo_visita) / 400))


def _multiplicador_goles(dif):
    dif = abs(dif)
    if dif <= 1:
        return 1.0
    if dif == 2:
        return 1.5
    return (11 + dif) / 8


def _elo_inicial(pais, nivel):
    if pais not in config.ELO_BASE_PAIS:
        return config.ELO_BASE_OTROS
    base = config.ELO_BASE_PAIS[pais]
    return base - config.PENALIZACION_SEGUNDA if nivel == 2 else base


def calcular(partidos, k=config.K, ventaja_local=config.VENTAJA_LOCAL):
    """Recorre los partidos en orden y devuelve (partidos con elo previo, ratings, historial).

    Los Elo 'previos' son los que había ANTES del partido: el modelo nunca ve el
    resultado que intenta predecir.
    """
    ratings, pais_equipo = {}, {}
    pre_l, pre_v, post_l, post_v, historial = [], [], [], [], []

    for fila in partidos.itertuples(index=False):
        for equipo in (fila.local, fila.visita):
            if equipo not in ratings:
                if fila.competicion == "CL":
                    ratings[equipo] = config.ELO_BASE_OTROS
                else:
                    ratings[equipo] = _elo_inicial(fila.pais, fila.nivel)
            if fila.competicion != "CL":
                pais_equipo[equipo] = fila.pais

        rl, rv = ratings[fila.local], ratings[fila.visita]
        pre_l.append(rl)
        pre_v.append(rv)

        ventaja = 0 if fila.neutral else ventaja_local
        e = esperado(rl, rv, ventaja)
        real = 1.0 if fila.gl > fila.gv else 0.5 if fila.gl == fila.gv else 0.0
        cambio = k * _multiplicador_goles(fila.gl - fila.gv) * (real - e)
        ratings[fila.local] = rl + cambio
        ratings[fila.visita] = rv - cambio
        post_l.append(ratings[fila.local])
        post_v.append(ratings[fila.visita])
        historial.append((fila.fecha, fila.local, ratings[fila.local]))
        historial.append((fila.fecha, fila.visita, ratings[fila.visita]))

    partidos = partidos.copy()
    partidos["elo_l"] = pre_l
    partidos["elo_v"] = pre_v
    partidos["elo_l_post"] = post_l  # Elo después del partido (sirve para la "racha")
    partidos["elo_v_post"] = post_v
    hist = pd.DataFrame(historial, columns=["fecha", "equipo", "elo"])
    tabla = pd.DataFrame({"equipo": list(ratings), "elo": list(ratings.values())})
    tabla["pais"] = tabla["equipo"].map(pais_equipo).fillna(tabla["equipo"].str[-4:-1])
    return partidos, tabla.sort_values("elo", ascending=False).reset_index(drop=True), hist
