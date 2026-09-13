"""Junta Elo, Poisson, ML, stacking y lesionados para predecir un partido."""
import json

import joblib
import numpy as np
import pandas as pd

from . import bajas, config, evaluacion, features, ml, modelo, poisson


def _leer_tabla(nombre, **kw):
    return pd.read_csv(config.DATA / nombre, **kw)


class Predictor:
    def __init__(self):
        self.info = json.loads((config.DATA / "modelo.json").read_text(encoding="utf-8"))
        self.params_poisson = json.loads((config.DATA / "poisson.json").read_text(encoding="utf-8"))
        self.tabla = _leer_tabla("elo_actual.csv")
        self.ml = joblib.load(config.DATA / "ml.pkl")
        self.estado = _leer_tabla("estado.csv", index_col="equipo", parse_dates=["fecha"])
        self.elo = dict(zip(self.tabla["equipo"], self.tabla["elo"]))
        self.liga = dict(zip(self.tabla["equipo"], self.tabla["liga"]))
        self.rho = self.info["poisson"]["rho"]
        self.stack = self.ml["stack"]

        # fase 4: lesionados (opcional: si no hay datos, la app funciona igual)
        self.cal_bajas = self.info.get("bajas")
        self.lesionados, self.valor_plantel, self.fecha_bajas = None, {}, None
        if self.cal_bajas and (config.DATA / "bajas.csv").exists():
            self.lesionados = _leer_tabla("bajas.csv", parse_dates=["hasta"])
            pla = _leer_tabla("planteles.csv")
            self.valor_plantel = dict(zip(pla["equipo"], pla["valor_plantel"]))
            if bajas.RUTA_FECHA.exists():
                self.fecha_bajas = bajas.RUTA_FECHA.read_text(encoding="utf-8").strip()

    def conoce(self, equipo):
        return equipo in self.elo

    def impacto_bajas(self, equipo, fecha=None):
        """None si no hay datos de Transfermarkt para ese equipo."""
        if self.lesionados is None or equipo not in self.valor_plantel:
            return None
        return bajas.impacto(self.lesionados[self.lesionados["equipo"] == equipo],
                             self.valor_plantel[equipo], self.cal_bajas, fecha)

    def partido(self, local, visita, competicion=None, neutral=False, fecha=None, con_bajas=True, cuotas=None):
        """Probabilidades finales + todos los mercados de goles.

        `cuotas`: dict con cuota_l, cuota_e, cuota_v, cuota_o25, cuota_u25 (si se conocen).
        """
        if competicion is None:
            # mismos equipos de liga -> esa liga; si son de ligas distintas -> como partido de Champions
            competicion = self.liga.get(local) if self.liga.get(local) == self.liga.get(visita) else "CL"
        nivel = 0 if competicion == "CL" else config.LIGAS[competicion][2]

        # fase 4: las bajas restan Elo y mueven los goles esperados
        imp_l = self.impacto_bajas(local, fecha) if con_bajas else None
        imp_v = self.impacto_bajas(visita, fecha) if con_bajas else None
        d_l = imp_l["delta_elo"] if imp_l else 0.0
        d_v = imp_v["delta_elo"] if imp_v else 0.0
        elo_l, elo_v = self.elo[local] + d_l, self.elo[visita] + d_v
        ll, lv = poisson.lambdas(self.params_poisson, [local], [visita], [competicion], [neutral])
        ajuste = np.exp(self.cal_bajas["log_goles_por_elo"] * (d_l - d_v) / 2) if (d_l or d_v) else 1.0
        ll, lv = float(ll[0] * ajuste), float(lv[0] / ajuste)

        # fases 1-3: cada modelo opina
        p_elo = modelo.predecir(self.info["params"], elo_l, elo_v, neutral)
        mk = poisson.mercados(ll, lv, self.rho)
        X = features.vector(self.estado, local, visita, elo_l, elo_v, ll, lv, neutral, nivel, fecha, cuotas)
        p_ml = {m: ml.predecir(self.ml["modelos"][m], X, m) for m in ml.MERCADOS}

        # stacking: combina y calibra
        preds = {
            "1x2": {"Elo": np.array([[p_elo[k] for k in "LEV"]]), "Poisson": np.array([[mk["1x2"][k] for k in "LEV"]]),
                    "ML": p_ml["1x2"]},
            "over25": {"Poisson": np.array([mk["over"]["2.5"]]), "ML": p_ml["over25"]},
            "btts": {"Poisson": np.array([mk["btts"]]), "ML": p_ml["btts"]},
        }
        final = {m: evaluacion.predecir_stack(self.stack[m], preds[m])[0] for m in ml.MERCADOS}

        return {
            **mk,
            "1x2": dict(zip("LEV", map(float, final["1x2"]))),
            "over": {**mk["over"], "2.5": float(final["over25"])},
            "btts": float(final["btts"]),
            "elo_1x2": p_elo,
            "poisson_1x2": mk["1x2"],
            "ml_1x2": dict(zip("LEV", map(float, p_ml["1x2"][0]))),
            "competicion": competicion,
            "bajas": {"L": imp_l, "V": imp_v},
            "elo_ajustado": (elo_l, elo_v),
        }
