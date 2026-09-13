"""Fase 3: machine learning con gradient boosting.

Analogía: es un scout que vio 60,000 partidos. No le decimos qué importa; él
arma cientos de "reglas" pequeñas (árboles de decisión), cada una corrigiendo
los errores de las anteriores. Por ejemplo: "si el local viene de 3 derrotas pero
tiene muchos tiros al arco, no está tan mal como dicen los puntos".

HistGradientBoosting (scikit-learn) es de la misma familia que XGBoost.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from . import features

MERCADOS = ("1x2", "over25", "btts")
# Con datos tan ruidosos como el fútbol ganan los modelos simples (pocas hojas, aprendizaje lento)
GRILLA = [{"learning_rate": lr, "max_leaf_nodes": h, "max_iter": it}
          for lr in (0.015, 0.03) for h in (7, 15) for it in (100, 200, 300)]


def objetivo(df, mercado):
    if mercado == "1x2":
        return df["resultado"].map({"L": 0, "E": 1, "V": 2}).to_numpy()
    if mercado == "over25":
        return ((df["gl"] + df["gv"]) > 2.5).astype(int).to_numpy()
    return ((df["gl"] > 0) & (df["gv"] > 0)).astype(int).to_numpy()


def etiquetas(mercado):
    return [0, 1, 2] if mercado == "1x2" else [0, 1]


def entrenar(X, y, params):
    return HistGradientBoostingClassifier(**params, min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=0).fit(X, y)


def predecir(modelo, X, mercado):
    p = modelo.predict_proba(X)
    return p if mercado == "1x2" else p[:, 1]


def buscar(X_tr, y_tr, X_va, y_va, mercado):
    """Prueba la grilla y devuelve (mejores parámetros, predicciones de validación)."""
    mejor = None
    for params in GRILLA:
        p = predecir(entrenar(X_tr, y_tr, params), X_va, mercado)
        ll = log_loss(y_va, np.clip(p, 1e-12, 1), labels=etiquetas(mercado))
        print(f"   {mercado:6s} lr={params['learning_rate']:.2f} hojas={params['max_leaf_nodes']:2d} "
              f"árboles={params['max_iter']:3d}  log-loss={ll:.4f}")
        if mejor is None or ll < mejor[0]:
            mejor = (ll, params, p)
    return mejor[1], mejor[2]


def importancia(modelo, X, y, n=4000):
    """Cuánto empeora el modelo si 'desordenamos' cada variable (más = más importante)."""
    idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
    r = permutation_importance(modelo, X.iloc[idx], y[idx], scoring="neg_log_loss", n_repeats=3, random_state=0)
    orden = np.argsort(r.importances_mean)[::-1]
    return [(features.nombre(X.columns[i]), float(r.importances_mean[i])) for i in orden]
