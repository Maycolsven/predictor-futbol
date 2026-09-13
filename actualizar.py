"""Baja los datos nuevos, recalcula Elo, Poisson, ML y stacking, evalúa y registra predicciones.

Uso: python actualizar.py          (completo: busca parámetros y hace el examen; ~15 min)
     python actualizar.py --rapido (reusa parámetros y examen guardados; ~3 min)
     --sin-bajas                   (no vuelve a bajar los lesionados de Transfermarkt)
"""
import json
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src import bajas, config, datos, elo, evaluacion, features, ml, modelo, poisson, registro
from src.prediccion import Predictor


def imprimir_resumen(ev):
    print(f"\nExamen final: temporadas {', '.join(ev['temporadas'])} ({ev['partidos']:,} partidos, promedio)")
    for titulo, clave in (("1X2", "1x2"), ("1X2 (solo partidos con cuotas)", "1x2_con_cuotas"),
                          ("Más de 2.5", "over25"), ("Más de 2.5 (con cuotas)", "over25_con_cuotas"),
                          ("Ambos marcan", "btts")):
        print(f"  {titulo}")
        for nombre, m in ev[clave].items():
            print(f"   {nombre:10s} log-loss {m['log_loss']:.4f}  acierto {m['acierto']:.1%}")
    print(f"  Marcador exacto: acierto {ev['marcador']['acierto']:.1%}")
    print("  Por temporada (1X2 Final / Casas):")
    for t, e in ev["por_temporada"].items():
        f, c = e["1x2_con_cuotas"]["Final"], e["1x2_con_cuotas"]["Casas"]
        print(f"   {t}: {f['log_loss']:.4f} / {c['log_loss']:.4f}   acierto {f['acierto']:.1%} / {c['acierto']:.1%}")
    if "importancia" in ev:
        print("  Lo que más mira el ML (1X2):", ", ".join(n for n, _ in ev["importancia"][:5]))


def predicciones_temporada(con_elo, feats, temporada, params_ml, rho):
    """Predicciones fuera de muestra de Elo, Poisson y ML para una temporada."""
    prueba, entreno = modelo.probs_fuera_de_muestra(con_elo, temporada)
    prueba = prueba.dropna(subset=["lam_l"])
    m = poisson.matrices(prueba["lam_l"].to_numpy(), prueba["lam_v"].to_numpy(), rho)
    mk = poisson.mercados_vec(m)
    usable = entreno.index[entreno["lam_l"].notna()]
    preds = {"1x2": {"Elo": prueba[["elo_pL", "elo_pE", "elo_pV"]].to_numpy(), "Poisson": mk["1x2"]},
             "over25": {"Poisson": mk["over"]["2.5"]}, "btts": {"Poisson": mk["btts"]}}
    modelos = {}
    for mercado in ml.MERCADOS:
        modelos[mercado] = ml.entrenar(feats.loc[usable], ml.objetivo(con_elo.loc[usable], mercado), params_ml[mercado])
        preds[mercado]["ML"] = ml.predecir(modelos[mercado], feats.loc[prueba.index], mercado)
    return prueba, preds, m, modelos, entreno


def main():
    print("1/8 Descargando datos…")
    datos.descargar_todo()

    print("2/8 Cargando partidos…")
    partidos, sin_cruce, programados_cl = datos.cargar_partidos(con_programados=True)
    print(f"   {len(partidos):,} partidos ({partidos.fecha.min().date()} a {partidos.fecha.max().date()})")
    if sin_cruce:
        print("   ! Equipos de Champions sin cruzar (agrégalos a ALIAS en src/datos.py):", sin_cruce)
    fixtures = datos.cargar_fixtures(programados_cl)

    ruta_modelo = config.DATA / "modelo.json"
    anterior = json.loads(ruta_modelo.read_text(encoding="utf-8")) if ruta_modelo.exists() else {}
    rapido = "--rapido" in sys.argv and all(k in anterior for k in ("elo", "poisson", "ml", "evaluacion")) \
        and "temporadas" in anterior["evaluacion"] and (config.DATA / "ml.pkl").exists()
    val_t = config.TEMPORADA_VALIDACION

    # ---- fase 1: Elo
    if rapido:
        param_elo = anterior["elo"]
        print("3/8 Elo con parámetros guardados:", param_elo)
    else:
        print(f"3/8 Elo: buscando K y ventaja local (validación {val_t})…")
        param_elo = modelo.buscar_parametros_elo(partidos)
        print("   Mejor:", param_elo)
    con_elo, tabla, historial = elo.calcular(partidos, **param_elo)
    params_elo = modelo.entrenar_final(con_elo)
    tabla = datos.agregar_liga_actual(tabla, partidos)

    # ---- fase 2: Poisson
    if rapido:
        cfg_p = anterior["poisson"]
        print("4/8 Poisson con parámetros guardados:", cfg_p)
    else:
        print(f"4/8 Poisson: buscando vida media y regularización (validación {val_t})…")
        cfg_p = poisson.buscar_parametros(partidos)
        print("   Mejor:", cfg_p)
    print("   Goles esperados 'en vivo' de cada partido desde 2014 (los meses viejos quedan guardados)…")
    lam = poisson.lambdas_historicos(partidos, cfg_p["vida_media"], cfg_p["alpha"],
                                     cache=config.DATA / "lambdas_hist.pkl")
    con_elo = con_elo.join(lam)
    if not rapido:
        val, _ = modelo.probs_fuera_de_muestra(con_elo, val_t)
        cfg_p["rho"] = poisson.ajustar_rho(val["lam_l"], val["lam_v"], val["gl"], val["gv"])
        print(f"   Dixon-Coles rho = {cfg_p['rho']:.3f}")

    # ---- fase 3: ML
    print("5/8 Armando las features del ML…")
    feats = features.historicas(con_elo, lam)
    validas = config.temporadas()[config.TEMPORADAS_CALENTAMIENTO:]
    usable = con_elo["temporada"].isin(validas) & con_elo["lam_l"].notna()
    if rapido:
        params_ml = anterior["ml"]["params"]
        print("6/8 ML con parámetros guardados")
    else:
        print(f"6/8 ML: buscando parámetros (entrena antes de {val_t}, valida en {val_t})…")
        val, _ = modelo.probs_fuera_de_muestra(con_elo, val_t)
        entreno_ml = usable & (con_elo["temporada"] < val_t)
        params_ml = {}
        for m in ml.MERCADOS:
            params_ml[m], _ = ml.buscar(feats[entreno_ml], ml.objetivo(con_elo[entreno_ml], m),
                                        feats.loc[val.index], ml.objetivo(val, m), m)

    # ---- examen rotativo + stacking
    stack_anterior = joblib.load(config.DATA / "ml.pkl")["stack"] if rapido else None
    if rapido:
        ev, stack = anterior["evaluacion"], stack_anterior
        print("7/8 Examen: se reusa el guardado")
    else:
        print(f"7/8 Examen rotativo en {config.TEMPORADAS_EXAMEN} (cada temporada solo con las anteriores)…")
        oos, evaluaciones, pool = {}, {}, {m: {} for m in ml.MERCADOS}
        for t in config.TEMPORADAS_EXAMEN:
            prueba, preds, matrices, modelos_t, entreno = predicciones_temporada(con_elo, feats, t, params_ml, cfg_p["rho"])
            oos[t] = (prueba, preds, matrices, entreno)
            for m in ml.MERCADOS:
                pool[m].setdefault("preds", []).append(preds[m])
                pool[m].setdefault("y", []).append(ml.objetivo(prueba, m))
            print(f"   {t}: {len(prueba)} partidos")
        for i, t in enumerate(config.TEMPORADAS_EXAMEN[1:], 1):
            t_prev = config.TEMPORADAS_EXAMEN[i - 1]
            prueba, preds, matrices, entreno = oos[t]
            prueba_prev, preds_prev, _, _ = oos[t_prev]
            for m in ml.MERCADOS:
                st = evaluacion.ajustar_stack(preds_prev[m], ml.objetivo(prueba_prev, m), m)
                preds[m]["Final"] = evaluacion.predecir_stack(st, preds[m])
            evaluaciones[t] = evaluacion.evaluar_temporada(prueba, preds, matrices, evaluacion.frecuencias(entreno))
        ev = evaluacion.promediar(evaluaciones)
        ev["por_temporada"] = evaluaciones
        # stacking definitivo: aprende de todas las temporadas fuera de muestra
        stack = {}
        for m in ml.MERCADOS:
            juntos = {k: np.concatenate([p[k] for p in pool[m]["preds"]]) for k in pool[m]["preds"][0]}
            stack[m] = evaluacion.ajustar_stack(juntos, np.concatenate(pool[m]["y"]), m)
        ultima, _, _, _ = oos[config.TEMPORADAS_EXAMEN[-1]]
        ev["importancia"] = ml.importancia(modelos_t["1x2"], feats.loc[ultima.index], ml.objetivo(ultima, "1x2"))

    # ---- fase 4: lesionados
    cal_bajas = anterior.get("bajas")
    if "--sin-bajas" not in sys.argv:
        print("   Lesionados de Transfermarkt (32 páginas con pausas, ~1.5 min)…")
        try:
            bajas.descargar()
        except Exception as e:  # la web puede cambiar o bloquear; el resto del pipeline sigue
            print(f"   ! Falló la descarga de lesionados: {e}")
    if bajas.RUTA_PLANTELES_TM.exists():
        les, pla, sin_cruce_tm = bajas.cargar(partidos)
        if sin_cruce_tm:
            print("   ! Clubes de Transfermarkt sin cruzar (agrégalos a ALIAS_TM en src/bajas.py):", sin_cruce_tm)
        cal_bajas = bajas.calibrar(tabla, pla, con_elo)
        les.to_csv(config.DATA / "bajas.csv", index=False)
        pla.to_csv(config.DATA / "planteles.csv", index=False)
        print(f"   {len(les)} lesionados en {les['equipo'].nunique()} equipos")

    print("8/8 Ajuste final con todos los datos, guardado y registro de predicciones…")
    manana = partidos["fecha"].max() + pd.Timedelta(days=1)
    params_p = poisson.ajustar(partidos, manana, cfg_p["vida_media"], cfg_p["alpha"])
    modelos = {m: ml.entrenar(feats[usable], ml.objetivo(con_elo[usable], m), params_ml[m]) for m in ml.MERCADOS}

    config.DATA.mkdir(exist_ok=True)
    columnas = ["fecha", "local", "visita", "gl", "gv", "competicion", "temporada", "neutral", "resultado",
                "elo_l", "elo_v", "cuota_l", "cuota_e", "cuota_v", "cuota_o25", "cuota_u25"]
    con_elo[columnas].to_csv(config.DATA / "partidos.csv", index=False, date_format="%Y-%m-%d")
    desde = historial["fecha"].max() - pd.DateOffset(years=5)
    historial[historial["fecha"] >= desde].to_csv(config.DATA / "historial_elo.csv", index=False, date_format="%Y-%m-%d")
    tabla.to_csv(config.DATA / "elo_actual.csv", index=False)
    features.estado_actual(con_elo).to_csv(config.DATA / "estado.csv", date_format="%Y-%m-%d")
    fixtures.to_csv(config.DATA / "fixtures.csv", index=False, date_format="%Y-%m-%d")
    joblib.dump({"modelos": modelos, "stack": stack, "features": features.FEATURES}, config.DATA / "ml.pkl")
    (config.DATA / "poisson.json").write_text(json.dumps(params_p, ensure_ascii=False), encoding="utf-8")
    ruta_modelo.write_text(json.dumps({
        "actualizado": datetime.now().isoformat(timespec="minutes"),
        "elo": param_elo,
        "params": params_elo,
        "poisson": cfg_p,
        "ml": {"params": params_ml},
        "bajas": cal_bajas,
        "evaluacion": ev,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # fase 5: registrar predicciones de los próximos días y cruzar resultados
    pred = Predictor()
    nuevos = registro.registrar(pred, fixtures)
    reg = registro.resolver(partidos)
    print(f"   Registro: {nuevos} predicciones nuevas · {int(reg['resultado'].notna().sum()) if len(reg) else 0} ya resueltas")

    if not rapido:
        imprimir_resumen(ev)
    print("\nTop 10 Elo:")
    print(tabla.head(10)[["equipo", "elo", "liga"]].to_string(index=False))


if __name__ == "__main__":
    main()
