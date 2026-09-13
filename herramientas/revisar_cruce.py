"""Muestra cómo se cruzaron los nombres de Champions (openfootball) con los de las ligas.

Uso: python herramientas/revisar_cruce.py
Si algún equipo está mal cruzado, agrégalo a ALIAS en src/datos.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config, datos

p, sin = datos.cargar_partidos()
print(p.groupby("competicion").size().to_string())
print("total", len(p), "rango", p.fecha.min().date(), p.fecha.max().date())
print("CL por temporada:", p[p.competicion == "CL"].groupby("temporada").size().to_dict())
print("SIN CRUCE:", sin)

ligas = p[p.competicion != "CL"]
filas = []
for temp in config.temporadas():
    ruta = config.RAW / f"cl_{temp}.txt"
    if ruta.exists():
        filas += datos.parsear_champions(ruta, temp)
raw = pd.DataFrame(filas)
equipos = (pd.concat([ligas[["local", "pais"]].rename(columns={"local": "e"}),
                      ligas[["visita", "pais"]].rename(columns={"visita": "e"})])
           .drop_duplicates().groupby("pais")["e"].apply(set).to_dict())
mapa, _ = datos.cruzar_nombres(list(raw.local_cl) + list(raw.visita_cl), equipos)
for k in sorted(mapa, key=lambda x: (x[-4:], x)):
    print(f"{k:45s} -> {mapa[k]}")

cl = p[p.competicion == "CL"]
conocidos = set(ligas.local)
print("CL sin liga:", sorted(n for n in set(cl.local) | set(cl.visita) if n not in conocidos))
