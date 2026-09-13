"""Descarga y limpieza de datos.

Fuentes:
- football-data.co.uk: resultados de ligas europeas + próximos partidos con cuotas.
- openfootball (GitHub): resultados de Champions League, que conectan las ligas
  entre sí (así el Elo de un equipo inglés se puede comparar con uno alemán).
"""
import difflib
import re
import unicodedata
from datetime import date

import pandas as pd
import requests

from . import config

URL_LIGA = "https://www.football-data.co.uk/mmz4281/{temp}/{div}.csv"
URL_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
URL_CL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{temp}/cl.txt"


# ---------------------------------------------------------------- descargas

def _descargar(url, destino, forzar=False):
    """Descarga url a destino (si no existe o si forzar). Devuelve True si hay archivo."""
    if destino.exists() and not forzar:
        return True
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "predictor-futbol"})
    except requests.RequestException as e:
        print(f"  ! sin conexión para {url}: {e}")
        return destino.exists()
    if r.status_code != 200 or len(r.content) < 200:
        return destino.exists()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    return True


def descargar_todo(forzar_actual=True):
    """Baja las temporadas que falten. La temporada en curso se vuelve a bajar siempre."""
    temps = config.temporadas()
    for temp in temps:
        actual = temp == temps[-1]
        for div in config.LIGAS:
            _descargar(URL_LIGA.format(temp=temp, div=div),
                       config.RAW / f"{temp}_{div}.csv", forzar=actual and forzar_actual)
        # openfootball usa "2025-26"
        a = 2000 + int(temp[:2])
        carpeta = f"{a}-{(a + 1) % 100:02d}"
        _descargar(URL_CL.format(temp=carpeta), config.RAW / f"cl_{temp}.txt",
                   forzar=actual and forzar_actual)
    _descargar(URL_FIXTURES, config.RAW / "fixtures.csv", forzar=True)


# ---------------------------------------------------------------- ligas

def _leer_csv(ruta):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(ruta, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


def _cuotas(df):
    """Cuotas promedio 1X2 (los nombres de columna cambiaron con los años)."""
    for h, d, a in (("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")):
        if h in df.columns:
            return df[h], df[d], df[a]
    nan = pd.Series(float("nan"), index=df.index)
    return nan, nan, nan


def _cuotas_ou(df):
    """Cuotas promedio de más / menos de 2.5 goles."""
    for o, u in (("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("B365>2.5", "B365<2.5")):
        if o in df.columns:
            return pd.to_numeric(df[o], errors="coerce"), pd.to_numeric(df[u], errors="coerce")
    nan = pd.Series(float("nan"), index=df.index)
    return nan, nan


def _num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def cargar_ligas():
    filas = []
    for temp in config.temporadas():
        for div, (_, pais, nivel) in config.LIGAS.items():
            ruta = config.RAW / f"{temp}_{div}.csv"
            if not ruta.exists():
                continue
            df = _leer_csv(ruta)
            if df.empty or "HomeTeam" not in df.columns:
                continue
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            ch, cd, ca = _cuotas(df)
            co, cu = _cuotas_ou(df)
            filas.append(pd.DataFrame({
                "fecha": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
                "local": df["HomeTeam"].str.strip(),
                "visita": df["AwayTeam"].str.strip(),
                "gl": df["FTHG"].astype(int),
                "gv": df["FTAG"].astype(int),
                "competicion": div,
                "pais": pais,
                "nivel": nivel,
                "temporada": temp,
                "neutral": False,
                "cuota_l": pd.to_numeric(ch, errors="coerce"),
                "cuota_e": pd.to_numeric(cd, errors="coerce"),
                "cuota_v": pd.to_numeric(ca, errors="coerce"),
                "cuota_o25": co,
                "cuota_u25": cu,
                # tiros totales y tiros al arco (la Champions de openfootball no los trae)
                "tiros_l": _num(df, "HS"),
                "tiros_v": _num(df, "AS"),
                "tap_l": _num(df, "HST"),
                "tap_v": _num(df, "AST"),
            }))
    return pd.concat(filas, ignore_index=True)


# ---------------------------------------------------------------- Champions

_MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_RE_FECHA = re.compile(r"^\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\w{3}) (\d{1,2})(?: (\d{4}))?\s*$")
_RE_PARTIDO = re.compile(
    r"^\s+(?:\d{1,2}:\d{2}\s+)?(?P<l>.+?\(\w{3}\))\s+v\s+(?P<v>.+?\(\w{3}\))\s+(?P<res>\d.*)$")
# partido programado, todavía sin resultado
_RE_PROGRAMADO = re.compile(
    r"^\s+(?:(?P<hora>\d{1,2}:\d{2})\s+)?(?P<l>.+?\(\w{3}\))\s+v\s+(?P<v>.+?\(\w{3}\))\s*$")
_RE_MARCADOR = re.compile(r"(\d+)-(\d+)")


def _marcador_90(res):
    """Marcador a los 90'. En '3-2 a.e.t. (3-0, 1-0)' el de 90' es el primero entre paréntesis."""
    if "a.e.t." in res:
        par = res[res.index("(") + 1:] if "(" in res else ""
        m = _RE_MARCADOR.search(par)
    else:
        m = _RE_MARCADOR.search(res)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parsear_champions(ruta, temp, programados=None):
    """Partidos jugados del archivo. Si se pasa una lista `programados`, ahí van los que faltan jugar."""
    anio_ini = 2000 + int(temp[:2])
    fecha, fase, hora = None, "", ""
    filas = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if linea.startswith("▪"):
            fase = linea.lstrip("▪ ").strip()
            continue
        m = _RE_FECHA.match(linea)
        if m:
            # El año casi nunca viene escrito y los grupos reinician en septiembre,
            # así que se deduce por la temporada: jul-dic = año inicial, ene-jun = siguiente.
            mes = _MESES[m.group(1)]
            anio = int(m.group(3)) if m.group(3) else anio_ini + (mes < 7)
            fecha = date(anio, mes, int(m.group(2)))
            continue
        m = _RE_PARTIDO.match(linea)
        if m and fecha:
            marcador = _marcador_90(m.group("res"))
            if marcador:
                filas.append({"fecha": pd.Timestamp(fecha), "local_cl": m.group("l").strip(),
                              "visita_cl": m.group("v").strip(), "gl": marcador[0], "gv": marcador[1],
                              "fase": fase, "temporada": temp})
            continue
        m = _RE_PROGRAMADO.match(linea)
        if m and fecha and programados is not None:
            hora = m.group("hora") or hora
            programados.append({"fecha": pd.Timestamp(fecha), "hora": hora, "local_cl": m.group("l").strip(),
                                "visita_cl": m.group("v").strip(), "fase": fase})
    return filas


# Nombre openfootball -> nombre football-data cuando el cruce automático no basta.
ALIAS = {
    "Club Atlético de Madrid": "Ath Madrid", "Atlético Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "FC Bayern München": "Bayern Munich",
    "Bor. Mönchengladbach": "M'gladbach", "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Internazionale Milano": "Inter", "Inter": "Inter",
    "Paris Saint-Germain FC": "Paris SG", "Paris Saint-Germain": "Paris SG",
    "Sporting Clube de Portugal": "Sp Lisbon", "Sporting CP": "Sp Lisbon",
    "SC Braga": "Sp Braga", "Sporting Braga": "Sp Braga", "Sporting Clube de Braga": "Sp Braga",
    "Stade Rennais": "Rennes", "Stade Rennais FC": "Rennes",
    "İstanbul Başakşehir": "Buyuksehyr", "İstanbul Başakşehir FK": "Buyuksehyr",
    "Royale Union Saint-Gilloise": "St. Gilloise",
    # equipos de países sin liga descargada que aparecen con varios nombres
    "FC Red Bull Salzburg": "RB Salzburg (AUT)",
    "PSV": "PSV Eindhoven",
    "Manchester City FC": "Man City", "Manchester City": "Man City",
    "Manchester United FC": "Man United", "Manchester United": "Man United",
}
# Países de openfootball que juegan en una liga de otro país
PAIS_EQUIVALENTE = {"MCO": "FRA", "WAL": "ENG"}


def _norm(nombre):
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(\w{3}\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    ruido = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "sl", "sk", "bc", "club", "de", "cd", "rc",
             "rcd", "ud", "sd", "fk", "pae", "sfp", "kv", "bv", "vfl", "vfb", "tsg", "sv", "us", "ogc",
             "losc", "calcio", "krc", "kaa", "rsc", "e", "ss", "1893", "1899", "1900", "1907", "04", "05"}
    return " ".join(t for t in s.split() if t not in ruido)


def cruzar_nombres(nombres_cl, equipos_por_pais):
    """Devuelve {nombre_openfootball: nombre_football_data} y lista de sin cruce."""
    mapa, sin_cruce = {}, []
    canonicos = {}  # país sin liga -> nombres ya elegidos como "oficiales"
    # los nombres cortos primero, para que sean los oficiales ("Qarabağ FK" antes que "Qarabağ Ağdam FK")
    for nombre in sorted(set(nombres_cl), key=lambda x: (len(x), x)):
        pais = nombre[-4:-1]
        pais = PAIS_EQUIVALENTE.get(pais, pais)
        limpio = nombre[:-5].strip()
        if limpio in ALIAS:
            mapa[nombre] = ALIAS[limpio]
            continue
        if pais not in equipos_por_pais:
            # país que no descargamos: se unifica con un nombre parecido del mismo país
            n = _norm(limpio)
            for c in canonicos.setdefault(pais, []):
                nc = _norm(c[:-5])
                if nc in n or n in nc or difflib.SequenceMatcher(None, n, nc).ratio() >= 0.8:
                    mapa[nombre] = c
                    break
            else:
                canonicos[pais].append(nombre)
            continue
        candidatos = equipos_por_pais[pais]
        n = _norm(limpio)
        mejor, puntaje = None, 0.0
        for c in candidatos:
            nc = _norm(c)
            p = difflib.SequenceMatcher(None, n, nc).ratio()
            if nc and (nc in n or n in nc):
                p = max(p, 0.9)
            if p > puntaje:
                mejor, puntaje = c, p
        if puntaje >= 0.6:
            mapa[nombre] = mejor
        else:
            sin_cruce.append((nombre, mejor, round(puntaje, 2)))
    return mapa, sin_cruce


def cargar_champions(ligas):
    """Devuelve (partidos jugados de Champions, sin_cruce, programados con nombres ya cruzados)."""
    filas, programados = [], []
    for temp in config.temporadas():
        ruta = config.RAW / f"cl_{temp}.txt"
        if ruta.exists():
            filas += parsear_champions(ruta, temp, programados if temp == config.temporadas()[-1] else None)
    cl = pd.DataFrame(filas)
    prog = pd.DataFrame(programados, columns=["fecha", "hora", "local_cl", "visita_cl", "fase"])
    if cl.empty:
        return cl, [], prog
    equipos = (pd.concat([ligas[["local", "pais"]].rename(columns={"local": "e"}),
                          ligas[["visita", "pais"]].rename(columns={"visita": "e"})])
               .drop_duplicates().groupby("pais")["e"].apply(set).to_dict())
    nombres = list(cl["local_cl"]) + list(cl["visita_cl"]) + list(prog["local_cl"]) + list(prog["visita_cl"])
    mapa, sin_cruce = cruzar_nombres(nombres, equipos)
    cl["local"] = cl["local_cl"].map(lambda x: mapa.get(x, x))
    cl["visita"] = cl["visita_cl"].map(lambda x: mapa.get(x, x))
    cl["competicion"] = "CL"
    cl["pais"] = "EUR"
    cl["nivel"] = 0
    cl["neutral"] = cl["fase"].str.endswith("Final") & ~cl["fase"].str.contains("Semi|Quarter")
    prog["local"] = prog["local_cl"].map(lambda x: mapa.get(x, x))
    prog["visita"] = prog["visita_cl"].map(lambda x: mapa.get(x, x))
    prog["competicion"] = "CL"
    return cl.drop(columns=["local_cl", "visita_cl", "fase"]), sin_cruce, prog.drop(columns=["local_cl", "visita_cl", "fase"])


# ---------------------------------------------------------------- todo junto

def cargar_partidos(con_programados=False):
    ligas = cargar_ligas()
    cl, sin_cruce, prog = cargar_champions(ligas)
    partidos = pd.concat([ligas, cl], ignore_index=True)
    partidos["resultado"] = (partidos["gl"] > partidos["gv"]).map({True: "L"}).fillna(
        (partidos["gl"] == partidos["gv"]).map({True: "E", False: "V"}))
    partidos = partidos.sort_values(["fecha", "competicion"], kind="stable").reset_index(drop=True)
    return (partidos, sin_cruce, prog) if con_programados else (partidos, sin_cruce)


def cargar_fixtures(programados_cl=None):
    """Próximos partidos de liga (football-data) + Champions programada (openfootball) si la hay."""
    ruta = config.RAW / "fixtures.csv"
    columnas = ["fecha", "hora", "competicion", "local", "visita", "cuota_l", "cuota_e", "cuota_v", "cuota_o25", "cuota_u25"]
    partes = []
    if ruta.exists():
        df = _leer_csv(ruta)
        df = df[df["Div"].isin(config.LIGAS)].copy()
        ch, cd, ca = _cuotas(df)
        co, cu = _cuotas_ou(df)
        partes.append(pd.DataFrame({
            "fecha": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
            "hora": df.get("Time", ""),
            "competicion": df["Div"],
            "local": df["HomeTeam"].str.strip(),
            "visita": df["AwayTeam"].str.strip(),
            "cuota_l": pd.to_numeric(ch, errors="coerce"),
            "cuota_e": pd.to_numeric(cd, errors="coerce"),
            "cuota_v": pd.to_numeric(ca, errors="coerce"),
            "cuota_o25": co,
            "cuota_u25": cu,
        }))
    if programados_cl is not None and len(programados_cl):
        partes.append(programados_cl.reindex(columns=columnas))
    if not partes:
        return pd.DataFrame(columns=columnas)
    return pd.concat(partes, ignore_index=True).sort_values(["fecha", "hora"]).reset_index(drop=True)


def agregar_liga_actual(tabla, partidos):
    """Agrega a la tabla de Elo la liga actual de cada equipo y si sigue activo."""
    largo = pd.concat([
        partidos[["fecha", "local", "competicion"]].rename(columns={"local": "equipo"}),
        partidos[["fecha", "visita", "competicion"]].rename(columns={"visita": "equipo"}),
    ]).sort_values("fecha", kind="stable")
    liga = largo[largo["competicion"] != "CL"].groupby("equipo")["competicion"].last()
    ultimo = largo.groupby("equipo")["fecha"].max()
    tabla = tabla.copy()
    tabla["liga"] = tabla["equipo"].map(liga).fillna("CL")
    tabla["ultimo_partido"] = tabla["equipo"].map(ultimo)
    tabla["activo"] = tabla["ultimo_partido"] >= partidos["fecha"].max() - pd.Timedelta(days=400)
    return tabla
