"""Fase 4: lesionados desde Transfermarkt y su impacto en cada partido.

Analogía: si a un equipo le faltan jugadores que valen el 20% de su plantel, es
como si ese día el equipo "costara" menos. Con los ~300 clubes vemos cuánto Elo
tiene en promedio un equipo según lo que vale su plantel, y así traducimos las
bajas a puntos de Elo. Como el que entra de reemplazo no es un cero, solo cuenta
una parte de lo que falta (FACTOR_REEMPLAZO).

Ojo: Transfermarkt no tiene una API oficial. Si cambian su web, hay que ajustar
los lectores de abajo. Se piden pocas páginas y con pausas para no molestar.
"""
import difflib
import html as html_lib
import re
import time

import numpy as np
import pandas as pd
import requests

from . import config
from .datos import _norm

# código football-data -> código Transfermarkt
LIGAS_TM = {"E0": "GB1", "E1": "GB2", "SP1": "ES1", "SP2": "ES2", "D1": "L1", "D2": "L2",
            "I1": "IT1", "I2": "IT2", "F1": "FR1", "F2": "FR2", "P1": "PO1", "N1": "NL1",
            "B1": "BE1", "T1": "TR1", "SC0": "SC1", "G1": "GR1"}
URL = "https://www.transfermarkt.com/x/{pagina}/wettbewerb/{codigo}"
CABECERAS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
             "Accept-Language": "en"}
PAUSA = 2.0             # segundos entre páginas
FACTOR_REEMPLAZO = 0.5  # qué parte del valor de las bajas se pierde de verdad (el suplente aporta el resto)
MAX_PCT = 0.6           # tope: nunca se considera que falta más del 60% del plantel

RUTA_BAJAS_TM = config.DATA / "bajas_tm.csv"
RUTA_PLANTELES_TM = config.DATA / "planteles_tm.csv"
RUTA_FECHA = config.DATA / "bajas_fecha.txt"

# Nombre Transfermarkt -> nombre football-data, cuando el cruce automático no basta
ALIAS_TM = {
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Queens Park Rangers": "QPR",
    "Borussia Mönchengladbach": "M'gladbach",
    "Stade Rennais FC": "Rennes",
    "Paris Saint-Germain": "Paris SG",
    "Sporting CP": "Sp Lisbon",
    "Heart of Midlothian FC": "Hearts",
    "Deportivo A Coruña": "La Coruna",
    "Inter Milan": "Inter",
    "RCD Espanyol Barcelona": "Espanol",
}


# ---------------------------------------------------------------- lectura de páginas

def _get(url):
    for intento in range(3):
        try:
            r = requests.get(url, headers=CABECERAS, timeout=30)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(5 * (intento + 1))
    return None


def _texto(fragmento):
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", fragmento)).strip()


def _valor(texto):
    """'€45.00m' -> 45000000 ; '€500k' -> 500000 ; '€1.43bn' -> 1430000000."""
    m = re.search(r"€\s*([\d.,]+)\s*(bn|m|k|Th\.)?", texto)
    if not m:
        return np.nan
    return float(m.group(1).replace(",", "")) * {"bn": 1e9, "m": 1e6, "k": 1e3, "Th.": 1e3}.get(m.group(2) or "", 1)


def _filas(pagina):
    """Filas de la tabla principal (class="items")."""
    i = pagina.find('class="items"')
    if i < 0:
        return []
    ini = pagina.find("<tbody>", i)
    fin = pagina.find("</tbody>", ini)
    return re.split(r'<tr class="(?:odd|even)">', pagina[ini:fin])[1:]


def parse_lesionados(pagina):
    salida = []
    for fila in _filas(pagina):
        jugador = re.search(r'class="hauptlink">\s*<a title="([^"]+)"', fila)
        club = re.search(r'<a title="([^"]+)" href="/[^"]+/startseite/verein/(\d+)', fila)
        if not (jugador and club):
            continue
        posicion = re.search(r"<tr>\s*<td>([^<]+)</td>\s*</tr>", fila)
        lesion = re.search(r'<td class="links">(.*?)</td>', fila, re.S)
        hasta = re.search(r'<td class="zentriert">(.*?)</td>', fila, re.S)
        valor = re.search(r'<td class="rechts">(.*?)</td>', fila, re.S)
        salida.append({
            "jugador": html_lib.unescape(jugador.group(1)),
            "posicion": posicion.group(1).strip() if posicion else "",
            "club_tm": html_lib.unescape(club.group(1)),
            "club_id": int(club.group(2)),
            "lesion": _texto(lesion.group(1)) if lesion else "",
            "hasta_txt": _texto(hasta.group(1)) if hasta else "",
            "valor": _valor(_texto(valor.group(1))) if valor else np.nan,
        })
    return salida


def parse_planteles(pagina):
    salida = []
    for fila in _filas(pagina):
        club = re.search(r'class="hauptlink no-border-links"><a title="([^"]+)" href="/[^"]+/startseite/verein/(\d+)',
                         fila)
        rechts = re.findall(r'<td class="rechts">(.*?)</td>', fila, re.S)
        if club and rechts:
            salida.append({"club_tm": html_lib.unescape(club.group(1)), "club_id": int(club.group(2)),
                           "valor_plantel": _valor(_texto(rechts[-1]))})
    return salida


def descargar():
    """Baja lesionados y valor de planteles de las 16 ligas. Si una falla, se conserva la versión anterior."""
    lesionados, planteles, fallidas = [], [], []
    for div, codigo in LIGAS_TM.items():
        p_les = _get(URL.format(pagina="verletztespieler", codigo=codigo))
        time.sleep(PAUSA)
        p_pla = _get(URL.format(pagina="startseite", codigo=codigo))
        time.sleep(PAUSA)
        les = parse_lesionados(p_les) if p_les else []
        pla = parse_planteles(p_pla) if p_pla else []
        if not pla:
            fallidas.append(div)
            continue
        lesionados += [{**x, "div": div} for x in les]
        planteles += [{**x, "div": div} for x in pla]

    les, pla = pd.DataFrame(lesionados), pd.DataFrame(planteles)
    if fallidas:
        print(f"   ! No se pudo leer {fallidas}; se usan sus datos anteriores")
        if RUTA_PLANTELES_TM.exists():
            les = pd.concat([les, pd.read_csv(RUTA_BAJAS_TM).query("div in @fallidas")], ignore_index=True)
            pla = pd.concat([pla, pd.read_csv(RUTA_PLANTELES_TM).query("div in @fallidas")], ignore_index=True)
    if pla.empty:
        return False
    les.to_csv(RUTA_BAJAS_TM, index=False)
    pla.to_csv(RUTA_PLANTELES_TM, index=False)
    RUTA_FECHA.write_text(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    return True


# ---------------------------------------------------------------- cruce de nombres

def cruzar(planteles, partidos):
    """Devuelve {club_id: nombre football-data} y la lista de clubes sin cruzar."""
    actual = partidos[partidos["temporada"] == partidos["temporada"].max()]
    mapa, puntajes, sin_cruce = {}, {}, []
    for div, grupo in planteles.groupby("div"):
        de_liga = actual[actual["competicion"] == div]
        candidatos = set(de_liga["local"]) | set(de_liga["visita"])
        for club, club_id in zip(grupo["club_tm"], grupo["club_id"]):
            if club in ALIAS_TM:
                mapa[club_id], puntajes[club_id] = ALIAS_TM[club], 2.0
                continue
            n = _norm(club)
            mejor, puntaje = None, 0.0
            for c in candidatos:
                nc = _norm(c)
                p = difflib.SequenceMatcher(None, n, nc).ratio()
                if nc and (nc in n or n in nc):
                    p = max(p, 0.9)
                if p > puntaje:
                    mejor, puntaje = c, p
            if puntaje >= 0.6:
                mapa[club_id], puntajes[club_id] = mejor, puntaje
            else:
                sin_cruce.append((div, club, mejor, round(puntaje, 2)))

    # uno a uno: si dos clubes cayeron en el mismo equipo (ej. "Paris FC" y "PSG"), se queda el más parecido
    nombre_tm = dict(zip(planteles["club_id"], planteles["club_tm"]))
    por_equipo = {}
    for club_id, equipo in mapa.items():
        por_equipo.setdefault(equipo, []).append(club_id)
    for equipo, ids in por_equipo.items():
        for club_id in sorted(ids, key=lambda i: puntajes[i], reverse=True)[1:]:
            del mapa[club_id]
            sin_cruce.append(("repetido", nombre_tm[club_id], equipo, round(puntajes[club_id], 2)))
    return mapa, sin_cruce


def cargar(partidos):
    """Lesionados y planteles con el nombre de equipo de football-data."""
    les = pd.read_csv(RUTA_BAJAS_TM)
    pla = pd.read_csv(RUTA_PLANTELES_TM)
    mapa, sin_cruce = cruzar(pla, partidos)
    les["equipo"] = les["club_id"].map(mapa)
    pla["equipo"] = pla["club_id"].map(mapa)
    # Transfermarkt escribe día/mes/año ("06/10/2026" = 6 de octubre)
    les["hasta"] = pd.to_datetime(les["hasta_txt"], errors="coerce", format="%d/%m/%Y")
    return les.dropna(subset=["equipo"]), pla.dropna(subset=["equipo"]), sin_cruce


# ---------------------------------------------------------------- impacto

def calibrar(tabla_elo, planteles, partidos_con_lam):
    """Dos números sacados de nuestros datos:
    - cuánto Elo de más tiene un equipo cuando su plantel vale e veces más (≈2.7x)
    - cuánto cambian los goles esperados por cada punto de Elo de diferencia
    """
    d = planteles.merge(tabla_elo[["equipo", "elo"]], on="equipo").query("valor_plantel > 0")
    elo_por_log_valor = float(np.polyfit(np.log(d["valor_plantel"]), d["elo"], 1)[0])
    h = partidos_con_lam.dropna(subset=["lam_l"])
    h = h[h["temporada"] >= config.temporadas()[-4]]
    goles_por_elo = float(np.polyfit(h["elo_l"] - h["elo_v"], np.log(h["lam_l"] / h["lam_v"]), 1)[0])
    return {"elo_por_log_valor": elo_por_log_valor, "log_goles_por_elo": goles_por_elo,
            "factor_reemplazo": FACTOR_REEMPLAZO, "clubes": int(len(d))}


def impacto(lesionados_equipo, valor_plantel, cal, fecha=None):
    """Qué % del plantel falta para `fecha` y cuántos puntos de Elo le resta."""
    fecha = pd.Timestamp(fecha) if fecha is not None else pd.Timestamp.today().normalize()
    activos = lesionados_equipo[lesionados_equipo["hasta"].isna() | (lesionados_equipo["hasta"] > fecha)]
    valor = float(activos["valor"].fillna(0).sum())
    pct = min(MAX_PCT, valor / valor_plantel) if valor_plantel > 0 else 0.0
    delta = cal["elo_por_log_valor"] * np.log(1 - cal["factor_reemplazo"] * pct)
    return {"pct": pct, "delta_elo": float(delta), "valor": valor,
            "jugadores": activos.sort_values("valor", ascending=False)}
