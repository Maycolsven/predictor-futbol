"""Configuración general: ligas, temporadas, rutas y parámetros del Elo."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
RAW = DATA / "raw"

# Código de football-data.co.uk -> (nombre, país, división)
LIGAS = {
    "E0": ("Premier League", "ENG", 1),
    "E1": ("Championship", "ENG", 2),
    "SP1": ("LaLiga", "ESP", 1),
    "SP2": ("LaLiga 2", "ESP", 2),
    "D1": ("Bundesliga", "GER", 1),
    "D2": ("2. Bundesliga", "GER", 2),
    "I1": ("Serie A", "ITA", 1),
    "I2": ("Serie B", "ITA", 2),
    "F1": ("Ligue 1", "FRA", 1),
    "F2": ("Ligue 2", "FRA", 2),
    "P1": ("Primeira Liga", "POR", 1),
    "N1": ("Eredivisie", "NED", 1),
    "B1": ("Pro League", "BEL", 1),
    "T1": ("Süper Lig", "TUR", 1),
    "SC0": ("Premiership", "SCO", 1),
    "G1": ("Super League", "GRE", 1),
}
LIGAS_PRINCIPALES = [c for c, (_, _, div) in LIGAS.items() if div == 1]

# Temporadas en formato football-data ("1213" = 2012/13). Las primeras sirven
# de "calentamiento" para que el Elo se estabilice antes de evaluar.
PRIMERA_TEMPORADA = 2012
TEMPORADA_ACTUAL = 2026  # 2026/27

# Cómo se evalúa sin hacer trampa:
# - las 2 primeras temporadas solo sirven para que el Elo se estabilice
# - la de VALIDACIÓN se usa para elegir parámetros (K, vida media, peso de cada modelo…)
# - la de PRUEBA es el examen final: nada se ajusta mirándola
TEMPORADAS_CALENTAMIENTO = 2
TEMPORADA_VALIDACION = "2425"
TEMPORADA_PRUEBA = "2526"
# Examen final rotativo: cada temporada se predice con modelos entrenados solo con las anteriores.
# La primera sirve para entrenar el "stacking" (el que combina los modelos) de la siguiente.
TEMPORADAS_EXAMEN = ["2122", "2223", "2324", "2425", "2526"]
DIAS_REGISTRO = 7  # se registra la predicción de los partidos de los próximos N días

# Elo inicial de un equipo según su país (primera división). Solo es el punto
# de partida: después de unas temporadas el Elo se ajusta con los resultados y
# con los cruces de Champions.
ELO_BASE_PAIS = {
    "ENG": 1600, "ESP": 1580, "GER": 1560, "ITA": 1560, "FRA": 1520,
    "POR": 1480, "NED": 1470, "BEL": 1440, "TUR": 1430, "GRE": 1400,
    "SCO": 1380,
}
ELO_BASE_OTROS = 1400       # equipos de países sin liga descargada (solo Champions)
PENALIZACION_SEGUNDA = 180  # un equipo nuevo de 2.ª división arranca más abajo

# Parámetros del Elo (se afinan en modelo.py con búsqueda en grilla)
K = 20
VENTAJA_LOCAL = 65


def temporadas():
    """Lista de códigos de temporada: ['1213', '1314', ..., '2627']."""
    return [f"{a % 100:02d}{(a + 1) % 100:02d}" for a in range(PRIMERA_TEMPORADA, TEMPORADA_ACTUAL + 1)]
