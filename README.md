# ⚽ Predictor Fútbol

Predice resultados (quién gana, goles, marcador) de 16 ligas europeas y la Champions.

## Cómo usarlo en tu PC

```bash
pip install -r requirements.txt
python actualizar.py        # baja datos, entrena y evalúa (la 1.ª vez ~15 min)
streamlit run app.py        # abre la app en el navegador
```

Después basta el botón **🔄 Actualizar datos** de la app o `python actualizar.py --rapido` (~3 min).
`python actualizar.py` sin `--rapido` repite la búsqueda de parámetros y el examen completo; hazlo una vez al mes.

## Cómo funciona

1. **Datos**: resultados de 16 ligas desde 2012 (football-data.co.uk, con cuotas y tiros) + Champions League
   (openfootball) + lesionados y valor de planteles (Transfermarkt).
2. **Elo** (fase 1): puntaje por equipo que sube o baja con cada resultado. La Champions conecta las ligas.
3. **Poisson + Dixon-Coles** (fase 2): ataque y defensa de cada equipo → goles esperados → probabilidad de
   cada marcador, más/menos goles, ambos marcan.
4. **Machine learning** (fase 3): gradient boosting que mira Elo, goles esperados, forma, tiros, descanso y
   **las cuotas de las casas** (cuando las hay). Predice 1X2, más de 2.5 y ambos marcan.
5. **Lesionados** (fase 4): el % del valor del plantel que falta se traduce en puntos de Elo menos.
6. **Stacking**: una regresión logística aprende, con temporadas fuera de muestra, cuánto confiar en cada
   modelo y de paso calibra las probabilidades. Eso es lo que muestra la app.
7. **Examen rotativo**: cada temporada desde 2022/23 se predice con modelos que solo vieron las anteriores,
   mes a mes. Se compara con las casas de apuestas.
8. **Historial en vivo** (fase 5): cada actualización registra las predicciones de la semana antes de que se
   jueguen y luego las compara con los resultados (incluye si el ajuste por lesionados ayudó).

## Publicar (gratis)

- **GitHub Actions** (`.github/workflows/actualizar.yml`) corre `actualizar.py --rapido` todos los días y
  guarda los resultados en `data/`.
- **Streamlit Community Cloud** sirve la app desde el repo: share.streamlit.io → New app → este repo,
  rama `main`, archivo `app.py`, Python 3.13.

## Estructura

```
actualizar.py          pipeline completo
app.py                 interfaz Streamlit
src/config.py          ligas, temporadas, parámetros
src/datos.py           descarga y limpieza, cruce de nombres Champions <-> ligas, próximos partidos
src/elo.py             cálculo del Elo
src/modelo.py          Elo -> probabilidades 1X2 (logit ordenado)
src/poisson.py         modelo de goles Poisson + Dixon-Coles
src/features.py        variables del ML (forma, tiros, descanso, cuotas…), siempre previas al partido
src/ml.py              modelos de machine learning
src/bajas.py           lesionados de Transfermarkt y su impacto (cruces en ALIAS_TM)
src/evaluacion.py      métricas, stacking y examen rotativo
src/registro.py        registro de predicciones y comparación con resultados
src/prediccion.py      junta todo para un partido (lo usa la app)
herramientas/          scripts de revisión
data/                  resultados del pipeline (los CSV crudos y cachés no van al repo)
```

## Ideas pendientes

- xG de Understat (5 grandes ligas) como variable del ML.
- Suspendidos (Transfermarkt cortó esas páginas).
- Alineaciones confirmadas / rating por jugador (necesita datos de pago).
