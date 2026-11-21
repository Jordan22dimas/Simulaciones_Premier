# Simulaciones Premier League 2026‑27 ⚽🦁

¿Quién va a ganar la Premier? ¿Quién baja? Este repositorio responde con datos: descarga más de 20 temporadas de Premier y Championship, construye el *estado de forma* de cada equipo antes de cada partido, entrena modelos XGBoost y simula **10.000 temporadas** con Monte Carlo a partir de la clasificación real. Es la evolución del proyecto [Simulaciones Mundial](../Simulaciones_Mundial/), adaptado a una liga.

---

## 🔄 El pipeline, paso a paso

Todo el código está en [`code/`](code/). Si ejecutas `python code/Pipeline.py`, corre de principio a fin en unos 90 segundos.

### 1️⃣ Descarga — [`code/Descarga_Datos.py`](code/Descarga_Datos.py)
Descarga los CSV de **football-data.co.uk** de Premier (E0) y Championship (E1) desde 2005‑06: resultados, remates, remates a puerta, córners, faltas, tarjetas, árbitro, **xG** (disponible desde 2026‑27) y **cuotas** de las casas de apuestas. No hace falta scraping. Las temporadas cerradas se cachean en [`Data/raw/`](Data/raw/).

### 1️⃣b (Opcional) FlashScore — [`code/Scrapper_FlashScore.py`](code/Scrapper_FlashScore.py)
Es el scraper Selenium del Mundial adaptado a la Premier: recoge el xG real, la posesión, los pases y las grandes ocasiones de temporadas pasadas. Es incremental. La limpieza integra automáticamente el xG real en los partidos que lo tengan. Ya están descargadas las temporadas 2021‑22 a 2026‑27 (2.000 partidos). FlashScore solo publica xG desde mediados de la 2022‑23, así que 1.389 de ellos aportan xG real.

### 2️⃣ Limpieza y variables — [`code/Data_Cleaning.py`](code/Data_Cleaning.py) + [`code/features.py`](code/features.py)
* **xG estimado** para las temporadas sin xG: `xG ≈ 0,19·remates_a_puerta + 0,07·remates_fuera`, ajustado sobre los partidos que sí tienen xG real.
* **Elo propio** en una sola escala Premier + Championship. Los ascendidos heredan el Elo medio de los descendidos y todos los equipos regresan un 20 % a la media en cada verano. Sustituye al ranking FIFA del Mundial.
* **Medias móviles sin fuga** (`shift()`) a 3, 5 y 10 partidos, más una media exponencial: goles, xG a favor y en contra, remates, remates a puerta, córners, tarjetas y puntos.
* **Forma de la temporada en curso**, forma **en casa y fuera**, días de descanso y `forma_vs_historia`.
* **% de partidos recientes en Championship**, para que el modelo aprenda a descontar las estadísticas infladas de los ascendidos.
* **H2H** de los últimos 6 enfrentamientos y **variables diferencia** Local − Visitante.
* **Filas centinela**, como en el Mundial: guardan la *foto actual* de cada equipo ([`Data/estado_equipos.csv`](Data/estado_equipos.csv)) para predecir partidos futuros con exactamente las mismas columnas.

### 3️⃣ Modelos — [`code/Modelo.py`](code/Modelo.py)
1. **Modelo de goles**: dos XGBoost Tweedie, uno para el local y otro para el visitante. Se ponderan por recencia y los hiperparámetros se buscan con validación temporal (`TimeSeriesSplit`).
2. **Matriz de marcadores** Poisson con corrección **Dixon‑Coles** (ρ ajustado), que corrige la infraestimación de 0‑0 y 1‑1.
3. **Clasificador 1X2** XGBoost con los goles esperados como meta‑variables (fuera de muestra) y **calibración isotónica**.
4. Se elige automáticamente la combinación con menor log‑loss en test.

**Validación** en 2023‑24, 2024‑25 y 2025‑26 (1.140 partidos nunca vistos):

| Modelo | Log‑loss | RPS | Acierto |
|---|---|---|---|
| **Poisson + Dixon‑Coles (producción)** | **0,9840** | **0,2013** | **53,0 %** |
| Clasificador calibrado | 0,9920 | 0,2022 | 53,3 % |
| Elo puro | 0,9842 | 0,2014 | 53,0 % |
| Mercado (cuotas medias) | 0,9650 | 0,1955 | 54,2 % |
| Base (frecuencias) | 1,0723 | 0,2321 | 43,2 % |

> Las casas de apuestas siguen por delante, algo esperable porque incorporan alineaciones, lesiones y dinero informado. Las cuotas se usan **solo como benchmark** y no como variable: así el modelo puede predecir partidos de dentro de 5 meses, para los que todavía no hay cuotas.

### 4️⃣ Monte Carlo — [`code/Simulacion_Temporada.py`](code/Simulacion_Temporada.py)
Parte de la **clasificación real** y simula 10.000 veces los partidos pendientes:
* En cada temporada simulada, cada equipo recibe un **shock de fuerza** aleatorio que dura toda la temporada (lesiones, fichajes, rachas). Sin él, el Monte Carlo es sobreconfiado.
* Primero se sortea el resultado 1X2 con las probabilidades del modelo. Después se sortea el marcador de la matriz Dixon‑Coles condicionada a ese resultado, para tener la diferencia de goles.
* Desempates: puntos → DG → GF.

### 5️⃣ Próxima jornada — [`code/Prediccion_Jornada.py`](code/Prediccion_Jornada.py)
Para cada partido calcula el 1X2, los goles esperados, el marcador más probable, la probabilidad de +2,5 goles y de que marquen ambos, y la **diferencia frente al mercado**: en el 1X2, en el más de 2,5 goles y en los goles esperados implícitos en las cuotas.

### 6️⃣ Backtest de la temporada en curso — [`code/Backtest_Temporada.py`](code/Backtest_Temporada.py)
Responde a la pregunta *¿qué habría predicho el modelo en cada jornada ya jugada?* Entrena solo con temporadas anteriores y predice cada partido con la información disponible justo antes de jugarse. Después compara los goles esperados con los goles reales, con el xG real y con los **goles implícitos en las cuotas** (1X2 + más/menos de 2,5). Con `--walk-forward` se reentrena antes de cada jornada, igual que el robot diario.

Jornadas 1 a 10 de la 2026‑27 (100 partidos, modelo congelado de pretemporada):

| | Modelo | Mercado | Media de la liga |
|---|---|---|---|
| Goles totales (reales: 273) | **282** | 294 | — |
| Error medio de goles por equipo (MAE) | **0,940** | 0,947 | 0,985 |
| Error frente al xG real | **0,614** | 0,619 | — |
| Acierto 1X2 | 45 % | **48 %** | — |
| Log‑loss 1X2 | 1,019 | **1,012** | — |

En goles, el modelo **empata con las casas de apuestas**. En el 1X2 el mercado sigue un poco por delante.

### 🌐 Web diaria — [`docs/`](docs/)
[`code/Generar_Web.py`](code/Generar_Web.py) genera una página estática (sin dependencias, con modo claro y oscuro) con las probabilidades de la temporada, un mapa de calor de posiciones, la jornada, la clasificación y las métricas. El workflow [`.github/workflows/actualizar_web.yml`](.github/workflows/actualizar_web.yml) la regenera **cada día a las 07:00 UTC**.

Para activarla: **Settings → Pages → Source: Deploy from a branch → `main` / `docs/`**, y en **Settings → Actions → General → Workflow permissions** marca *Read and write*.

---

## 📊 Predicción actual (tras la jornada 10)

| # | Equipo | Pts | Pts esperados (90 %) | Campeón | Top 4 | Top 5 | Descenso |
|---|---|---|---|---|---|---|---|
| 1 | Arsenal | 27 | 82.0 (69–94) | 60 % | 98 % | 99 % | 0 % |
| 2 | Man City | 21 | 75.2 (63–87) | 21 % | 91 % | 96 % | 0 % |
| 3 | Brighton | 22 | 73.6 (61–86) | 16 % | 88 % | 95 % | 0 % |
| 4 | Liverpool | 18 | 64.7 (52–77) | 2 % | 48 % | 66 % | <1 % |
| 5 | Bournemouth | 15 | 63.5 (51–76) | 1 % | 41 % | 62 % | <1 % |
| 6 | Chelsea | 16 | 54.8 (42–68) | <1 % | 9 % | 18 % | 1 % |
| 7 | Everton | 16 | 52.3 (40–65) | <1 % | 5 % | 12 % | 2 % |
| 8 | Leeds | 14 | 51.9 (39–65) | <1 % | 4 % | 10 % | 3 % |
| 9 | Man United | 8 | 51.4 (38–64) | <1 % | 4 % | 10 % | 3 % |
| 10 | Brentford | 14 | 50.4 (38–63) | 0 % | 3 % | 8 % | 4 % |
| 11 | Tottenham | 12 | 49.9 (37–63) | <1 % | 3 % | 7 % | 4 % |
| 12 | Aston Villa | 13 | 49.8 (37–63) | <1 % | 3 % | 7 % | 5 % |
| 13 | Nott'm Forest | 9 | 48.6 (36–61) | 0 % | 2 % | 5 % | 6 % |
| 14 | Fulham | 10 | 45.2 (33–58) | <1 % | 1 % | 2 % | 12 % |
| 15 | Newcastle | 12 | 43.0 (31–56) | 0 % | <1 % | 1 % | 20 % |
| 16 | Sunderland | 8 | 42.5 (30–55) | 0 % | <1 % | 1 % | 23 % |
| 17 | Ipswich | 11 | 38.9 (27–51) | 0 % | <1 % | <1 % | 39 % |
| 18 | Hull | 11 | 38.3 (27–51) | 0 % | <1 % | <1 % | 42 % |
| 19 | Coventry | 9 | 35.7 (24–48) | 0 % | 0 % | <1 % | 58 % |
| 20 | Crystal Palace | 7 | 31.4 (20–44) | 0 % | 0 % | <1 % | 79 % |

Los resultados completos están en [`Predicciones/`](Predicciones/) y se actualizan a diario.

---

## 📂 Estructura

```
├── code/
│   ├── config.py                # Rutas, temporadas y parámetros
│   ├── Descarga_Datos.py        # football-data.co.uk → Data/partidos.csv
│   ├── Scrapper_FlashScore.py   # (opcional) stats avanzadas con Selenium
│   ├── Data_Cleaning.py         # Elo, xG, medias móviles, H2H → dataset modelable
│   ├── features.py              # Ingeniería de variables compartida (train = predicción)
│   ├── nombres.py               # Mapeo de nombres FlashScore ↔ football-data
│   ├── Modelo.py                # XGBoost Tweedie + Dixon-Coles + 1X2 calibrado
│   ├── Prediccion_Jornada.py    # Próximos partidos vs mercado
│   ├── Simulacion_Temporada.py  # Monte Carlo de 10.000 temporadas
│   ├── Backtest_Temporada.py    # Jornadas jugadas: modelo vs realidad vs mercado
│   ├── Generar_Web.py           # docs/index.html
│   └── Pipeline.py              # Todo de principio a fin
├── Data/                        # Datos descargados y procesados
├── Predicciones/                # Salidas: simulación, posiciones, jornada, métricas
├── docs/                        # Web (GitHub Pages)
└── .github/workflows/           # Actualización diaria
```

## 🚀 Replicarlo

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python code/Pipeline.py               # diario: reutiliza hiperparámetros (~1,5 min)
python code/Pipeline.py --completo    # con búsqueda de hiperparámetros y validación (~8 min)
```

## 💡 Siguientes pasos

* Separar el **xG real** y el **xG estimado** en variables distintas. Ahora se mezclan, y la validación empeora un poco (log‑loss de 0,9805 a 0,9840) porque el entrenamiento usa casi siempre xG estimado.
* Añadir **valor de mercado** de la plantilla (Transfermarkt), como en el Mundial.
* Modelos de **córners y tarjetas** (los datos ya están descargados, incluido el árbitro).
* Actualizar el Elo **dentro** de cada temporada simulada para que la forma evolucione a lo largo del Monte Carlo.
* Añadir otras ligas cambiando el par `E0`/`E1` por `SP1`/`SP2` (LaLiga), `I1`/`I2`, `D1`/`D2` o `F1`/`F2` en `config.py`, y ajustando las zonas y el número de equipos.

*Proyecto educativo; no es consejo de apuestas.*
