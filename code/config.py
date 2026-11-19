"""Configuración central del proyecto Simulaciones Premier."""
import os

RUTA_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA      = os.path.join(RUTA_RAIZ, 'Data')
RAW       = os.path.join(DATA, 'raw')
PRED      = os.path.join(RUTA_RAIZ, 'Predicciones')
MODELOS   = os.path.join(RUTA_RAIZ, 'modelos')
DOCS      = os.path.join(RUTA_RAIZ, 'docs')

# ── Temporadas ────────────────────────────────────────────────
TEMPORADA_ACTUAL = '2627'                     # 2026-27
PRIMERA_TEMPORADA = 2005                      # 2005-06 en adelante
def temporadas(desde=PRIMERA_TEMPORADA, hasta=int('20' + TEMPORADA_ACTUAL[:2])):
    """Códigos football-data: 2005 -> '0506'."""
    return [f'{str(a)[2:]}{str(a + 1)[2:]}' for a in range(desde, hasta + 1)]

DIVISIONES = {'E0': 'Premier League', 'E1': 'Championship'}
URL_FD       = 'https://www.football-data.co.uk/mmz4281/{temp}/{div}.csv'
URL_FIXTURES = 'https://www.football-data.co.uk/fixtures.csv'

# ── Archivos ──────────────────────────────────────────────────
RUTA_PARTIDOS   = os.path.join(DATA, 'partidos.csv')          # unificado y limpio
RUTA_FLASHSCORE = os.path.join(DATA, 'flashscore_premier.csv')  # opcional (Scrapper)
RUTA_HISTORICO  = os.path.join(DATA, 'datos_historicos.csv')  # dataset modelable
RUTA_ESTADO     = os.path.join(DATA, 'estado_equipos.csv')    # foto actual por equipo
RUTA_PROXIMOS   = os.path.join(DATA, 'proximos_partidos.csv')

# ── Feature engineering ───────────────────────────────────────
VENTANAS   = [3, 5, 10]
ELO_K      = 20
ELO_HFA    = 60          # ventaja de local en puntos Elo
ELO_INICIAL = 1500

# ── Modelo / simulación ───────────────────────────────────────
TWEEDIE_POWER   = 1.2
N_SIMULACIONES  = 10_000
SIGMA_FUERZA    = 0.15   # sd del "shock" de fuerza por equipo y simulación (escala logit 1 vs 2)
DECAY_RECENCIA  = 0.0007 # peso muestral exp(-decay * días de antigüedad)
SEMILLA         = 42

# Zonas de la tabla (Premier 2026-27)
ZONAS = {
    'Campeón':   [1],
    'Top 4':     [1, 2, 3, 4],
    'Top 5 (UCL)': [1, 2, 3, 4, 5],
    'Top 7 (Europa)': list(range(1, 8)),
    'Descenso':  [18, 19, 20],
}

for _d in (DATA, RAW, PRED, MODELOS, DOCS):
    os.makedirs(_d, exist_ok=True)
