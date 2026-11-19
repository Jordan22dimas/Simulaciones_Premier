"""
3. MODELADO
  · 2 regresores XGBoost Tweedie → goles esperados local / visitante
  · Matriz de marcadores Poisson con corrección Dixon-Coles (rho) → 1X2 "estructural"
  · Clasificador XGBoost 1X2 (con los goles esperados como meta-variables) calibrado isotónicamente
  · Ensemble elegido por log-loss en test y comparado contra las cuotas del mercado
Uso:  python code/Modelo.py            → búsqueda de hiperparámetros + validación + reentreno (~7 min)
      python code/Modelo.py --rapido   → reutiliza hiperparámetros y modelo elegido; solo reentrena (~1 min)
      from Modelo import predecir      → probabilidades para cualquier lista de partidos
"""
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import KFold, TimeSeriesSplit, ParameterSampler
import config as C
import features as F

MAX_GOLES = 10
TEMPORADAS_TEST = ['2324', '2425', '2526']
N_ITER_BUSQUEDA = 12

PARAMS_REG = dict(objective='reg:tweedie', tweedie_variance_power=C.TWEEDIE_POWER,
                  n_estimators=500, learning_rate=0.02, max_depth=3, subsample=0.8,
                  colsample_bytree=0.5, min_child_weight=20, reg_lambda=5.0, n_jobs=-1,
                  random_state=C.SEMILLA)
ESPACIO_REG = {'n_estimators': [300, 500, 800], 'learning_rate': [0.01, 0.02, 0.04],
               'max_depth': [2, 3, 4], 'min_child_weight': [10, 20, 50],
               'colsample_bytree': [0.3, 0.5, 0.7], 'subsample': [0.7, 0.8, 0.9]}
PARAMS_CLF = dict(objective='multi:softprob', num_class=3, n_estimators=300, learning_rate=0.02,
                  max_depth=2, subsample=0.8, colsample_bytree=0.5, min_child_weight=30,
                  reg_lambda=5.0, n_jobs=-1, random_state=C.SEMILLA)
CLASES = ['1', 'X', '2']


# ─────────────────────────────────────────────────────────────
# Utilidades de probabilidad
# ─────────────────────────────────────────────────────────────
def matriz_marcadores(lam_l, lam_v, rho=0.0):
    """(n, G, G) probabilidades de marcador con ajuste Dixon-Coles en 0-0, 1-0, 0-1, 1-1."""
    g = np.arange(MAX_GOLES + 1)
    pl = poisson.pmf(g[None, :], np.asarray(lam_l)[:, None])
    pv = poisson.pmf(g[None, :], np.asarray(lam_v)[:, None])
    M = pl[:, :, None] * pv[:, None, :]
    ll, lv = np.asarray(lam_l), np.asarray(lam_v)
    M[:, 0, 0] *= 1 - ll * lv * rho
    M[:, 1, 0] *= 1 + lv * rho
    M[:, 0, 1] *= 1 + ll * rho
    M[:, 1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum(axis=(1, 2), keepdims=True)


def probs_1x2_desde_matriz(M):
    tri_inf = np.tril(np.ones(M.shape[1:]), -1).astype(bool)
    p1 = M[:, tri_inf].sum(1)
    px = np.trace(M, axis1=1, axis2=2)
    return np.c_[p1, px, 1 - p1 - px]


def rps(probs, y):
    """Ranked Probability Score (métrica estándar en fútbol; menor es mejor)."""
    obs = np.eye(3)[y]
    cp, co = np.cumsum(probs, 1), np.cumsum(obs, 1)
    return float(np.mean(np.sum((cp - co)[:, :2] ** 2, 1) / 2))


def metricas(probs, y):
    probs = np.clip(probs, 1e-6, 1)
    probs = probs / probs.sum(1, keepdims=True)
    return {'log_loss': round(log_loss(y, probs, labels=[0, 1, 2]), 4),
            'rps': round(rps(probs, y), 4),
            'brier': round(float(np.mean([brier_score_loss(y == k, probs[:, k]) for k in range(3)])), 4),
            'accuracy': round(float((probs.argmax(1) == y).mean()), 4)}


def pesos_recencia(fechas):
    dias = (fechas.max() - fechas).dt.days.values
    return np.exp(-C.DECAY_RECENCIA * dias)


# ─────────────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────────────
def buscar_hiperparametros(X, y, w):
    """Búsqueda aleatoria con validación temporal (TimeSeriesSplit) y deviance Tweedie/Poisson."""
    tss = TimeSeriesSplit(n_splits=4)
    mejor, mejor_score = None, np.inf
    for p in ParameterSampler(ESPACIO_REG, n_iter=N_ITER_BUSQUEDA, random_state=C.SEMILLA):
        scores = []
        for tr, va in tss.split(X):
            m = xgb.XGBRegressor(**{**PARAMS_REG, **p}).fit(X.iloc[tr], y.iloc[tr], sample_weight=w[tr])
            mu = np.clip(m.predict(X.iloc[va]), 1e-3, None)
            scores.append(np.mean(mu - y.iloc[va].values * np.log(mu)))   # deviance Poisson
        if np.mean(scores) < mejor_score:
            mejor, mejor_score = p, np.mean(scores)
    return {**PARAMS_REG, **mejor}


def ajustar_rho(lam_l, lam_v, y):
    rejilla = np.linspace(-0.25, 0.10, 36)
    ll = [log_loss(y, probs_1x2_desde_matriz(matriz_marcadores(lam_l, lam_v, r)), labels=[0, 1, 2]) for r in rejilla]
    return float(rejilla[int(np.argmin(ll))])


def con_meta(X, lam_l, lam_v):
    meta = pd.DataFrame({'meta_lam_L': lam_l, 'meta_lam_V': lam_v, 'meta_lam_diff': lam_l - lam_v}, index=X.index)
    return pd.concat([X, meta], axis=1)


def entrenar(df, params_L, params_V, cols):
    """Entrena el sistema completo sobre df y devuelve un dict de modelos."""
    X, w = df[cols], pesos_recencia(df['Fecha'])
    y1x2 = df['Resultado_1X2'].map({c: i for i, c in enumerate(CLASES)}).values
    reg_L = xgb.XGBRegressor(**params_L).fit(X, df['FTHG'], sample_weight=w)
    reg_V = xgb.XGBRegressor(**params_V).fit(X, df['FTAG'], sample_weight=w)

    # Meta-variables fuera de muestra para el clasificador (evita fuga)
    oof = np.zeros((len(df), 2))
    for tr, va in KFold(5, shuffle=True, random_state=C.SEMILLA).split(X):
        oof[va, 0] = xgb.XGBRegressor(**params_L).fit(X.iloc[tr], df['FTHG'].iloc[tr], sample_weight=w[tr]).predict(X.iloc[va])
        oof[va, 1] = xgb.XGBRegressor(**params_V).fit(X.iloc[tr], df['FTAG'].iloc[tr], sample_weight=w[tr]).predict(X.iloc[va])
    rho = ajustar_rho(oof[:, 0], oof[:, 1], y1x2)

    Xm = con_meta(X, oof[:, 0], oof[:, 1])
    clf = CalibratedClassifierCV(xgb.XGBClassifier(**PARAMS_CLF), method='isotonic', cv=5)
    clf.fit(Xm, y1x2, sample_weight=w)
    return {'reg_L': reg_L, 'reg_V': reg_V, 'clf': clf, 'rho': rho, 'cols': cols, 'peso_clf': 0.5}


def aplicar(modelos, X):
    lam_l = np.clip(modelos['reg_L'].predict(X[modelos['cols']]), 0.05, None)
    lam_v = np.clip(modelos['reg_V'].predict(X[modelos['cols']]), 0.05, None)
    M = matriz_marcadores(lam_l, lam_v, modelos['rho'])
    p_pois = probs_1x2_desde_matriz(M)
    Xm = con_meta(X[modelos['cols']], lam_l, lam_v)
    p_clf = modelos['clf'].predict_proba(Xm)
    w = modelos['peso_clf']
    return {'lam_L': lam_l, 'lam_V': lam_v, 'M': M, 'p_poisson': p_pois, 'p_clf': p_clf,
            'p_final': w * p_clf + (1 - w) * p_pois}


# ─────────────────────────────────────────────────────────────
# Predicción de partidos futuros
# ─────────────────────────────────────────────────────────────
_CACHE = {}


def cargar_modelos():
    if 'm' not in _CACHE:
        _CACHE['m'] = joblib.load(os.path.join(C.MODELOS, 'modelos.pkl'))
    return _CACHE['m']


def features_futuros(fixtures, estado=None, partidos=None):
    """fixtures: DataFrame con HomeTeam y AwayTeam → filas con las mismas columnas que el histórico."""
    estado = estado if estado is not None else pd.read_csv(C.RUTA_ESTADO, encoding='utf-8-sig')
    partidos = partidos if partidos is not None else pd.read_csv(C.RUTA_PARTIDOS, encoding='utf-8-sig', parse_dates=['Fecha'])
    est = estado.set_index('Equipo')
    cols = [c for c in estado.columns if c not in ('Equipo', 'Elo')]
    base = fixtures[['HomeTeam', 'AwayTeam']].reset_index(drop=True).copy()
    base['Elo_Local'] = est.loc[base.HomeTeam, 'Elo'].values
    base['Elo_Visitante'] = est.loc[base.AwayTeam, 'Elo'].values
    h2h = pd.DataFrame([F.h2h_par(partidos, h, a) for h, a in zip(base.HomeTeam, base.AwayTeam)])
    base = pd.concat([base, h2h], axis=1)
    return F.montar_filas(base, est.loc[base.HomeTeam].reset_index(), est.loc[base.AwayTeam].reset_index(), cols)


def predecir(fixtures, **kw):
    X = features_futuros(fixtures, **kw)
    return X, aplicar(cargar_modelos(), X)


# ─────────────────────────────────────────────────────────────
# Script principal: validación + reentrenamiento final
# ─────────────────────────────────────────────────────────────
RUTA_METRICAS = os.path.join(C.PRED, 'metricas_modelo.json')


def reentreno_rapido(df, cols):
    with open(RUTA_METRICAS, encoding='utf-8') as f:
        info = json.load(f)
    pL = {**PARAMS_REG, **info['hiperparametros_local']}
    pV = {**PARAMS_REG, **info['hiperparametros_visitante']}
    final = entrenar(df, pL, pV, cols)
    final['peso_clf'] = {'Poisson + Dixon-Coles': 0.0, 'Clasificador calibrado': 1.0}.get(info['modelo_elegido'], 0.5)
    joblib.dump(final, os.path.join(C.MODELOS, 'modelos.pkl'))
    info['fecha_entrenamiento'] = str(df.Fecha.max().date()); info['rho'] = final['rho']
    with open(RUTA_METRICAS, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'Reentreno rápido con {len(df)} partidos ({info["modelo_elegido"]}, rho={final["rho"]:.3f}) → modelos/modelos.pkl')


def main():
    df = pd.read_csv(C.RUTA_HISTORICO, encoding='utf-8-sig', parse_dates=['Fecha'], dtype={'Temporada': str})
    df = df.sort_values('Fecha').reset_index(drop=True)
    cols = F.columnas_modelo(df)
    if '--rapido' in sys.argv and os.path.exists(RUTA_METRICAS):
        return reentreno_rapido(df, cols)
    train = df[df.Temporada < TEMPORADAS_TEST[0]].reset_index(drop=True)
    test = df[df.Temporada.isin(TEMPORADAS_TEST)].reset_index(drop=True)
    print(f'Train: {len(train)} partidos ({train.Temporada.min()}–{train.Temporada.max()}) | '
          f'Test: {len(test)} ({"/".join(TEMPORADAS_TEST)}) | {len(cols)} features')

    print(f'Búsqueda de hiperparámetros ({N_ITER_BUSQUEDA} combinaciones × 4 folds temporales)...')
    wtr = pesos_recencia(train['Fecha'])
    pL = buscar_hiperparametros(train[cols], train['FTHG'], wtr)
    pV = buscar_hiperparametros(train[cols], train['FTAG'], wtr)
    print('  local    :', {k: pL[k] for k in ESPACIO_REG})
    print('  visitante:', {k: pV[k] for k in ESPACIO_REG})

    print('Validando en test...')
    m = entrenar(train, pL, pV, cols)
    r = aplicar(m, test)
    y = test['Resultado_1X2'].map({c: i for i, c in enumerate(CLASES)}).values
    pm = test[['Prob_Mercado_1', 'Prob_Mercado_X', 'Prob_Mercado_2']].values
    res = {
        'Poisson + Dixon-Coles': metricas(r['p_poisson'], y),
        'Clasificador calibrado': metricas(r['p_clf'], y),
        'Ensemble 50/50': metricas(0.5 * r['p_poisson'] + 0.5 * r['p_clf'], y),
        'Elo puro': metricas(np.c_[test.Prob_Elo * 0.75, np.full(len(test), 0.25), (1 - test.Prob_Elo) * 0.75], y),
        'Mercado (cuotas medias)': metricas(pm, y),
        'Base (frecuencias)': metricas(np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1)), y),
    }
    tabla = pd.DataFrame(res).T
    print(tabla.to_string())
    mae_goles = float(np.mean(np.abs(np.r_[r['lam_L'] - test.FTHG, r['lam_V'] - test.FTAG])))

    candidatos = {'Poisson + Dixon-Coles': 0.0, 'Clasificador calibrado': 1.0, 'Ensemble 50/50': 0.5}
    elegido = min(candidatos, key=lambda k: res[k]['log_loss'])
    print(f'\nModelo elegido por log-loss: {elegido}  (rho Dixon-Coles = {m["rho"]:.3f}, MAE goles = {mae_goles:.3f})')

    # Importancia de variables (ganancia media de ambos regresores)
    imp = (pd.Series(m['reg_L'].feature_importances_, cols) + pd.Series(m['reg_V'].feature_importances_, cols)) / 2
    imp.sort_values(ascending=False).head(25).rename('importancia').to_csv(
        os.path.join(C.PRED, 'importancia_variables_top25.csv'), encoding='utf-8-sig')

    print('Reentrenando con el 100 % de los datos...')
    final = entrenar(df, pL, pV, cols)
    final['peso_clf'] = candidatos[elegido]
    joblib.dump(final, os.path.join(C.MODELOS, 'modelos.pkl'))

    with open(RUTA_METRICAS, 'w', encoding='utf-8') as f:
        json.dump({'test_temporadas': TEMPORADAS_TEST, 'n_test': len(test), 'metricas': res,
                   'modelo_elegido': elegido, 'rho': final['rho'], 'mae_goles': round(mae_goles, 4),
                   'hiperparametros_local': {k: pL[k] for k in ESPACIO_REG},
                   'hiperparametros_visitante': {k: pV[k] for k in ESPACIO_REG},
                   'fecha_entrenamiento': str(df.Fecha.max().date())}, f, ensure_ascii=False, indent=2)
    print('→ modelos/modelos.pkl y Predicciones/metricas_modelo.json')


if __name__ == '__main__':
    main()
