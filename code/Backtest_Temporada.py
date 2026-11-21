"""
BACKTEST DE LA TEMPORADA EN CURSO — ¿qué habría predicho el modelo en cada jornada ya jugada?

Entrena SOLO con temporadas anteriores (nada de la temporada actual entra en el entrenamiento) y
predice cada partido jugado con las variables disponibles justo antes de él (medias móviles con
shift, Elo previo). Compara los goles esperados con:
  · los goles reales,
  · el xG real del partido,
  · los goles esperados implícitos en las cuotas (1X2 + más/menos de 2,5).

Uso:
    python code/Backtest_Temporada.py                  → modelo congelado de pretemporada
    python code/Backtest_Temporada.py --walk-forward   → reentrena antes de cada jornada (como el robot diario)
Salidas: Predicciones/backtest_partidos.csv y Predicciones/backtest_resumen.csv
"""
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import config as C
import features as F
import Modelo as Mo

warnings.filterwarnings('ignore', category=UserWarning)


def numerar_jornadas(partidos):
    """Jornada = nº de partido de la temporada del equipo que más lleva (aguanta aplazamientos)."""
    p = partidos.sort_values('Fecha').reset_index(drop=True)
    largo = pd.concat([p[['Fecha']].assign(eq=p.HomeTeam, i=p.index), p[['Fecha']].assign(eq=p.AwayTeam, i=p.index)])
    largo = largo.sort_values(['Fecha', 'i'])
    largo['n'] = largo.groupby('eq').cumcount() + 1
    p['Jornada'] = largo.groupby('i')['n'].max()
    return p


def cargar_params():
    with open(Mo.RUTA_METRICAS, encoding='utf-8') as f:
        info = json.load(f)
    peso = {'Poisson + Dixon-Coles': 0.0, 'Clasificador calibrado': 1.0}.get(info['modelo_elegido'], 0.5)
    return {**Mo.PARAMS_REG, **info['hiperparametros_local']}, {**Mo.PARAMS_REG, **info['hiperparametros_visitante']}, peso


def predecir_bloque(train, test, cols, pL, pV, peso):
    m = Mo.entrenar(train, pL, pV, cols)
    m['peso_clf'] = peso
    return Mo.aplicar(m, test)


def main():
    walk = '--walk-forward' in sys.argv
    df = pd.read_csv(C.RUTA_HISTORICO, encoding='utf-8-sig', parse_dates=['Fecha'], dtype={'Temporada': str})
    df = df.sort_values('Fecha').reset_index(drop=True)
    cols = F.columnas_modelo(df)
    pL, pV, peso = cargar_params()

    # Partidos reales de la temporada (goles, xG real, cuotas de goles) y nº de jornada
    raw = pd.read_csv(C.RUTA_PARTIDOS, encoding='utf-8-sig', parse_dates=['Fecha'], dtype={'Temporada': str})
    raw = numerar_jornadas(raw[(raw.Div == 'E0') & (raw.Temporada == C.TEMPORADA_ACTUAL)])
    extra = [c for c in ('Avg>2.5', 'Avg<2.5', 'AHh') if c in raw.columns]
    test = df[df.Temporada == C.TEMPORADA_ACTUAL].merge(
        raw[['Fecha', 'HomeTeam', 'AwayTeam', 'Jornada'] + extra], on=['Fecha', 'HomeTeam', 'AwayTeam'], how='left')
    test = test.sort_values(['Jornada', 'Fecha']).reset_index(drop=True)
    print(f'Backtest {C.TEMPORADA_ACTUAL}: {len(test)} partidos, jornadas {int(test.Jornada.min())}–{int(test.Jornada.max())} '
          f'| modo: {"walk-forward (reentrena cada jornada)" if walk else "modelo congelado de pretemporada"}')

    lam_L, lam_V = np.zeros(len(test)), np.zeros(len(test))
    P, M = np.zeros((len(test), 3)), np.zeros((len(test), Mo.MAX_GOLES + 1, Mo.MAX_GOLES + 1))
    bloques = sorted(test.Jornada.unique()) if walk else [None]
    for j in bloques:
        if walk:
            idx = np.where(test.Jornada == j)[0]
            corte = test.Fecha.iloc[idx].min()
            train = df[df.Fecha < corte].reset_index(drop=True)
            print(f'  Jornada {int(j)}: entrenando con {len(train)} partidos anteriores a {corte.date()}', flush=True)
        else:
            idx = np.arange(len(test))
            train = df[df.Temporada < C.TEMPORADA_ACTUAL].reset_index(drop=True)
            print(f'  Entrenando con {len(train)} partidos (hasta {train.Fecha.max().date()})...', flush=True)
        r = predecir_bloque(train, test.iloc[idx].reset_index(drop=True), cols, pL, pV, peso)
        lam_L[idx], lam_V[idx], P[idx], M[idx] = r['lam_L'], r['lam_V'], r['p_final'], r['M']

    # ── Marcador más probable (dentro del 1X2 más probable), over 2,5 y mercado ──
    n, G = M.shape[0], M.shape[1]
    flat = M.reshape(n, -1)
    gl, gv = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
    cat = np.select([gl > gv, gl == gv], [0, 1], 2).ravel()
    mejor = np.where(cat[None, :] == P.argmax(1)[:, None], flat, -1).argmax(1)
    p_over = flat[:, (gl + gv).ravel() > 2.5].sum(1)

    pm = test[['Prob_Mercado_1', 'Prob_Mercado_X', 'Prob_Mercado_2']].values
    po_m = F.prob_over_mercado(test['Avg>2.5'], test['Avg<2.5']) if 'Avg>2.5' in test else np.full(n, np.nan)
    gm = F.goles_mercado(pm, po_m)
    y = test.Resultado_1X2.map({c: i for i, c in enumerate(Mo.CLASES)}).values
    total_real = test.FTHG + test.FTAG

    out = pd.DataFrame({
        'Jornada': test.Jornada.astype(int), 'Fecha': test.Fecha.dt.date, 'Local': test.HomeTeam, 'Visitante': test.AwayTeam,
        'Goles_Local': test.FTHG.astype(int), 'Goles_Visitante': test.FTAG.astype(int),
        'Modelo_Goles_Local': lam_L, 'Modelo_Goles_Visitante': lam_V,
        'Mercado_Goles_Local': gm[:, 0], 'Mercado_Goles_Visitante': gm[:, 1],
        'xG_real_Local': test.get('HxG'), 'xG_real_Visitante': test.get('AxG'),
        'Marcador_probable': [f'{i // G}-{i % G}' for i in mejor],
        'Prob_marcador_real': M[np.arange(n), np.minimum(test.FTHG.astype(int), G - 1), np.minimum(test.FTAG.astype(int), G - 1)],
        'Prob_1': P[:, 0], 'Prob_X': P[:, 1], 'Prob_2': P[:, 2],
        'Mercado_1': pm[:, 0], 'Mercado_X': pm[:, 1], 'Mercado_2': pm[:, 2],
        'Resultado': test.Resultado_1X2,
        'Prob_Over25': p_over, 'Mercado_Over25': po_m, 'Over25_real': (total_real > 2.5).astype(int),
    })
    out['Acierto_marcador'] = (out.Marcador_probable == out.Goles_Local.astype(str) + '-' + out.Goles_Visitante.astype(str)).astype(int)
    out['Acierto_1X2_modelo'] = (P.argmax(1) == y).astype(int)
    out['Acierto_1X2_mercado'] = (pm.argmax(1) == y).astype(int)
    out = out.round(4)

    # ── Resumen por jornada y total ──
    def resumir(g):
        yy = g.Resultado.map({c: i for i, c in enumerate(Mo.CLASES)}).values
        gr = np.r_[g.Goles_Local, g.Goles_Visitante]
        gmod = np.r_[g.Modelo_Goles_Local, g.Modelo_Goles_Visitante]
        gmer = np.r_[g.Mercado_Goles_Local, g.Mercado_Goles_Visitante]
        xg = np.r_[g.xG_real_Local, g.xG_real_Visitante].astype(float)
        ll = lambda p: log_loss(yy, np.clip(p, 1e-6, 1) / np.clip(p, 1e-6, 1).sum(1, keepdims=True), labels=[0, 1, 2])
        return pd.Series({
            'Partidos': len(g),
            'Goles_reales': gr.sum(), 'Goles_modelo': gmod.sum(), 'Goles_mercado': np.nansum(gmer),
            'MAE_modelo': np.mean(np.abs(gmod - gr)), 'MAE_mercado': np.nanmean(np.abs(gmer - gr)),
            'MAE_media_liga': np.mean(np.abs(media_liga - gr)),
            'MAE_modelo_vs_xG': np.nanmean(np.abs(gmod - xg)), 'MAE_mercado_vs_xG': np.nanmean(np.abs(gmer - xg)),
            'Marcadores_exactos': g.Acierto_marcador.sum(),
            'Acierto_1X2_modelo': g.Acierto_1X2_modelo.mean(), 'Acierto_1X2_mercado': g.Acierto_1X2_mercado.mean(),
            'LogLoss_modelo': ll(g[['Prob_1', 'Prob_X', 'Prob_2']].values),
            'LogLoss_mercado': ll(g[['Mercado_1', 'Mercado_X', 'Mercado_2']].values),
            'Brier_Over25_modelo': np.mean((g.Prob_Over25 - g.Over25_real) ** 2),
            'Brier_Over25_mercado': np.nanmean((g.Mercado_Over25 - g.Over25_real) ** 2),
        })

    hist = df[df.Temporada < C.TEMPORADA_ACTUAL]
    media_liga = np.r_[hist.FTHG, hist.FTAG][-3800:].mean()      # referencia ingenua: media de las últimas temporadas
    resumen = out.groupby('Jornada').apply(resumir, include_groups=False)
    resumen.loc['Total'] = resumir(out)
    resumen = resumen.round(3)

    sufijo = '_walkforward' if walk else ''
    out.to_csv(os.path.join(C.PRED, f'backtest_partidos{sufijo}.csv'), index=False, encoding='utf-8-sig')
    resumen.to_csv(os.path.join(C.PRED, f'backtest_resumen{sufijo}.csv'), encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    print('\n' + resumen[['Partidos', 'Goles_reales', 'Goles_modelo', 'Goles_mercado', 'MAE_modelo', 'MAE_mercado',
                          'MAE_media_liga', 'Marcadores_exactos', 'Acierto_1X2_modelo', 'Acierto_1X2_mercado',
                          'LogLoss_modelo', 'LogLoss_mercado']].to_string())
    t = resumen.loc['Total']
    print(f'\nvs xG real → MAE modelo {t.MAE_modelo_vs_xG:.3f} | mercado {t.MAE_mercado_vs_xG:.3f}'
          f'\nOver 2,5 (Brier) → modelo {t.Brier_Over25_modelo:.3f} | mercado {t.Brier_Over25_mercado:.3f}')
    print(f'→ Predicciones/backtest_partidos{sufijo}.csv y backtest_resumen{sufijo}.csv')


if __name__ == '__main__':
    main()
