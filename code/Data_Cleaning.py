"""
2. LIMPIEZA + INGENIERÍA DE VARIABLES
Entrada : Data/partidos.csv (+ Data/flashscore_premier.csv opcional)
Salidas : Data/datos_historicos.csv  → una fila por partido de Premier con features pre-partido
          Data/estado_equipos.csv    → "foto" actual de cada equipo para predecir partidos futuros
"""
import os
import numpy as np
import pandas as pd
import config as C
import features as F
from nombres import normalizar_flashscore


def cargar():
    df = pd.read_csv(C.RUTA_PARTIDOS, encoding='utf-8-sig', parse_dates=['Fecha'])
    df['Temporada'] = df['Temporada'].astype(str).str.zfill(4)
    df = df.dropna(subset=['HS', 'AS', 'HST', 'AST']).copy()      # sin stats básicas → fuera
    for c in ['HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HF', 'AF']:
        df[c] = df[c].fillna(df[c].median())
    for c in ['HxG', 'AxG']:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values(['Fecha', 'Div', 'HomeTeam']).reset_index(drop=True)
    df['match_id'] = (df['Div'] + '_' + df['Fecha'].dt.strftime('%Y%m%d') + '_'
                      + df['HomeTeam'].str.replace(' ', '') + '_' + df['AwayTeam'].str.replace(' ', ''))
    return df


def integrar_flashscore(df):
    """Si existe el CSV del scraper de FlashScore, usa su xG real donde falte."""
    if not os.path.exists(C.RUTA_FLASHSCORE):
        return df
    fs = pd.read_csv(C.RUTA_FLASHSCORE, encoding='utf-8-sig')
    col_l, col_v = 'Goles_esperados_(xG)_Local', 'Goles_esperados_(xG)_Visitante'
    if col_l not in fs.columns:
        return df
    fs['Fecha'] = pd.to_datetime(fs['Fecha'].astype(str).str[:10], dayfirst=True, errors='coerce').dt.normalize()
    fs['HomeTeam'] = fs['Equipo_Local'].map(normalizar_flashscore)
    fs['AwayTeam'] = fs['Equipo_Visitante'].map(normalizar_flashscore)
    fs = fs.dropna(subset=['Fecha']).rename(columns={col_l: 'fs_HxG', col_v: 'fs_AxG'})
    fs = fs[['Fecha', 'HomeTeam', 'AwayTeam', 'fs_HxG', 'fs_AxG']].drop_duplicates(['Fecha', 'HomeTeam', 'AwayTeam'])
    df['_f'] = df['Fecha'].dt.normalize()
    df = df.merge(fs, left_on=['_f', 'HomeTeam', 'AwayTeam'], right_on=['Fecha', 'HomeTeam', 'AwayTeam'],
                  how='left', suffixes=('', '_fs')).drop(columns=['_f', 'Fecha_fs'])
    for lado in ('H', 'A'):
        df[f'{lado}xG'] = df[f'{lado}xG'].fillna(pd.to_numeric(df[f'fs_{lado}xG'], errors='coerce'))
    print(f'  FlashScore: xG real añadido en {df.fs_HxG.notna().sum()} partidos')
    return df.drop(columns=['fs_HxG', 'fs_AxG'])


def main():
    print('Cargando partidos...')
    df = integrar_flashscore(cargar())
    df = F.añadir_xg(df)

    print('Calculando Elo...')
    df['Elo_Local'], df['Elo_Visitante'], elo_final = F.calcular_elo(df)

    print('Calculando H2H...')
    df = pd.concat([df, F.calcular_h2h(df)], axis=1)

    print('Formato largo + medias móviles...')
    actual = df[(df.Div == 'E0') & (df.Temporada == C.TEMPORADA_ACTUAL)]
    equipos_actuales = sorted(set(actual.HomeTeam) | set(actual.AwayTeam))
    largo = F.formato_largo(df)
    largo = F.añadir_centinelas(largo, equipos_actuales, df['Fecha'].max() + pd.Timedelta(days=1))
    feats = F.medias_moviles(largo)
    cols = F.columnas_equipo(feats)

    # ── Dataset histórico (solo Premier, a partir de la 2ª temporada para tener historial) ──
    prem = df[(df.Div == 'E0') & (df.Temporada > C.temporadas()[0])].copy()
    fi = feats.set_index(['match_id', 'Equipo'])
    fL = fi.loc[list(zip(prem.match_id, prem.HomeTeam))].reset_index()
    fV = fi.loc[list(zip(prem.match_id, prem.AwayTeam))].reset_index()

    base = prem[['match_id', 'Fecha', 'Temporada', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                 'HxG', 'AxG', 'Elo_Local', 'Elo_Visitante',
                 'h2h_dg_Local', 'h2h_pts_Local', 'h2h_pts_Visitante', 'h2h_n']].copy()
    base['Resultado_1X2'] = np.select([base.FTHG > base.FTAG, base.FTHG == base.FTAG], ['1', 'X'], '2')
    pm = F.probs_mercado(prem.AvgH.values, prem.AvgD.values, prem.AvgA.values)
    base[['Prob_Mercado_1', 'Prob_Mercado_X', 'Prob_Mercado_2']] = pm
    if 'AvgCH' in prem.columns:
        pc = F.probs_mercado(prem.AvgCH.values, prem.AvgCD.values, prem.AvgCA.values)
        base[['Prob_Cierre_1', 'Prob_Cierre_X', 'Prob_Cierre_2']] = pc

    hist = F.montar_filas(base, fL, fV, cols)
    hist.to_csv(C.RUTA_HISTORICO, index=False, encoding='utf-8-sig')
    print(f'→ {C.RUTA_HISTORICO}: {hist.shape[0]} partidos × {len(F.columnas_modelo(hist))} features')

    # ── Estado actual de cada equipo de la Premier ──
    est = feats[feats.match_id.str.startswith('ESTADO_')].copy()
    est['Elo'] = est['Equipo'].map(elo_final)
    est['Descanso'] = 7.0
    est = est[['Equipo', 'Elo'] + cols]
    est.to_csv(C.RUTA_ESTADO, index=False, encoding='utf-8-sig')
    print(f'→ {C.RUTA_ESTADO}: {len(est)} equipos')
    print(est[['Equipo', 'Elo', 'Puntos_temp', 'xGF_ewm', 'xGC_ewm']].sort_values('Elo', ascending=False)
          .round(2).to_string(index=False))


if __name__ == '__main__':
    main()
