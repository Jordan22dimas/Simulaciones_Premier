"""
Ingeniería de variables compartida por la limpieza (entrenamiento) y la predicción.
Garantiza que los partidos futuros se construyen con exactamente las mismas columnas.
"""
import numpy as np
import pandas as pd
import config as C

# Estadísticas por equipo y partido (formato largo)
STATS = ['GF', 'GC', 'xGF', 'xGC', 'Remates_F', 'Remates_C', 'RemPuerta_F', 'RemPuerta_C',
         'Corners_F', 'Corners_C', 'Tarjetas', 'Puntos']
STATS_TEMPORADA = ['Puntos', 'GF', 'GC', 'xGF', 'xGC']
SPAN_EWM = 10


# ─────────────────────────────────────────────────────────────
# xG estimado (para temporadas sin xG real)
# ─────────────────────────────────────────────────────────────
def ajustar_xg_estimado(df):
    """xG ≈ a·remates_a_puerta + b·remates_fuera (mínimos cuadrados sin intercepto,
    ajustado sobre los partidos que sí tienen xG real). Devuelve (a, b)."""
    filas = []
    for lado, o in (('H', 'A'), ('A', 'H')):
        m = df[[f'{lado}xG', f'{lado}ST', f'{lado}S']].dropna()
        filas.append(np.c_[m[f'{lado}ST'], m[f'{lado}S'] - m[f'{lado}ST'], m[f'{lado}xG']])
    M = np.vstack(filas)
    if len(M) < 50:                      # sin datos suficientes → coeficientes de referencia
        return 0.30, 0.05
    coef, *_ = np.linalg.lstsq(M[:, :2], M[:, 2], rcond=None)
    return float(coef[0]), float(coef[1])


def añadir_xg(df):
    a, b = ajustar_xg_estimado(df)
    for lado in ('H', 'A'):
        est = a * df[f'{lado}ST'] + b * (df[f'{lado}S'] - df[f'{lado}ST']).clip(lower=0)
        df[f'{lado}xG_final'] = df[f'{lado}xG'].fillna(est)
    print(f'  xG estimado = {a:.3f}·remates_a_puerta + {b:.3f}·remates_fuera')
    return df


# ─────────────────────────────────────────────────────────────
# Elo (Premier + Championship en una única escala)
# ─────────────────────────────────────────────────────────────
def calcular_elo(df):
    """Recorre los partidos en orden cronológico. Devuelve (elo_local_pre, elo_visit_pre, elo_final).
    Al cambiar de temporada: los ascendidos heredan la media Elo de los descendidos (y viceversa),
    y todos regresan un 20 % hacia la media de su división."""
    elo, prev = {}, {'E0': set(), 'E1': set()}
    pre_h = np.empty(len(df)); pre_a = np.empty(len(df))
    orden_temporadas = df.groupby('Temporada')['Fecha'].min().sort_values().index

    for temp in orden_temporadas:
        idx = np.where(df['Temporada'].values == temp)[0]
        g = df.iloc[idx]
        ahora = {d: set(g.loc[g.Div == d, 'HomeTeam']) for d in ('E0', 'E1')}

        if prev['E0']:
            suben, bajan = ahora['E0'] - prev['E0'], prev['E0'] - ahora['E0']
            elo_bajan = np.mean([elo[t] for t in bajan]) if bajan else None
            elo_suben = np.mean([elo[t] for t in suben if t in elo]) if suben else None
            for t in suben:
                if elo_bajan is not None: elo[t] = elo_bajan
            for t in bajan:
                if elo_suben is not None: elo[t] = elo_suben
        for d, ini in (('E0', 1600), ('E1', 1400)):
            for t in ahora[d]:
                elo.setdefault(t, ini - (50 if prev[d] else 0))    # recién llegados desde League One
            media = np.mean([elo[t] for t in ahora[d]])
            for t in ahora[d]:
                elo[t] = 0.8 * elo[t] + 0.2 * media
        prev = ahora

        for i in idx:
            h, a = df.at[i, 'HomeTeam'], df.at[i, 'AwayTeam']
            gh, ga = df.at[i, 'FTHG'], df.at[i, 'FTAG']
            pre_h[i], pre_a[i] = elo[h], elo[a]
            esperado = 1 / (1 + 10 ** (-(elo[h] + C.ELO_HFA - elo[a]) / 400))
            real = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
            mult = np.log(abs(gh - ga) + 1) + 1
            delta = C.ELO_K * mult * (real - esperado)
            elo[h] += delta; elo[a] -= delta
    return pre_h, pre_a, elo


def prob_elo(elo_l, elo_v):
    return 1 / (1 + 10 ** (-(elo_l + C.ELO_HFA - elo_v) / 400))


# ─────────────────────────────────────────────────────────────
# H2H: últimos 6 enfrentamientos (cualquier campo), perspectiva del local
# ─────────────────────────────────────────────────────────────
def calcular_h2h(df, n=6):
    d = df[['Fecha', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
    primero = np.where(d.HomeTeam < d.AwayTeam, d.HomeTeam, d.AwayTeam)
    d['_par'] = primero + '|' + np.where(d.HomeTeam < d.AwayTeam, d.AwayTeam, d.HomeTeam)
    d['_local_es_1'] = d.HomeTeam == primero
    d['_gd1'] = np.where(d._local_es_1, d.FTHG - d.FTAG, d.FTAG - d.FTHG)
    d['_pts1'] = np.select([d._gd1 > 0, d._gd1 == 0], [3, 1], 0)
    d['_pts2'] = np.select([d._gd1 < 0, d._gd1 == 0], [3, 1], 0)
    d = d.sort_values('Fecha')
    g = d.groupby('_par')
    roll = lambda s: s.shift().rolling(n, min_periods=1).mean()
    gd1, p1, p2 = g['_gd1'].transform(roll), g['_pts1'].transform(roll), g['_pts2'].transform(roll)
    nprev = g.cumcount().clip(upper=n)
    out = pd.DataFrame(index=d.index)
    out['h2h_dg_Local'] = np.where(d._local_es_1, gd1, -gd1)
    out['h2h_pts_Local'] = np.where(d._local_es_1, p1, p2)
    out['h2h_pts_Visitante'] = np.where(d._local_es_1, p2, p1)
    out['h2h_n'] = nprev
    return out.reindex(df.index)


def h2h_par(df, local, visitante, n=6):
    m = df[((df.HomeTeam == local) & (df.AwayTeam == visitante)) |
           ((df.HomeTeam == visitante) & (df.AwayTeam == local))].sort_values('Fecha').tail(n)
    if m.empty:
        return {'h2h_dg_Local': np.nan, 'h2h_pts_Local': np.nan, 'h2h_pts_Visitante': np.nan, 'h2h_n': 0}
    gd = np.where(m.HomeTeam == local, m.FTHG - m.FTAG, m.FTAG - m.FTHG)
    pl = np.select([gd > 0, gd == 0], [3, 1], 0); pv = np.select([gd < 0, gd == 0], [3, 1], 0)
    return {'h2h_dg_Local': gd.mean(), 'h2h_pts_Local': pl.mean(),
            'h2h_pts_Visitante': pv.mean(), 'h2h_n': len(m)}


# ─────────────────────────────────────────────────────────────
# Formato largo + medias móviles
# ─────────────────────────────────────────────────────────────
def formato_largo(df):
    partes = []
    for lado, o, es_local in (('H', 'A', 1), ('A', 'H', 0)):
        p = pd.DataFrame({
            'match_id': df['match_id'], 'Fecha': df['Fecha'], 'Temporada': df['Temporada'],
            'Div': df['Div'], 'Equipo': df[f'{"Home" if lado == "H" else "Away"}Team'],
            'Es_Local': es_local,
            'GF': df[f'FT{lado}G'], 'GC': df[f'FT{o}G'],
            'xGF': df[f'{lado}xG_final'], 'xGC': df[f'{o}xG_final'],
            'Remates_F': df[f'{lado}S'], 'Remates_C': df[f'{o}S'],
            'RemPuerta_F': df[f'{lado}ST'], 'RemPuerta_C': df[f'{o}ST'],
            'Corners_F': df[f'{lado}C'], 'Corners_C': df[f'{o}C'],
            'Tarjetas': df[f'{lado}Y'] + 2 * df[f'{lado}R'],
        })
        dif = p['GF'] - p['GC']
        p['Puntos'] = np.select([dif > 0, dif == 0], [3, 1], 0).astype(float)
        partes.append(p)
    return pd.concat(partes, ignore_index=True)


def añadir_centinelas(largo, equipos, fecha):
    """Fila 'fantasma' posterior al último partido de cada equipo: tras shift() contiene
    el estado actual (medias que SÍ incluyen el último partido jugado)."""
    temp_actual = C.TEMPORADA_ACTUAL
    filas = pd.DataFrame({'match_id': [f'ESTADO_{e}' for e in equipos], 'Fecha': fecha,
                          'Temporada': temp_actual, 'Div': 'E0', 'Equipo': list(equipos),
                          'Es_Local': np.nan})
    return pd.concat([largo, filas], ignore_index=True)


def medias_moviles(largo):
    largo = largo.sort_values(['Fecha', 'match_id']).reset_index(drop=True)
    g = largo.groupby('Equipo')
    nuevas = {}
    for var in STATS:
        s = g[var].shift()
        gs = s.groupby(largo['Equipo'])
        for w in C.VENTANAS:
            nuevas[f'{var}_{w}'] = gs.transform(lambda x: x.rolling(w, min_periods=1).mean())
        nuevas[f'{var}_ewm'] = gs.transform(lambda x: x.ewm(span=SPAN_EWM, ignore_na=True).mean())
    # Media de la temporada en curso (NaN en la jornada 1 → XGBoost lo gestiona)
    clave = largo['Equipo'] + '|' + largo['Temporada'].astype(str)
    for var in STATS_TEMPORADA:
        s = largo.groupby(clave)[var].shift()
        nuevas[f'{var}_temp'] = s.groupby(clave).transform(lambda x: x.expanding().mean())
    nuevas['Partidos_temp'] = largo.groupby(clave).cumcount()
    # Forma en casa / fuera (últimos 5 en la misma condición)
    cond = largo['Equipo'] + '|' + largo['Es_Local'].astype(str)
    s = largo.groupby(cond)['Puntos'].shift()
    nuevas['Puntos_condicion_5'] = s.groupby(cond).transform(lambda x: x.rolling(5, min_periods=1).mean())
    # % de los últimos 10 partidos jugados en Championship (ascendidos)
    e1 = (largo['Div'] == 'E1').astype(float)
    nuevas['pct_championship_10'] = e1.groupby(largo['Equipo']).transform(
        lambda x: x.shift().rolling(10, min_periods=1).mean())
    # Descanso (días desde el último partido de liga, acotado)
    nuevas['Descanso'] = g['Fecha'].diff().dt.days.clip(upper=14)
    feats = pd.DataFrame(nuevas, index=largo.index)
    feats['forma_vs_historia'] = (feats['xGF_5'] - feats['xGC_5']) - (feats['xGF_ewm'] - feats['xGC_ewm'])
    return pd.concat([largo[['match_id', 'Fecha', 'Temporada', 'Div', 'Equipo', 'Es_Local']], feats], axis=1)


COLS_EQUIPO = None  # se rellena tras medias_moviles


def columnas_equipo(feats):
    return [c for c in feats.columns if c not in ('match_id', 'Fecha', 'Temporada', 'Div', 'Equipo', 'Es_Local')]


# ─────────────────────────────────────────────────────────────
# Ensamblado de filas partido (idéntico para histórico y futuro)
# ─────────────────────────────────────────────────────────────
def montar_filas(base, feats_L, feats_V, cols):
    """base: DataFrame con HomeTeam, AwayTeam, Elo_Local, Elo_Visitante y columnas h2h_*.
    feats_L / feats_V: features por equipo alineadas fila a fila con base."""
    L = feats_L[cols].add_suffix('_Local').reset_index(drop=True)
    V = feats_V[cols].add_suffix('_Visitante').reset_index(drop=True)
    D = pd.DataFrame(feats_L[cols].values - feats_V[cols].values, columns=[f'diff_{c}' for c in cols])
    out = pd.concat([base.reset_index(drop=True), L, V, D], axis=1).copy()
    out['diff_Elo'] = out['Elo_Local'] - out['Elo_Visitante']
    out['Prob_Elo'] = prob_elo(out['Elo_Local'], out['Elo_Visitante'])
    return out


COLUMNAS_NO_FEATURE = {
    'match_id', 'Fecha', 'Temporada', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
    'Resultado_1X2', 'Prob_Mercado_1', 'Prob_Mercado_X', 'Prob_Mercado_2',
    'Prob_Cierre_1', 'Prob_Cierre_X', 'Prob_Cierre_2', 'HxG', 'AxG',
}


def columnas_modelo(df):
    return [c for c in df.columns if c not in COLUMNAS_NO_FEATURE and pd.api.types.is_numeric_dtype(df[c])]


def probs_mercado(h, d, a):
    """Probabilidades implícitas sin margen (normalización proporcional)."""
    inv = np.c_[1 / h, 1 / d, 1 / a]
    return inv / inv.sum(axis=1, keepdims=True)
