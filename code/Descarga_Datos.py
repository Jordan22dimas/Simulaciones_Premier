"""
1. DESCARGA DE DATOS — football-data.co.uk
Descarga Premier League (E0) y Championship (E1) desde 2005-06 hasta la temporada actual,
más el fichero de próximos partidos. Las temporadas cerradas se cachean en Data/raw/.
"""
import io
import os
import time
import requests
import pandas as pd
import config as C

COLUMNAS = [
    'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
    'Referee', 'HxG', 'AxG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
    # cuotas medias (pre-partido y cierre) → solo benchmark
    'AvgH', 'AvgD', 'AvgA', 'AvgCH', 'AvgCD', 'AvgCA', 'BbAvH', 'BbAvD', 'BbAvA',
    'B365H', 'B365D', 'B365A',
]

SESION = requests.Session()
SESION.headers['User-Agent'] = 'Simulaciones-Premier/1.0 (proyecto educativo)'


def descargar_csv(url, destino=None, forzar=False):
    if destino and os.path.exists(destino) and not forzar:
        return pd.read_csv(destino, encoding='utf-8-sig')
    r = SESION.get(url, timeout=30)
    r.raise_for_status()
    texto = r.content.decode('utf-8-sig', errors='replace')
    df = pd.read_csv(io.StringIO(texto), on_bad_lines='skip')
    if destino:
        df.to_csv(destino, index=False, encoding='utf-8-sig')
    time.sleep(0.5)  # cortesía con el servidor
    return df


def main():
    trozos = []
    for temp in C.temporadas():
        for div in C.DIVISIONES:
            destino = os.path.join(C.RAW, f'{div}_{temp}.csv')
            actual = temp == C.TEMPORADA_ACTUAL
            try:
                df = descargar_csv(C.URL_FD.format(temp=temp, div=div), destino, forzar=actual)
            except Exception as e:
                print(f'  ! {div} {temp}: {e}')
                continue
            df = df[[c for c in COLUMNAS if c in df.columns]].copy()
            df['Temporada'] = temp
            trozos.append(df)
            print(f'  {div} {temp}: {len(df)} partidos')

    df = pd.concat(trozos, ignore_index=True)
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    df['Fecha'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')

    # Cuotas medias: el nombre de la columna cambió con los años
    for a, b in [('AvgH', 'BbAvH'), ('AvgD', 'BbAvD'), ('AvgA', 'BbAvA')]:
        if b in df.columns:
            df[a] = df[a].fillna(df[b]) if a in df.columns else df[b]
    df = df.drop(columns=[c for c in ('BbAvH', 'BbAvD', 'BbAvA', 'Date') if c in df.columns])
    df = df.sort_values(['Fecha', 'Div']).reset_index(drop=True)
    df.to_csv(C.RUTA_PARTIDOS, index=False, encoding='utf-8-sig')
    print(f'\n→ {C.RUTA_PARTIDOS}: {len(df)} partidos ({df.Fecha.min().date()} – {df.Fecha.max().date()})')

    # Próximos partidos de la Premier
    try:
        fx = descargar_csv(C.URL_FIXTURES)
        fx = fx[fx['Div'] == 'E0'].copy()
        fx['Fecha'] = pd.to_datetime(fx['Date'], dayfirst=True, format='mixed')
        keep = ['Fecha', 'Time', 'HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA', 'B365H', 'B365D', 'B365A']
        fx = fx[[c for c in keep if c in fx.columns]]
        fx.to_csv(C.RUTA_PROXIMOS, index=False, encoding='utf-8-sig')
        print(f'→ {C.RUTA_PROXIMOS}: {len(fx)} próximos partidos')
    except Exception as e:
        print(f'  ! fixtures: {e}')


if __name__ == '__main__':
    main()
