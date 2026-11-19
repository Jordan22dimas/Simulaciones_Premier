"""
4. MONTE CARLO DE LA TEMPORADA
Parte de la clasificación real actual y simula N veces todos los partidos pendientes.

En cada simulación:
  1. Cada equipo recibe un "shock" de fuerza ~ N(0, SIGMA_FUERZA) en escala logit, que se mantiene
     toda la temporada (lesiones, fichajes, rachas...). Sin esto el Monte Carlo es sobreconfiado.
  2. El resultado 1X2 se sortea con las probabilidades del modelo desplazadas por ese shock.
  3. El marcador se sortea de la matriz Poisson/Dixon-Coles condicionada al resultado (para la DG).
Desempates: puntos → diferencia de goles → goles a favor → sorteo.
"""
import os
import numpy as np
import pandas as pd
import config as C
from Modelo import predecir, MAX_GOLES

G = MAX_GOLES + 1


def clasificacion_actual(partidos):
    p = partidos[(partidos.Div == 'E0') & (partidos.Temporada == C.TEMPORADA_ACTUAL)]
    filas = []
    for lado, o in (('Home', 'Away'), ('Away', 'Home')):
        gf = p['FTHG'] if lado == 'Home' else p['FTAG']
        gc = p['FTAG'] if lado == 'Home' else p['FTHG']
        filas.append(pd.DataFrame({'Equipo': p[f'{lado}Team'], 'GF': gf, 'GC': gc,
                                   'G': (gf > gc).astype(int), 'E': (gf == gc).astype(int), 'P': (gf < gc).astype(int)}))
    t = pd.concat(filas).groupby('Equipo').sum()
    t['PJ'] = t[['G', 'E', 'P']].sum(axis=1)
    t['DG'] = t.GF - t.GC
    t['Pts'] = 3 * t.G + t.E
    t = t.sort_values(['Pts', 'DG', 'GF'], ascending=False)
    t.insert(0, 'Pos', range(1, len(t) + 1))
    return t.reset_index(), p


def cdf_condicionales(M):
    """Para cada partido y resultado (1, X, 2): CDF sobre las G×G celdas de marcador."""
    n = M.shape[0]
    gl, gv = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
    cat = np.select([gl > gv, gl == gv], [0, 1], 2).ravel()
    flat = M.reshape(n, -1)
    cdfs = np.zeros((n, 3, G * G))
    for k in range(3):
        m = np.where(cat == k, flat, 0)
        cdfs[:, k] = np.cumsum(m, 1) / np.clip(m.sum(1, keepdims=True), 1e-12, None)
    return cdfs


def simular(tabla, pendientes, probs, M, n_sims=C.N_SIMULACIONES, bloque=250, semilla=C.SEMILLA):
    rng = np.random.default_rng(semilla)
    equipos = tabla['Equipo'].tolist()
    idx = {e: i for i, e in enumerate(equipos)}
    T, n = len(equipos), len(pendientes)
    h = pendientes['HomeTeam'].map(idx).values
    a = pendientes['AwayTeam'].map(idx).values
    cdfs = cdf_condicionales(M)
    logp = np.log(np.clip(probs, 1e-9, 1))
    base_pts, base_gf, base_gc = (tabla[c].values.astype(float) for c in ('Pts', 'GF', 'GC'))

    conteo = np.zeros((T, T), dtype=np.int64)
    puntos_finales = np.zeros((n_sims, T), dtype=np.int16)
    hecho = 0
    while hecho < n_sims:
        b = min(bloque, n_sims - hecho)
        shock = rng.normal(0, C.SIGMA_FUERZA, (b, T))
        d = shock[:, h] - shock[:, a]                                  # (b, n)
        lp = logp[None] + np.stack([d, np.zeros_like(d), -d], -1)      # (b, n, 3)
        p = np.exp(lp - lp.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        res = (rng.random((b, n, 1)) > np.cumsum(p, -1)).sum(-1).clip(max=2)   # 0=1, 1=X, 2=2
        cdf = cdfs[np.arange(n)[None, :], res]                         # (b, n, G*G)
        celda = (cdf < rng.random((b, n, 1))).sum(-1).clip(max=G * G - 1)
        gl, gv = celda // G, celda % G

        pts = np.tile(base_pts, (b, 1)); gf = np.tile(base_gf, (b, 1)); gc = np.tile(base_gc, (b, 1))
        pl = np.select([res == 0, res == 1], [3, 1], 0); pv = np.select([res == 2, res == 1], [3, 1], 0)
        for arr, eq, val in ((pts, h, pl), (pts, a, pv), (gf, h, gl), (gf, a, gv), (gc, h, gv), (gc, a, gl)):
            np.add.at(arr, (np.arange(b)[:, None], eq[None, :]), val)

        clave = pts * 1e6 + (gf - gc + 500) * 1e3 + gf + rng.random((b, T)) * 0.5
        orden = np.argsort(-clave, 1)
        pos = np.empty_like(orden); pos[np.arange(b)[:, None], orden] = np.arange(T)
        for t in range(T):
            conteo[t] += np.bincount(pos[:, t], minlength=T)
        puntos_finales[hecho:hecho + b] = pts
        hecho += b
    return conteo / n_sims, puntos_finales


def resumen(tabla, prob_pos, puntos_finales):
    pos = np.arange(1, prob_pos.shape[1] + 1)
    out = tabla[['Equipo', 'Pos', 'PJ', 'Pts', 'DG']].rename(columns={'Pos': 'Pos_actual', 'Pts': 'Pts_actuales'}).copy()
    out['Pts_esperados'] = puntos_finales.mean(0).round(1)
    out['Pts_p5'] = np.percentile(puntos_finales, 5, axis=0)
    out['Pts_p95'] = np.percentile(puntos_finales, 95, axis=0)
    out['Pos_media'] = (prob_pos * pos).sum(1).round(2)
    for zona, posiciones in C.ZONAS.items():
        out[f'P_{zona}'] = prob_pos[:, np.array(posiciones) - 1].sum(1).round(4)
    return out.sort_values(['Pts_esperados', 'Pos_media'], ascending=[False, True]).reset_index(drop=True)


def main():
    partidos = pd.read_csv(C.RUTA_PARTIDOS, encoding='utf-8-sig', parse_dates=['Fecha'])
    partidos['Temporada'] = partidos['Temporada'].astype(str).str.zfill(4)
    tabla, jugados = clasificacion_actual(partidos)
    hechos = set(zip(jugados.HomeTeam, jugados.AwayTeam))
    equipos = tabla['Equipo'].tolist()
    pendientes = pd.DataFrame([(h, a) for h in equipos for a in equipos if h != a and (h, a) not in hechos],
                              columns=['HomeTeam', 'AwayTeam'])
    print(f'Clasificación tras {int(tabla.PJ.mean())} jornadas · {len(pendientes)} partidos pendientes')

    X, r = predecir(pendientes, partidos=partidos)
    probs = r['p_final']
    print(f'Simulando {C.N_SIMULACIONES:,} temporadas...')
    prob_pos, pts_fin = simular(tabla, pendientes, probs, r['M'])

    res = resumen(tabla, prob_pos, pts_fin)
    res.to_csv(os.path.join(C.PRED, 'simulacion_temporada.csv'), index=False, encoding='utf-8-sig')
    mat = pd.DataFrame(prob_pos, index=tabla.Equipo, columns=range(1, len(equipos) + 1)).loc[res.Equipo]
    mat.to_csv(os.path.join(C.PRED, 'probabilidades_posicion.csv'), encoding='utf-8-sig')
    tabla.to_csv(os.path.join(C.PRED, 'clasificacion_actual.csv'), index=False, encoding='utf-8-sig')

    pend = pendientes.copy()
    pend[['Prob_1', 'Prob_X', 'Prob_2']] = probs.round(4)
    pend['xG_Local'], pend['xG_Visitante'] = r['lam_L'].round(2), r['lam_V'].round(2)
    pend.to_csv(os.path.join(C.PRED, 'partidos_pendientes.csv'), index=False, encoding='utf-8-sig')

    cols = ['Equipo', 'Pts_actuales', 'Pts_esperados', 'Pts_p5', 'Pts_p95'] + [f'P_{z}' for z in C.ZONAS]
    with pd.option_context('display.width', 200):
        print(res[cols].to_string(index=False))
    print('→ Predicciones/simulacion_temporada.csv, probabilidades_posicion.csv, partidos_pendientes.csv')


if __name__ == '__main__':
    main()
