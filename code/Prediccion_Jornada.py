"""
5. PREDICCIÓN DE LA PRÓXIMA JORNADA
Toma los próximos partidos publicados por football-data (Data/proximos_partidos.csv) y devuelve
probabilidades 1X2, goles esperados, marcador más probable, over 2.5, ambos marcan y la
comparación con la probabilidad implícita del mercado ("valor" = modelo − mercado).
"""
import os
import numpy as np
import pandas as pd
import config as C
import features as F
from Modelo import predecir

COLUMNAS = ['Fecha', 'Hora', 'Local', 'Visitante', 'Prob_1', 'Prob_X', 'Prob_2', 'xG_Local', 'xG_Visitante',
            'Marcador_probable', 'Prob_marcador', 'Prob_Over25', 'Prob_BTTS',
            'Mercado_1', 'Mercado_X', 'Mercado_2', 'Valor_1', 'Valor_X', 'Valor_2',
            'Mercado_Over25', 'Valor_Over25', 'Mercado_xG_Local', 'Mercado_xG_Visitante', 'Hándicap_asiático']


def main():
    salida = os.path.join(C.PRED, 'predicciones_jornada.csv')
    fx = pd.read_csv(C.RUTA_PROXIMOS, encoding='utf-8-sig', parse_dates=['Fecha']) if os.path.exists(C.RUTA_PROXIMOS) else pd.DataFrame()
    estado = pd.read_csv(C.RUTA_ESTADO, encoding='utf-8-sig')
    if not fx.empty:
        fx = fx[fx.HomeTeam.isin(estado.Equipo) & fx.AwayTeam.isin(estado.Equipo)].reset_index(drop=True)
    if fx.empty:
        pd.DataFrame(columns=COLUMNAS).to_csv(salida, index=False, encoding='utf-8-sig')
        print('No hay próximos partidos publicados (¿parón internacional?). Archivo vacío generado.')
        return

    _, r = predecir(fx, estado=estado)
    M, p = r['M'], r['p_final']
    n, G = M.shape[0], M.shape[1]
    flat = M.reshape(n, -1)
    gl, gv = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
    # Marcador más probable DENTRO del resultado 1X2 más probable (si no, casi siempre sale 1-1)
    cat = np.select([gl > gv, gl == gv], [0, 1], 2).ravel()
    resultado = p.argmax(1)
    mascara = cat[None, :] == resultado[:, None]
    mejor = np.where(mascara, flat, -1).argmax(1)
    over = flat[:, (gl + gv).ravel() > 2.5].sum(1)
    btts = flat[:, ((gl > 0) & (gv > 0)).ravel()].sum(1)

    out = pd.DataFrame({
        'Fecha': fx.Fecha.dt.date, 'Hora': fx.get('Time', ''), 'Local': fx.HomeTeam, 'Visitante': fx.AwayTeam,
        'Prob_1': p[:, 0], 'Prob_X': p[:, 1], 'Prob_2': p[:, 2],
        'xG_Local': r['lam_L'], 'xG_Visitante': r['lam_V'],
        'Marcador_probable': [f'{i // G}-{i % G}' for i in mejor], 'Prob_marcador': flat[np.arange(n), mejor],
        'Prob_Over25': over, 'Prob_BTTS': btts,
    })
    if {'AvgH', 'AvgD', 'AvgA'} <= set(fx.columns) and fx[['AvgH', 'AvgD', 'AvgA']].notna().all().all():
        pm = F.probs_mercado(fx.AvgH.values, fx.AvgD.values, fx.AvgA.values)
        out[['Mercado_1', 'Mercado_X', 'Mercado_2']] = pm
        out[['Valor_1', 'Valor_X', 'Valor_2']] = p - pm
        if {'Avg>2.5', 'Avg<2.5'} <= set(fx.columns):
            po = F.prob_over_mercado(fx['Avg>2.5'], fx['Avg<2.5'])
            out['Mercado_Over25'] = po
            out['Valor_Over25'] = over - po
            out[['Mercado_xG_Local', 'Mercado_xG_Visitante']] = F.goles_mercado(pm, po)
        if 'AHh' in fx.columns:
            out['Hándicap_asiático'] = fx['AHh'].values
    out = out.reindex(columns=COLUMNAS).round(4)
    out.to_csv(salida, index=False, encoding='utf-8-sig')
    print(out[['Fecha', 'Local', 'Visitante', 'Prob_1', 'Prob_X', 'Prob_2', 'Marcador_probable', 'Mercado_1']].to_string(index=False))
    print(f'→ {salida}')


if __name__ == '__main__':
    main()
