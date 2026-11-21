"""
6. WEB ESTÁTICA (GitHub Pages) → docs/index.html
Embebe los CSV de Predicciones/ como JSON en una única página sin dependencias externas.
"""
import json
import os
from datetime import datetime, timezone
import pandas as pd
import config as C


def leer(nombre):
    ruta = os.path.join(C.PRED, nombre)
    if not os.path.exists(ruta) or os.path.getsize(ruta) == 0:
        return []
    df = pd.read_csv(ruta, encoding='utf-8-sig')
    return json.loads(df.to_json(orient='records', force_ascii=False))


def main():
    pos = pd.read_csv(os.path.join(C.PRED, 'probabilidades_posicion.csv'), encoding='utf-8-sig', index_col=0)
    with open(os.path.join(C.PRED, 'metricas_modelo.json'), encoding='utf-8') as f:
        metricas = json.load(f)
    datos = {
        'actualizado': datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M UTC'),
        'n_sims': C.N_SIMULACIONES,
        'resumen': leer('simulacion_temporada.csv'),
        'tabla': leer('clasificacion_actual.csv'),
        'jornada': leer('predicciones_jornada.csv'),
        'posiciones': {'equipos': pos.index.tolist(), 'matriz': pos.values.round(4).tolist()},
        'metricas': metricas,
        'backtest_resumen': leer('backtest_resumen.csv'),
        'backtest_partidos': leer('backtest_partidos.csv'),
    }
    with open(os.path.join(os.path.dirname(__file__), 'plantilla_web.html'), encoding='utf-8') as f:
        html = f.read().replace('/*__DATOS__*/null', json.dumps(datos, ensure_ascii=False))
    with open(os.path.join(C.DOCS, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'→ docs/index.html ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
