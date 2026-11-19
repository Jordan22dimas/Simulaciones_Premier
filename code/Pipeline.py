"""
Ejecuta el pipeline completo de principio a fin.
    python code/Pipeline.py            → reentreno rápido (hiperparámetros guardados) — uso diario
    python code/Pipeline.py --completo → búsqueda de hiperparámetros + validación (~7 min)
"""
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import Descarga_Datos, Data_Cleaning, Modelo, Prediccion_Jornada, Simulacion_Temporada, Generar_Web

PASOS = [
    ('1. Descarga de datos', Descarga_Datos.main),
    ('2. Limpieza y features', Data_Cleaning.main),
    ('3. Modelo', Modelo.main),
    ('4. Próxima jornada', Prediccion_Jornada.main),
    ('5. Monte Carlo temporada', Simulacion_Temporada.main),
    ('6. Web', Generar_Web.main),
]

if __name__ == '__main__':
    if '--completo' not in sys.argv:
        sys.argv.append('--rapido')
    for nombre, paso in PASOS:
        t = time.time()
        print(f'\n{"=" * 60}\n{nombre}\n{"=" * 60}')
        paso()
        print(f'({time.time() - t:.0f} s)')
