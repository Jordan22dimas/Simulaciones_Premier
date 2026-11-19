"""
(OPCIONAL) ENRIQUECIMIENTO CON FLASHSCORE — stats avanzadas (xG, posesión, pases, grandes ocasiones...)
Adaptado de Simulaciones_Mundial/code/Scrapper.py.

Requisito: abre Chrome con depuración remota antes de ejecutarlo:
    chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\\chrome-scraper"

Uso:
    python code/Scrapper_FlashScore.py              → temporada actual
    python code/Scrapper_FlashScore.py 2024-2025    → una temporada pasada

Es incremental: salta las URLs ya guardadas en Data/flashscore_premier.csv.
Data_Cleaning.py usa automáticamente su xG real donde football-data no lo tiene.
"""
import os
import random
import sys
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import config as C

BASE = 'https://www.flashscore.es/futbol/inglaterra/premier-league{sufijo}/resultados/'


def conectar():
    opciones = Options()
    opciones.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    return webdriver.Chrome(options=opciones)


def recoger_urls(driver, temporada=None):
    """Abre la página de resultados, pulsa 'Mostrar más partidos' hasta el final y devuelve las URLs."""
    driver.get(BASE.format(sufijo=f'-{temporada}' if temporada else ''))
    time.sleep(4)
    while True:
        botones = driver.find_elements(By.XPATH, "//*[contains(text(),'Mostrar más partidos')]")
        if not botones:
            break
        try:
            driver.execute_script('arguments[0].click();', botones[0])
            time.sleep(random.uniform(2, 3.5))
        except Exception:
            break
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    urls = []
    for a in soup.select("a[href*='/partido/']"):
        href = a['href'].split('?')[0].rstrip('/')
        if href.startswith('/'):
            href = 'https://www.flashscore.es' + href
        mid = a.get('href').split('mid=')[-1] if 'mid=' in a.get('href', '') else None
        url = href.replace('/resumen', '') + '/resumen/estadisticas/general/' + (f'?mid={mid}' if mid else '')
        if url not in urls:
            urls.append(url)
    return urls


def extraer_partido(url, driver):
    """Misma lógica que el scraper del Mundial: fecha, equipos, marcador y filas de stats (local|categoría|visitante)."""
    datos = {'Fecha': None, 'URL': url, 'Equipo_Local': None, 'Equipo_Visitante': None, 'Resultado': None}
    driver.get(url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    f = soup.find(class_=lambda x: x and 'startTime' in x)
    if f: datos['Fecha'] = f.text.strip()
    eqs = []
    for a in soup.find_all('a', class_=lambda x: x and 'participantName' in x):
        n = a.text.strip()
        if n and n not in eqs: eqs.append(n)
    if len(eqs) >= 2: datos['Equipo_Local'], datos['Equipo_Visitante'] = eqs[:2]
    m = soup.find(class_=lambda x: x and 'detailScore' in x)
    if m: datos['Resultado'] = m.text.replace('\n', '').strip()
    for div in soup.find_all('div'):
        hijos = div.find_all('div', recursive=False)
        if len(hijos) == 3:
            l, cat, v = (h.text.strip() for h in hijos)
            if l and cat and v and any(c.isalpha() for c in cat) and len(cat) < 30:
                col = cat.replace(' ', '_')
                datos[f'{col}_Local'], datos[f'{col}_Visitante'] = l, v
    return datos


def main():
    temporada = sys.argv[1] if len(sys.argv) > 1 else None
    previo = pd.read_csv(C.RUTA_FLASHSCORE, encoding='utf-8-sig') if os.path.exists(C.RUTA_FLASHSCORE) else pd.DataFrame()
    hechas = set(previo['URL']) if 'URL' in previo else set()

    driver = conectar()
    urls = [u for u in recoger_urls(driver, temporada) if u not in hechas]
    print(f'{len(urls)} partidos nuevos por descargar')
    nuevos = []
    for i, url in enumerate(urls, 1):
        print(f'[{i}/{len(urls)}] {url}')
        try:
            d = extraer_partido(url, driver)
            if d['Equipo_Local']:
                nuevos.append(d)
        except Exception as e:
            print(f'  ! {e}')
        if i % 20 == 0 and nuevos:   # guardado parcial
            pd.concat([previo, pd.DataFrame(nuevos)]).to_csv(C.RUTA_FLASHSCORE, index=False, encoding='utf-8-sig')
        time.sleep(random.uniform(3, 6))   # pausa de cortesía
    if nuevos:
        pd.concat([previo, pd.DataFrame(nuevos)]).to_csv(C.RUTA_FLASHSCORE, index=False, encoding='utf-8-sig')
    print(f'→ {C.RUTA_FLASHSCORE}: {len(previo) + len(nuevos)} partidos')


if __name__ == '__main__':
    main()
