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
    url_resultados = BASE.format(sufijo=f'-{temporada}' if temporada else '')
    print(f'Abriendo {url_resultados} ...', flush=True)
    driver.get(url_resultados)
    time.sleep(4)

    # Banner de cookies (OneTrust): si aparece, se acepta para que no tape la página
    for b in driver.find_elements(By.ID, 'onetrust-accept-btn-handler'):
        try:
            b.click(); time.sleep(1)
        except Exception:
            pass

    contar = lambda: len(driver.find_elements(By.CSS_SELECTOR, "a[href*='/partido/futbol/']"))
    n = contar()
    print(f'  {n} partidos visibles', flush=True)
    fallos_seguidos = 0
    for intento in range(60):                      # límite de seguridad (60 clics ≈ 1.500 partidos)
        botones = [b for b in driver.find_elements(By.XPATH, "//*[contains(text(),'Mostrar más partidos')]")
                   if b.is_displayed()]
        if not botones:
            # el botón puede tardar en reaparecer tras cargar un bloque: se comprueba otra vez
            time.sleep(3)
            botones = [b for b in driver.find_elements(By.XPATH, "//*[contains(text(),'Mostrar más partidos')]")
                       if b.is_displayed()]
            if not botones:
                print('  No hay más botón "Mostrar más partidos": lista completa', flush=True)
                break
        try:
            driver.execute_script('arguments[0].scrollIntoView({block: "center"}); arguments[0].click();', botones[0])
        except Exception as e:
            print(f'  No se pudo pulsar el botón ({e.__class__.__name__}); reintento', flush=True)
        # Espera hasta 15 s a que aparezcan partidos nuevos (las temporadas antiguas cargan lento)
        nuevo = n
        for _ in range(15):
            time.sleep(1)
            nuevo = contar()
            if nuevo > n:
                time.sleep(1)                      # deja terminar de pintar el bloque
                nuevo = contar()
                break
        print(f'  Clic {intento + 1}: {nuevo} partidos visibles', flush=True)
        if nuevo <= n:
            fallos_seguidos += 1
            if fallos_seguidos >= 3:               # 3 clics seguidos sin cargar nada → fin
                print('  Tres clics sin partidos nuevos: doy la lista por completa', flush=True)
                break
        else:
            fallos_seguidos = 0
            n = nuevo
        time.sleep(random.uniform(1, 2))
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    urls = []
    # Solo fútbol y, si es posible, solo las filas de la lista de resultados (no los widgets laterales
    # de "destacados", que traen tenis, baloncesto u otras ligas)
    enlaces = soup.select("[class*='event__match'] a[href*='/partido/futbol/']") \
        or soup.select("a[href*='/partido/futbol/']")
    for a in enlaces:
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
            # el nombre de la estadística no puede llevar números (evita filas como
            # "2.18xGOT enfrentados0.30", que crearían una columna nueva por partido)
            if l and cat and v and any(c.isalpha() for c in cat) and not any(c.isdigit() for c in cat) \
                    and len(cat) < 30:
                col = cat.replace(' ', '_')
                datos[f'{col}_Local'], datos[f'{col}_Visitante'] = l, v
    return datos


def main():
    temporada = sys.argv[1] if len(sys.argv) > 1 else None
    previo = pd.read_csv(C.RUTA_FLASHSCORE, encoding='utf-8-sig') if os.path.exists(C.RUTA_FLASHSCORE) else pd.DataFrame()
    hechas = set(previo['URL']) if 'URL' in previo else set()

    print('Conectando con Chrome en el puerto 9222...', flush=True)
    driver = conectar()
    print(f'  Conectado. Ya guardados previamente: {len(hechas)} partidos', flush=True)
    urls = [u for u in recoger_urls(driver, temporada) if u not in hechas]
    print(f'{len(urls)} partidos nuevos por descargar')
    nuevos = []
    guardar = lambda: nuevos and pd.concat([previo, pd.DataFrame(nuevos)]).to_csv(
        C.RUTA_FLASHSCORE, index=False, encoding='utf-8-sig')
    try:
        for i, url in enumerate(urls, 1):
            print(f'[{i}/{len(urls)}] {url}', flush=True)
            try:
                d = extraer_partido(url, driver)
                if d['Equipo_Local']:
                    nuevos.append(d)
            except Exception as e:
                msg = str(e).split('\n')[0]
                if 'invalid session' in msg or 'disconnected' in msg or 'no such window' in msg:
                    print('\n  ✖ Se ha perdido la conexión con Chrome (ventana cerrada o colgada).')
                    print('    Vuelve a abrir Chrome en el puerto 9222 y relanza: continuará donde lo dejó.')
                    break
                print(f'  ! {msg}')
            if i % 10 == 0:          # guardado parcial
                guardar()
            time.sleep(random.uniform(3, 6))   # pausa de cortesía
    except KeyboardInterrupt:
        print('\n  Detenido con Ctrl+C. Guardando lo descargado...')
    finally:
        guardar()
        print(f'→ {C.RUTA_FLASHSCORE}: {len(previo) + len(nuevos)} partidos guardados en total')


if __name__ == '__main__':
    main()
