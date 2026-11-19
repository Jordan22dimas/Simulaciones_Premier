"""Mapeo de nombres de equipos: FlashScore (es) → football-data.co.uk."""
import unicodedata

FLASHSCORE_A_FD = {
    'arsenal': 'Arsenal', 'aston villa': 'Aston Villa', 'bournemouth': 'Bournemouth',
    'brentford': 'Brentford', 'brighton': 'Brighton', 'chelsea': 'Chelsea',
    'coventry': 'Coventry', 'crystal palace': 'Crystal Palace', 'everton': 'Everton',
    'fulham': 'Fulham', 'hull': 'Hull', 'hull city': 'Hull', 'ipswich': 'Ipswich',
    'leeds': 'Leeds', 'leeds united': 'Leeds', 'leicester': 'Leicester', 'liverpool': 'Liverpool',
    'manchester city': 'Man City', 'man city': 'Man City',
    'manchester utd': 'Man United', 'manchester united': 'Man United', 'man utd': 'Man United',
    'newcastle': 'Newcastle', 'nottingham': "Nott'm Forest", 'nottingham forest': "Nott'm Forest",
    'sunderland': 'Sunderland', 'tottenham': 'Tottenham', 'west ham': 'West Ham',
    'wolves': 'Wolves', 'wolverhampton': 'Wolves', 'burnley': 'Burnley',
    'southampton': 'Southampton', 'sheffield utd': 'Sheffield United', 'luton': 'Luton',
    'west brom': 'West Brom', 'watford': 'Watford', 'norwich': 'Norwich',
}


def _clave(nombre):
    s = unicodedata.normalize('NFKD', str(nombre)).encode('ascii', 'ignore').decode().lower().strip()
    return s.replace(' fc', '').replace('afc ', '').strip()


def normalizar_flashscore(nombre):
    return FLASHSCORE_A_FD.get(_clave(nombre), nombre)
