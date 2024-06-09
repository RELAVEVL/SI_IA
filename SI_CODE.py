from encodings import utf_8
import pandas as pd
import csv

try:
    # Intentar leer el archivo CSV con varios ajustes
    df = pd.read_csv('./1a_RUIAS.csv', sep=';'  , encoding='utf-8', nrows=8)
    print(df)
    
     
except pd.errors.ParserError as e:
    print(f'Error al leer el archivo CSV: {e}')
    

