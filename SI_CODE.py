# -*- coding: utf-8 -*- 
import pandas as pd
"""
import csv
import os
"""
try:
    # Intentar leer el archivo CSV con varios ajustes
    df_RUIAS = pd.read_csv('./1a_RUIAS.csv', sep=';'  , nrows=8)
    print( df_RUIAS )

    for col in  df_RUIAS.columns:
        print(col)    

except pd.errors.ParserError as e:
    print(f'Error al leer el archivo CSV: {e}')


