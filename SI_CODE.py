# -*- coding: utf-8 -*-
import copy
from encodings import utf_8
from enum import UNIQUE
from pickle import FALSE
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer

# Configuración de pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Leer el archivo CSV
df_RUIAS_os = pd.read_csv('./1a_RUIAS.csv', sep=';', encoding='utf_8')

# Eliminar columnas innecesarias
columns_to_drop = [
    "N", "NOMBRE_ADMINISTRADO", "NOMBRE_UNIDAD_FISCALIZABLE", "NRO_EXPEDIENTE", "NRO_RD", "DETALLE_INFRACCION",
    "TIPO_SANCION", "ACTO_RESUELVE", "FECHA_ACTO", "FECHA_CORTE", "TIPO_DOC", "NORMA_TIPIFICADORA", "MEDIDA_DICTADA",
    "NRO_RD_MULTA", "FECHA_RD_MULTA", "TIPO_RECURSO_IMPUGNATIVO", "CANTIDAD_MULTA", "CANTIDAD_INFRACCIONES", "MULTA_EXPEDIENTE"
]
df_RUIAS_os.drop(columns=columns_to_drop, inplace=True)

# Eliminar filas donde 'DEPARTAMENTO' es "-"
df_RUIAS_os = df_RUIAS_os[df_RUIAS_os['DEPARTAMENTO'] != "-"]

# Filtrar departamentos
departamentos_interes = ["Loreto", "Ucayali", "Madre de Dios", "Amazonas", "San Martín"]
regex_pattern = '|'.join([f'.*{dept}.*' for dept in departamentos_interes])
data_frame_filtrado = df_RUIAS_os[df_RUIAS_os['DEPARTAMENTO'].str.contains(regex_pattern, case=False, na=False)]

# Selección de columnas
columnas_seleccionadas = [
    "ID_DOC_ADMINISTRADO", "SUBSECTOR_ECONOMICO", "DEPARTAMENTO", "PROVINCIA", 
    "DISTRITO", "FECHA_RD", "FECHA_INICIO_SUP", "FECHA_FIN_SUP", "TIPO_INFRACCION"
]

# Binarizamos las categorisa
data_frame_seleccionado = data_frame_filtrado[columnas_seleccionadas]
data_frame_seleccionado["DEPARTAMENTO"].unique()

