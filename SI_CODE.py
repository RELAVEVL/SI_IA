# -*- coding: utf-8 -*-
from encodings import utf_8
from enum import UNIQUE
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Nos situamos en el directorio de los datos
#os.chdir("C:/Users/Luis/Downloads/code/workspace/SI/SI_IA")
# Leemos el csv por medio de pandas
df_RUIAS_os = pd.read_csv('./1a_RUIAS.csv', sep=';', encoding='utf_8')
#df_RUIAS_os.info()
#df_RUIAS_os.describe()
# Eliminamos columnas no deseadas
columnas_borrar = [
    "N", "NOMBRE_ADMINISTRADO", "NOMBRE_UNIDAD_FISCALIZABLE", "NRO_EXPEDIENTE", "NRO_RD", "DETALLE_INFRACCION", 
    "TIPO_SANCION", "ACTO_RESUELVE", "FECHA_ACTO", "FECHA_CORTE", "TIPO_DOC", "NORMA_TIPIFICADORA", "MEDIDA_DICTADA",
    "NRO_RD_MULTA", "FECHA_RD_MULTA", "TIPO_RECURSO_IMPUGNATIVO", "CANTIDAD_MULTA", "CANTIDAD_INFRACCIONES", "MULTA_EXPEDIENTE"
]
df_RUIAS_os.drop(columns=columnas_borrar, inplace=True)

# Eliminar filas donde 'DEPARTAMENTO' es "-"
df_RUIAS_os = df_RUIAS_os[df_RUIAS_os['DEPARTAMENTO'] != "-"]

# Departamentos a filtrar
departamentos_interes = ["Loreto", "Ucayali", "Madre de Dios", "Amazonas", "San Martín"]
# Crear una expresión regular para buscar los departamentos ignorando caracteres adicionales
regex_pattern = '|'.join([f'.*{dept}.*' for dept in departamentos_interes])

# Filtrar el DataFrame para que solo contenga filas de los departamentos especificados
data_frame_filtrado = df_RUIAS_os[df_RUIAS_os['DEPARTAMENTO'].str.contains(regex_pattern, case=False, na=False)]
columnas_seleccionadas = [
    "ID_DOC_ADMINISTRADO", 
    "SUBSECTOR_ECONOMICO", 
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
    "FECHA_RD",
    "FECHA_INICIO_SUP",
    "FECHA_FIN_SUP",
    "TIPO_INFRACCION"
]

data_frame_seleccionado = data_frame_filtrado[columnas_seleccionadas]
data_frame_seleccionado["DEPARTAMENTO"].unique()

# Cambiar los datos string true false
class BinarizadorCategorico(preprocessing.LabelBinarizer):
    def fit(self, x, y=None):
        return super(BinarizadorCategorico, self).fit(x)
    def transform(self, x, y=None):
        return super(BinarizadorCategorico, self).transform(x)
    def fit_transform(self, x, y=None):
        return super(BinarizadorCategorico, self).fit(x).transform(x)

class ColumnExtractor(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def transform(self, x, **transform_params):
        return x[self.columns].to_numpy()
    def fit(self, x, y=None, **fit_params):
        return self

# Verificar si hay más de dos categorías en la columna
unique_values = data_frame_seleccionado['SUBSECTOR_ECONOMICO'].nunique()
if unique_values > 2:
    one_hot = OneHotEncoder()
    encoded_columns = one_hot.fit_transform(data_frame_seleccionado[['SUBSECTOR_ECONOMICO']]).toarray()
    encoded_columns_df = pd.DataFrame(encoded_columns, columns=one_hot.get_feature_names_out(['SUBSECTOR_ECONOMICO']))
    data_frame_seleccionado = data_frame_seleccionado.join(encoded_columns_df)
else:
    binarizador = BinarizadorCategorico()
    data_frame_seleccionado['SUBSECTOR_ECONOMICO_BIN'] = binarizador.fit_transform(data_frame_seleccionado['SUBSECTOR_ECONOMICO'])

# Extraer columnas específicas
columns_to_extract = ["DEPARTAMENTO", "ID_DOC_ADMINISTRADO"] + (["SUBSECTOR_ECONOMICO_BIN"] if unique_values == 2 else one_hot.get_feature_names_out(['SUBSECTOR_ECONOMICO']).tolist())
column_extractor = ColumnExtractor(columns=columns_to_extract)
extracted_columns = column_extractor.transform(data_frame_seleccionado)

# Convertir el resultado a un DataFrame para facilidad de visualización
df_final = pd.DataFrame(extracted_columns, columns=columns_to_extract)

# Mostrar las primeras 10 filas del DataFrame resultante
print("DataFrame Final (primeras 10 filas):")
print(df_final.head(10))


#crear nueva bd 
#df_final.to_csv('NRUIAS.csv')

#leemos la nueva bd 
#df_RUIAS = pd.read_csv('./NRUIAS.csv', sep=';',encoding='utf_8')
