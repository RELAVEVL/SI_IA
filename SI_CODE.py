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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Nos situamos en el directorio de los datos
# os.chdir("C:/Users/Luis/Downloads/code/workspace/SI/SI_IA")
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

# Selección de columnas
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
"""En esta parte binarizamos las categorisa """


data_frame_seleccionado = data_frame_filtrado[columnas_seleccionadas]
data_frame_seleccionado["DEPARTAMENTO"].unique()

# Binarización de categorías
def binarize_column(data_frame, column_name):
    unique_values = data_frame[column_name].nunique()
    if unique_values > 2:
        encoder = OneHotEncoder()
        encoded_columns = encoder.fit_transform(data_frame[[column_name]]).toarray()
        encoded_columns_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out([column_name]))
    else:
        encoder = LabelBinarizer()
        encoded_columns = encoder.fit_transform(data_frame[column_name])
        encoded_columns_df = pd.DataFrame(encoded_columns, columns=[f"{column_name}_BIN"])
    return encoded_columns_df



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


# Binarizar la columna 'DEPARTAMENTO'
one_hot_departamento = OneHotEncoder()
departamento_bin = one_hot_departamento.fit_transform(data_frame_seleccionado[['DEPARTAMENTO']]).toarray()
departamento_bin_df = pd.DataFrame(departamento_bin, columns=one_hot_departamento.get_feature_names_out(['DEPARTAMENTO']))
data_frame_seleccionado = data_frame_seleccionado.join(departamento_bin_df)

# Binarizar la columna 'TIPO_INFRACCION'
one_hot_infraccion = OneHotEncoder()
infraccion_bin = one_hot_infraccion.fit_transform(data_frame_seleccionado[['TIPO_INFRACCION']]).toarray()
infraccion_bin_df = pd.DataFrame(infraccion_bin, columns=one_hot_infraccion.get_feature_names_out(['TIPO_INFRACCION']))

# Concatenar el DataFrame original con las nuevas columnas binarizadas de 'TIPO_INFRACCION'
data_frame_seleccionado = data_frame_seleccionado.join(infraccion_bin_df)


# Extraer columnas específicas
columns_to_extract = ["ID_DOC_ADMINISTRADO"] + one_hot_departamento.get_feature_names_out(['DEPARTAMENTO']).tolist() + (["SUBSECTOR_ECONOMICO_BIN"] if unique_values == 2 else one_hot.get_feature_names_out(['SUBSECTOR_ECONOMICO']).tolist()) + one_hot_infraccion.get_feature_names_out(['TIPO_INFRACCION']).tolist()

column_extractor = ColumnExtractor(columns=columns_to_extract)
extracted_columns = column_extractor.transform(data_frame_seleccionado)


# Convertir el resultado a un DataFrame para facilidad de visualización
df_final = pd.DataFrame(extracted_columns, columns=columns_to_extract) 
# Mostrar las primeras 10 filas del DataFrame resultante
print("DataFrame Final (primeras 10 filas):")
print(df_final.head(10))
df_final.dropna(inplace=True)

"""Normalizacion de datos """
# Normalización
# Simple feature scaling
df_final['ID_DOC_ADMINISTRADO'] = df_final['ID_DOC_ADMINISTRADO'] / df_final['ID_DOC_ADMINISTRADO'].max()

df_final.describe()

"""En esta parte verificamos la cantidad optima de cluster
segun el resultado es k =2 pero esto se modificara xq no tomamos todos los datos a anlizar 
"""
#Busqueda de cantidad optima de clusters 

wcss = []

for i in range (1, 11):
    kmeans =KMeans(n_clusters=i, max_iter = 300)
    kmeans.fit(df_final)
    wcss.append(kmeans.inertia_)

#Aplico K-means a la base de datos
#Graficando los resultados de WCSS para formar el Codo de Jambú
plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Codo de Jambú')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

#segun la grafica el numero optimo  K =3

"""AQui ejecutaremos el kmeans y veremos los cluster 
""" 
 
#aplicando el metodo kmeans a la bd  
kmeans= KMeans(n_clusters=3 , max_iter=3000).fit(df_final) #crea el modelo 
centroids=kmeans.cluster_centers_
print(centroids)

# Convertir el DataFrame a array de NumPy para la gráfica
df_final_array = df_final.to_numpy()

plt.scatter(df_final_array[:,0], df_final_array[:,1], c=kmeans.labels_.astype(float), s=50)
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='*', s=50)
plt.show()


# Agregar la columna de clusters al DataFrame original
df_final['KMeans_clusters'] = kmeans.labels_

# Guardar el DataFrame con los clusters en un nuevo archivo CSV
df_final.to_csv('NRUIAS.csv', sep=';', index=False)

print("DataFrame con clusters añadido:")
print(df_final.head())


