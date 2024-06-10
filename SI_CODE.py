# -*- coding: utf-8 -*- 
from encodings import utf_8
from numpy import indices
import pandas as pd
import csv
import os
import io 
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder , LabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer 



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Nos situamos en el directorio de los datos
os.chdir("C:/Users/Luis/Downloads/code/workspace/SI/SI_IA")
#leemos el csv por medio de pandas 
df_RUIAS_os = pd.read_csv('./1a_RUIAS.csv', sep=';',encoding='utf_8')

del df_RUIAS_os["N"]
del df_RUIAS_os["NOMBRE_ADMINISTRADO"]
del df_RUIAS_os["NOMBRE_UNIDAD_FISCALIZABLE"]
del df_RUIAS_os["NRO_EXPEDIENTE"]
del df_RUIAS_os["NRO_RD"]
del df_RUIAS_os["DETALLE_INFRACCION"]
del df_RUIAS_os["TIPO_SANCION"]
del df_RUIAS_os["ACTO_RESUELVE"]
del df_RUIAS_os["FECHA_ACTO"]
del df_RUIAS_os["FECHA_CORTE"]

# Eliminar filas donde 'DEPARTAMENTO' es "-"
df_RUIAS_os= df_RUIAS_os[df_RUIAS_os['DEPARTAMENTO'] != "-"]
#  departamentos a filtrar
departamentos_interes = ["Loreto", "Ucayali", "Madre de Dios", "Amazonas", "San Martín"]
# Crear una expresión regular para buscar los departamentos ignorando caracteres adicionales
regex_pattern = '|'.join([f'.*{dept}.*' for dept in departamentos_interes])


#data_frame_seleccionado = data_frame_filtrado[["DEPARTAMENTO", "ID_DOC_ADMINISTRADO", "TIPO_DOC"]]

"""
verificando este filtro de zona de interes , podemos apreciar que todos son de tipo RUC 
y que no seria necesario tener TIPO_DOC , PROVINCIA, DISTRITO , NORMA_TIPIFICADORA , 
"""
del df_RUIAS_os["TIPO_DOC"]
#del df_RUIAS_os["PROVINCIA"] 
#del df_RUIAS_os["DISTRITO"] 
del df_RUIAS_os["NORMA_TIPIFICADORA"] 
del df_RUIAS_os["MEDIDA_DICTADA"] 
del df_RUIAS_os["NRO_RD_MULTA"] 
del df_RUIAS_os["FECHA_RD_MULTA"] 
del df_RUIAS_os ["TIPO_RECURSO_IMPUGNATIVO"] 
del df_RUIAS_os ["CANTIDAD_MULTA"]
del df_RUIAS_os ["CANTIDAD_INFRACCIONES"]
del df_RUIAS_os ["MULTA_EXPEDIENTE"]

# Filtrar el DataFrame para que solo contenga filas de los departamentos especificados
data_frame_filtrado = df_RUIAS_os[df_RUIAS_os['DEPARTAMENTO'].str.contains(regex_pattern, case=False, na=False)]


# Mostrar el DataFrame filtrado
print(data_frame_filtrado.head(10) )
#data_frame_filtrado.to_csv('NRUIAS.csv')

#leemos la nueva bd 
df_RUIAS_os = pd.read_csv('./NRUIAS.csv', sep=';',encoding='utf_8')
"""
#cambiams los datos string true false
class BinarizadorCategorico(preprocessing.LabelBinarizer):
    def fit(self , x , y=None):
        super (BinarizadorCategorico,self).fit(x)
    def transform(self , x , y=None):
        return super(BinarizadorCategorico,self).transform(x)
    def fit_transform(self , x , y=None):
        return super(BinarizadorCategorico,self).fit(x).transform(x)
class ColumnExtractor(TransformerMixin):
    def __init__(self,columns):
        self.columns = columns
    def transform (self , x , **transform_params):
        return x [self.columns].to_numpy()
    def fit(self, x , y=None , **fit_params) :
        return self
    """
    