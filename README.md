# SI_IA

<h1>“Análisis de patrones de los tipos de infracciones ambientales en la selva peruana” </h1>
<span>Desarrollo de trabajo final del Cursos de Sistemas Inteligentes    </span>
<hr>

Integrantes : </br>
Valer Cierto, Luis Enrique  
Vasquez Pariona, Vania Rosa </br>
Rivera Rimac, Alexander William  

Pasos a seguir : 

<h1>Pre procesamiento<h1>
 
<h2>1.Delimitamos la zona de estudio </h2>

-Loreto</br>
-Ucayali</br>
-Madre de Dios</br>
-Amazonas</br>
-San Martín</br>
 
<h2>2.analisamos el datset para borrar o verificar las columnas de interes para el procesamiento </h2>
<strong>Columnas que se eliminarian </strong>
-N
-NOMBRE_ADMINISTRADO
-NOMBRE_UNIDAD_FISCALIZABLE
-NRO_EXPEDIENTE
-NRO_RD
-DETALLE_INFRACCION
-TIPO_SANCION
-ACTO_RESUELVE
-FECHA_ACTO
-FECHA_CORTE
-TIPO_DOC
-NORMA_TIPIFICADORA
-MEDIDA_DICTADA
-NRO_RD_MULTA
-FECHA_RD_MULTA
-TIPO_RECURSO_IMPUGNATIVO
-CANTIDAD_MULT
-CANTIDAD_INFRACCIONE
-MULTA_EXPEDIENT

-De las columnas departamento,provincia , distrito 
2164 filas estan vacias  - > se procedera a eliminar estas filas 

- Hay 18 tipos de infraciones 


<strong> Columnas que quedarian </strong>
ID_DOC_ADMINISTRADO <br>
SUBSECTOR_ECONOMICO <br>
DEPARTAMENTO<br>
PROVINCIA<br>
DISTRITO<br>
FECHA_RD<br>
FECHA_INICIO_SUP<br>
FECHA_FIN_SUP<br>
TIPO_INFRACCION<br>


<h2>3.Generamos una nueva bd </h2>
-Para no perder la bd original, creamos un segundo, que es el que modificaremos

Buscaremos la cantidad optima de clusters a formar 
Calculando que tan similares son los individuos dentro del cluster 
usaremos metodo de codo de jambu 



Buscaremos un valor donde deje de disminuir de manera drastica 

- Debemos identificar la cantidad e cluster a trabajar, esto lo apreciaremos cuando veamos en la grafica un 
 
