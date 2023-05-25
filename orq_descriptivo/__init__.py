import findspark
findspark.init()

# Librerías básicas
import pandas as pd
import datetime
import settings as sts

from gensim.models import LdaModel
from gensim.models import Phrases
from gensim.corpora import Dictionary

# Paquetes del proyecto
from modules.etl.conexiones import Conexion
from modules.etl.etl_secop import *
from modules.etl.etl_dane_pob import *
from modules.etl.etl_dane_ipm import *
from modules.modelos.lda  import *



#************************************************************************************************************************************************
# Conexion a Spark

# Crear una instancia de la clase Conexion
conexion = Conexion()
conexion.init_spark_session()
spark_session = conexion.spark_session
spark_session.conf.set("spark.sql.repl.eagerEval.enabled", True) # para generar mejor formato de tablas
spark_session

# Cargar base de datos SECOP I en Spark DataFrame
df_secop_crudo = spark_session.read.csv(sts.crudos+'SECOP.csv', header=True, schema = sts.schema_secop)
# Cargar información DANE y equivalencia del Departamento con SECOP
df_dpto_reg = spark_session.read.option("delimiter", ";").option("header", True).csv(sts.crudos+"Regiones_Departamentos.csv")
# Se carga el archivo csv con los datos de los contratos
datos = pd.read_csv(sts.datamart+'df_secop_obra.csv', encoding='utf-8', low_memory=False)
pobl_2010_2019 = pd.read_excel(sts.crudos+'DCD-area-proypoblacion-dep-2005-2019.xlsx')
pobl_2020_2030 = pd.read_excel(sts.crudos+'DCD-area-proypoblacion-dep-2020-2050-ActPostCOVID-19.xlsx')
ipm_departamentos = pd.read_excel(sts.crudos+"anexo_dptal_pobreza_multidimensional_21.xls", sheet_name=sts.hojas[1], skiprows=range(14), nrows=33)
ipm_regiones = pd.read_excel(sts.crudos+"anexo_dptal_pobreza_multidimensional_21.xls", sheet_name=sts.hojas[2], skiprows=range(14), nrows=9)


# Intanciando Clases
etl=SECOP_ETL(spark_session)
poblacion_dane = PoblacionDANE(pobl_2010_2019, pobl_2020_2030)
ipm=IPM_DANE()

# Constructor
if __name__=='__main__':
    etl.cargar_df_secop(etl,df_secop_crudo,sts.columnas_drop,df_dpto_reg,sts.old_new_names_dict,sts.cols_integer,sts.cols_date,sts.cols_str,sts.cols_money)
    spark_session.stop()
    poblacion_dane.poblacion(poblacion_dane,pobl_2010_2019,pobl_2020_2030)
    ipm.procesar_datos_ipm(ipm,sts.ruta_archivo_21, sts.ruta_archivo2, 
                           sts.crudos,ipm_departamentos,sts.columnas, sts.tipos, sts.anios,
                           ipm_regiones, sts.Area_dicc, sts.Region_dict, 
                           sts.region_dict, sts.hojas[4], sts.skiprows_dict, sts.nrows_dict,sts.hojas[5],
                           sts.Area_dicc_zona, sts.hojas[6], sts.skiprows_dict1, sts.hojas[7], 
                           sts.skiprows_dict2, sts.anios2, sts.Area_dicc1, sts.skiprows_dict3,sts.departamentos_tupla1)
    execute_analysis(datos)

