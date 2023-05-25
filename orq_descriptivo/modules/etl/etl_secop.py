import findspark
findspark.init()
import settings as sts
import sys

# Librerías básicas
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import udf
from pyspark.sql.functions import year, col


class SECOP_ETL:
    
    
    def __init__(self, spark_session):
        self.spark = spark_session
             
    
    def eliminar_columnas_innecesarias(self,df,columnas_drop):
           
        return df.drop(*columnas_drop)
    
    def rename_columns(self, df, old_new_names_dict):
        for old_name, new_name in old_new_names_dict.items():
            df = df.withColumnRenamed(old_name, new_name)
        return df
    
    
    def impute_with_zero(self,df, cols):
        df = df.fillna(0, subset = cols)
        return df

    def impute_with_date(self,df, cols):
        df = df.fillna('1900-01-01 00:00:00', subset = cols)
        return df

    def impute_with_string(self,df, cols):
        df = df.fillna('No Definido', subset = cols)
        return df

    def remove_commas(self, df, cols):
        quitar_coma = udf(lambda x: x.replace(',', ''))
        for col in cols:
            df = df.withColumn(col + '_Int', quitar_coma(col))
        return df
    
    def remove_decimal_and_convert_to_int(self,df, cols):
        for col in cols:
            df = df.withColumn(col + "_Int", regexp_extract(col, "[0-9]+", 0).cast("int"))
            df = df.drop(col)
            df = df.withColumnRenamed(col + "_Int", col)
        return df



    def procesar_fechas_contrato(self,df):
        # Agregar columnas indicando el año de inicio y el año de fin del contrato y dejar en formato int
        df = df.withColumn("Anno_Inicio_Contrato", year('Fecha_Ini_Ejec_Contrato')). \
                            withColumn("Anno_Fin_Contrato", year('Fecha_Fin_Ejec_Contrato'))

        columnas_drop = ('Fecha_Ini_Ejec_Contrato', 'Fecha_Fin_Ejec_Contrato')
        df = df.drop(*columnas_drop)

        df = df.fillna(0, subset=['Anno_Inicio_Contrato', 'Anno_Fin_Contrato'])

        df = df.withColumn("Anno_Inicio_Contrato", df['Anno_Inicio_Contrato'].cast(IntegerType())). \
                            withColumn("Anno_Fin_Contrato", df['Anno_Fin_Contrato'].cast(IntegerType()))

        return df
    
    def filtrar_base(self,df, conteo_minimo=2000):
        

        # Calcular el conteo por grupo
        conteo_por_grupo = df.groupBy('Objeto_Contratar').agg(count('UID').alias('conteo'))

        # Filtrar los grupos que cumplen con el conteo mínimo
        grupos_filtrados = conteo_por_grupo.filter(col('conteo') >= conteo_minimo)

        # Aplicar el filtro utilizando el método join
        datos_filtrados = df.join(grupos_filtrados, 'Objeto_Contratar', 'inner')

        return datos_filtrados
                

    
    def crear_diccionarios(self,df_dpto_reg, df_secop):
        # Crear diccionarios para cambiar departamento (según descripción DANE) y agregar Región DANE
        df_dpto_reg_pd = df_dpto_reg.toPandas()
        df_dpto_reg_pd.set_index('Dpto_SECOP', inplace=True)
        dict_dpto = df_dpto_reg_pd['Dpto_DANE'].to_dict()  # Diccionario Dpto Secop -> DANE
        dict_reg = df_dpto_reg_pd['Region_DANE'].to_dict()  # Diccionario Dpto DANE -> Región DANE

        # Crear función para traducir según diccionario
        def traductor(dictionary):
            return udf(lambda col: dictionary.get(col), StringType())

        # Crear columna con la región correspondiente según clasificación del DANE
        df_secop = df_secop.withColumn("Region", traductor(dict_reg)("ID_Departamento"))

        # Modificar columna Departamento_Entidad con el nombre del departamenteo según el DANE
        df_secop = df_secop.withColumn("Departamento_Entidad_DANE", traductor(dict_dpto)("ID_Departamento"))

        # Eliminar columnas original Departamento Entidad y actualizar nombre
        columnas_drop = ['ID_Departamento']
        df_secop = df_secop.drop(*columnas_drop)
        
        return df_secop
    
    def filtrar_datos(self,df):
        df = df.where((df.Estado_Proceso == 'Celebrado') | (df.Estado_Proceso == 'Liquidado') | \
                    (df.Estado_Proceso == 'Terminado sin Liquidar'))
        df = df.where((df.Moneda == 'No Definido') | (df.Moneda == 'Peso Colombiano'))
        df = df.where((df.Anno_Cargue_SECOP != 0))
        df = df.where((df.Tipo_Contrato != 'Consultoría') | (df.Tipo_Contrato != 'Crédito') | \
                    (df.Tipo_Contrato != 'Fiducia') | (df.Tipo_Contrato != 'Interventoría'))
        df = df.where((df.Departamento_Entidad_DANE != 'No Definido') | (df.Departamento_Entidad_DANE != 'Colombia'))
        df = df.where((df.Tipo_Proceso != 'Concurso de Méritos Abierto') | (df.Tipo_Proceso != 'Concurso de Méritos con Lista Corta') | \
                    (df.Tipo_Proceso != 'Concurso de Méritos con Lista Multiusos') | (df.Tipo_Proceso != 'Concurso de diseño Arquitectónico') | \
                    (df.Tipo_Proceso != 'Iniciativa Privada sin recursos públicos') | (df.Tipo_Proceso != 'Lista Multiusos') | \
                    (df.Tipo_Proceso != 'Llamado a presentar expresiones de interés'))
        df = df.where((df.Tipo_Contrato == 'Obra'))

        return df
    
    def cargar_df_secop(self,etl,df_secop_crudo,columnas_drop,df_dpto_reg,old_new_names_dict,cols_integer,cols_date,cols_str,cols_money):
        try:

                df_secop=etl.eliminar_columnas_innecesarias(df_secop_crudo,columnas_drop)
                df_secop = etl.rename_columns(df_secop, old_new_names_dict)
                df_secop =etl.impute_with_zero(df_secop, cols_integer)
                df_secop =etl.impute_with_date(df_secop, cols_date)
                df_secop = etl.impute_with_string(df_secop, cols_str)
                df_secop = etl.remove_commas(df_secop, cols_money)
                df_secop = etl.remove_decimal_and_convert_to_int(df_secop, cols_money)
                df_secop = etl.procesar_fechas_contrato(df_secop)

                df_secop=etl.crear_diccionarios(df_dpto_reg, df_secop)

                df_secop_filtered = etl.filtrar_datos(df_secop)

                df_secop_filtered =etl.filtrar_base(df_secop_filtered)
                # Crear DataFrame en Pandas
                df_secop_obra = df_secop_filtered.toPandas()
                # Guardar dataframe en formato csv
                df_secop_obra.to_csv(sts.datamart+'df_secop_obra.csv')
            
                return df_secop_obra

        except ValueError as e:
            print('Error setting the ticket:')
            print(f'Exception type: {type(e).__name__}')
            print(f'Error message: {str(e)}')
