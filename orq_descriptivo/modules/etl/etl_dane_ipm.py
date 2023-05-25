import findspark
findspark.init()
import settings as sts
import sys

# Librerías básicas
import pandas as pd
from pyspark.sql.functions import *

class IPM_DANE:
    
    def __init__(self):
        self.nuevos_nombres = None
        self.nombre_columna = None

    def procesar_dataframe(self,df, nuevos_nombres, nombre_columna):
        # Asignar los nuevos nombres de las columnas al df
        df.columns = nuevos_nombres
        #Reemplazar las columnas con el primer valor de la fila
        primer_valor = df.iloc[0][nombre_columna]
        df.loc[:, nombre_columna] = primer_valor
        return df
   
    def unir_dataframes(self,df, anios):
        # Verificar la existencia de las columnas de año
        columnas_anios = [str(year) for year in anios if str(year) in df.columns]

        # Lista de dataframes pivote para cada año
        df_pivots = []
        for year in columnas_anios:
            df_year = df.loc[:, ['region', 'variable', year]]
            df_pivot = df_year.pivot(index='region', columns='variable', values=year)
            df_pivot['anio'] = int(year)
            df_pivots.append(df_pivot)

        # Concatenar los dataframes pivote
        df_final = pd.concat(df_pivots, axis=0).reset_index()

        return df_final

        
    def reordenar_dataframes(self, df, year_structure):
        dfs_pivot = []
        
        for year, config in year_structure.items():
            df_year = df.iloc[:, config['columns']]
            df_year.columns = config['nombres']
            df_pivot = df_year.pivot(index='departamento', columns='variable', values='total')
            df_pivot['tipo'] = config['tipo']
            df_pivot['anio'] = year
            dfs_pivot.append(df_pivot)
        
        df_final = pd.concat(dfs_pivot, axis=0).reset_index()
        return df_final

    def procesar_departamentos(self, ipm):
        dfs = []
        
        for departamento, skiprows in sts.departamentos:
            df = pd.read_excel(ruta_archivo=sts.ruta_archivo_21, sheet_name=sts.hojas[1], skiprows=range(skiprows), nrows=15)
            df = ipm.procesar_dataframe(df, sts.nuevos_nombres, sts.nombre_columna)
            df = ipm.reordenar_dataframes(df, sts.year_structure)
            dfs.append(df)
        
        return pd.concat(dfs)
    
    def transformar_tabla(self,variables_departamento):
        variables_departamento = variables_departamento.melt(id_vars=["departamento", "anio", "tipo"],
                                                            var_name="ID_Variable_IPM",
                                                            value_name="IPM")

        variables_departamento = variables_departamento.rename(columns={"departamento": "ID_Ubicacion",
                                                                        "anio": "ID_Anio",
                                                                        "tipo": "ID_Area"})

        Area_dicc = {'cabeceras': 'Cabecera',
                    'centros_poblados_rural_disperso': 'Centros Poblados y Rural Disperso',
                    'total': 'Total'}

        variables_departamento['ID_Area'] = variables_departamento['ID_Area'].map(Area_dicc)

        variables_departamento.IPM.fillna(0, inplace=True)

        variables_departamento['ID_Tipo_ubicacion'] = 'Departamento'

        variables_departamento = variables_departamento[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Area', 'ID_Variable_IPM', 'IPM']]

        return variables_departamento

    def transformar_ipm_departamentos(self,ipm_departamentos, columnas, tipos, años):
        dfs = []
        
        for i in range(len(columnas)):
            df = ipm_departamentos.iloc[:, columnas[i]]
            df.columns = ['departamento', 'total', 'cabeceras', 'centros_poblados_rural_disperso']
            df = df.melt(id_vars=['departamento'], value_vars=['total', 'cabeceras', 'centros_poblados_rural_disperso'], var_name='tipo', value_name='ipm')
            df['tipo'] = df['tipo'].map(tipos[i])
            df = df.loc[:, ['departamento', 'ipm', 'tipo']]
            df['año'] = años[i]
            dfs.append(df)
        
        ipm_departamentos_final = pd.concat(dfs, axis=0).reset_index(drop=True)
        
        return ipm_departamentos_final
    
    def modificar_ipm_departamentos(self,ipm_departamentos_final):
        # Cambiar nombre de columnas
        ipm_departamentos_final = ipm_departamentos_final.rename(columns={"departamento": "ID_Ubicacion",
                                                                        'ipm': 'IPM',
                                                                        'tipo': 'ID_Area',
                                                                        "año": "ID_Anio"})

        # Crear columna con tipo ubicación
        ipm_departamentos_final['ID_Tipo_ubicacion'] = 'Departamento'

        # Ordenar columnas
        ipm_departamentos_final = ipm_departamentos_final[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Area', 'IPM']]

        # Estandarizar términos de la columna 'Area_IPM' mediante map de diccionario
        Area_dicc = {'cabeceras': 'Cabecera',
                    'centros_poblados_rural_disperso': 'Centros Poblados y Rural Disperso',
                    'total': 'Total'}

        ipm_departamentos_final['ID_Area'] = ipm_departamentos_final['ID_Area'].map(Area_dicc)

        # Imputar valores IPM Nan con cero
        ipm_departamentos_final.IPM.fillna(0, inplace=True)
        
        return ipm_departamentos_final
    
    def transformar_ipm_regiones(self,ipm_regiones, columnas, tipos, años):
        dfs = []
        
        for i in range(len(columnas)):
            df = ipm_regiones.iloc[:, columnas[i]]
            df.columns = ['region', 'total', 'cabeceras', 'centros_poblados_rural_disperso']
            df = df.melt(id_vars=['region'], value_vars=['total', 'cabeceras', 'centros_poblados_rural_disperso'], var_name='tipo', value_name='ipm')
            df['tipo'] = df['tipo'].map(tipos[i])
            df = df.loc[:, ['region', 'ipm', 'tipo']]
            df['año'] = años[i]
            dfs.append(df)
        
        ipm_regiones_final = pd.concat(dfs, axis=0).reset_index(drop=True)
        
        return ipm_regiones_final
    
    def modificar_ipm_regiones(self,ipm_regiones_final,Area_dicc,Region_dict):
        ipm_regiones_final = ipm_regiones_final.rename(columns={"region": "ID_Ubicacion",
                                                                'ipm': 'IPM',
                                                                'tipo': 'ID_Area',
                                                                "año": "ID_Anio"})
        
        ipm_regiones_final['ID_Area'] = ipm_regiones_final['ID_Area'].map(Area_dicc)
        
        ipm_regiones_final['IPM'].fillna(0, inplace=True)
        
        ipm_regiones_final['ID_Ubicacion'] = ipm_regiones_final['ID_Ubicacion'].map(Region_dict)
        
        ipm_regiones_final['ID_Tipo_ubicacion'] = 'Región'
        
        ipm_regiones_final = ipm_regiones_final[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Area', 'IPM']]
        
        return ipm_regiones_final
    
    def procesar_variables_region(self,ipm,region_dict,ruta_archivo, nombre_hoja, skiprows_dict, nrows_dict,anios):
        variables_region = pd.DataFrame()
        
        # Leer y procesar los datos para cada región
        for region, skiprows, nrows in zip(skiprows_dict.keys(), skiprows_dict.values(), nrows_dict.values()):
            df = pd.read_excel(ruta_archivo, sheet_name=nombre_hoja, skiprows=skiprows, nrows=nrows)
            df = ipm.procesar_dataframe(df, ['region', 'variable', '2018', '2019', '2020', '2021'], 'region')
            df = ipm.unir_dataframes(df,anios)
            variables_region = pd.concat([variables_region, df], axis=0)
        
        # Cambiar estructura de la tabla
        variables_region = variables_region.melt(id_vars=["region", "anio"], var_name="ID_Variable_IPM", value_name="IPM")
        
        # Cambiar nombre de las columnas
        variables_region = variables_region.rename(columns={"region": "ID_Ubicacion", "anio": "ID_Anio"})
        
        variables_region['ID_Ubicacion'] = variables_region['ID_Ubicacion'].map(region_dict)
        
        variables_region['ID_Tipo_ubicacion'] = 'Región'
        variables_region['ID_Area'] = 'Total'
        
        variables_region = variables_region[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Area', 'ID_Variable_IPM', 'IPM']]
        
        return variables_region
    
    def procesar_ipm_nacional(self, df,Area_dicc):
        # Cambiar nombre de columnas
        df.rename(columns={'Unnamed: 0': 'ID_Area', '2020**': 2020}, inplace=True)



        df['ID_Area'] = df['ID_Area'].map(Area_dicc)

        # Crear columnas con ubicación y tipo ubicación
        df['ID_Tipo_ubicacion'] = 'Nacional'
        df['ID_Ubicacion'] = 'Nacional'

        # Eliminar columna año 2017 dado que no tiene toda la información
        df.drop([2017], axis=1, inplace=True)

        # Cambiar estructura de la tabla
        df = df.melt(id_vars=['ID_Tipo_ubicacion', 'ID_Ubicacion', 'ID_Area'],
                    var_name="ID_Anio",
                    value_name="IPM")

        # Organizar columnas
        df = df[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Area', 'IPM']]

        return df

    def procesar_variables_nacionales(self, ruta_archivo, nombre_hoja, skiprows_dict):
        # Leer y procesar los datos para cada región
        variables_nacionales = []
        for area, skiprows in skiprows_dict.items():
            df = pd.read_excel(ruta_archivo, sheet_name=nombre_hoja, skiprows=skiprows, nrows=15)
            df['Unnamed: 0'] = area
            variables_nacionales.append(df)

        # Unir la información
        variables_nacionales = pd.concat(variables_nacionales, axis=0)

        # Crear columnas con ubicación y tipo ubicación
        variables_nacionales['ID_Tipo_ubicacion'] = 'Nacional'
        variables_nacionales['ID_Ubicacion'] = 'Nacional'

        # Cambiar nombre de columnas
        variables_nacionales.rename(columns={'Unnamed: 0': 'ID_Area', 'Unnamed: 1': 'ID_Variable_IPM', '2020**': 2020}, inplace=True)

        # Eliminar columna año 2017 dado que no tiene toda la información
        variables_nacionales.drop([2017], axis=1, inplace=True)

        # Cambiar estructura de la tabla
        variables_nacionales = variables_nacionales.melt(id_vars=['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Area', 'ID_Variable_IPM'],
                                                        var_name="ID_Anio",
                                                        value_name="IPM")

        variables_nacionales = variables_nacionales[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Area', 'ID_Variable_IPM', 'IPM']]

        return variables_nacionales
    
    def procesar_contribucion_nacional(SELF, ruta_archivo, nombre_hoja, skiprows_dict, anios,Area_dicc):
        # Leer y procesar los datos para cada año
        contribuciones_nacionales = []
        for anio, skiprows in zip(anios, skiprows_dict.values()):
            df = pd.read_excel(ruta_archivo, sheet_name=nombre_hoja, skiprows=skiprows, nrows=5)
            df['ID_Anio'] = anio
            df.rename(columns={'Unnamed: 0': 'Dimensión'}, inplace=True)
            contribuciones_nacionales.append(df)

        # Unir las tablas
        contribucion_nacional = pd.concat(contribuciones_nacionales, axis=0, ignore_index=True)

        # Crear columnas de Ubicación
        contribucion_nacional['ID_Tipo_ubicacion'] = 'Nacional'
        contribucion_nacional['ID_Ubicacion'] = 'Nacional'

        # Eliminar columnas sin datos
        contribucion_nacional.drop(['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1, inplace=True)

        # Cambiar estructura de la tabla
        contribucion_nacional = contribucion_nacional.melt(id_vars=['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'Dimensión'],
                                                        var_name="ID_Area",
                                                        value_name="Contribucion_IPM")


        contribucion_nacional['ID_Area'] = contribucion_nacional.ID_Area.map(Area_dicc)

        # Ajustar nombre de columnas
        contribucion_nacional.rename(columns={'Dimensión':'ID_Dimension_IPM'}, inplace=True)

        # Eliminar datos de Cabecera, Centros poblados y Rural disperso
        contribucion_nacional = contribucion_nacional[contribucion_nacional['ID_Area'] == 'Total']

        return contribucion_nacional
    
    def procesar_contribucion_regional(self, ruta_archivo, nombre_hoja, skiprows_dict, anios):
        # Leer y procesar los datos para cada año
        contribuciones_regionales = []
        for anio, skiprows in zip(anios, skiprows_dict.values()):
            df = pd.read_excel(ruta_archivo, sheet_name=nombre_hoja, skiprows=skiprows, nrows=5)
            df['ID_Anio'] = anio
            contribuciones_regionales.append(df)

        # Unir las tablas
        contribucion_regional = pd.concat(contribuciones_regionales, axis=0, ignore_index=True)

        # Cambiar nombre de columnas
        contribucion_regional.rename(columns={'Unnamed: 0': 'ID_Dimension_IPM'}, inplace=True)

        # Agregar columna con información de Ubicación
        contribucion_regional['ID_Tipo_ubicacion'] = 'Región'

        # Cambiar estructura de la tabla
        contribucion_regional = contribucion_regional.melt(id_vars=['ID_Tipo_ubicacion', 'ID_Anio', 'ID_Dimension_IPM'],
                                                        var_name="ID_Ubicacion",
                                                        value_name="Contribucion_IPM")

        # Agregar información de Área_IPM
        contribucion_regional['ID_Area'] = 'Total'

        # Cambiar nombre de columnas
        contribucion_regional.rename(columns={'Dimensión':'Dimension_IPM'}, inplace=True)

        # Organizar columnas
        contribucion_regional = contribucion_regional[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Dimension_IPM',
                                                    'ID_Area', 'Contribucion_IPM']]

        return contribucion_regional
    
    
    def read_excel_with_location(self,ruta_archivo, nombre_hoja, skiprows, nrows, ubicacion):
        df = pd.read_excel(ruta_archivo, sheet_name=nombre_hoja, skiprows=skiprows, nrows=nrows)
        df['ID_Ubicacion'] = ubicacion
        return df
    
    def procesar_datos_ipm(self,ipm,ruta_archivo_21, ruta_archivo2, 
                           dir_fuente_crudo,ipm_departamentos,columnas, tipos, anios,
                           ipm_regiones, Area_dicc, Region_dict,
                           region_dict, nombre_hoja2, skiprows_dict, nrows_dict,nombre_hoja3,
                           Area_dicc_zona, nombre_hoja4, skiprows_dict1, nombre_hoja5, 
                           skiprows_dict2, anios2, Area_dicc1, skiprows_dict3,departamentos_tupla1):
        try:
            # Transformar y modificar IPM departamentos
            ipm_departamentos_final = ipm.transformar_ipm_departamentos(ipm_departamentos, columnas, tipos, anios)
            ipm_departamentos_final = ipm.modificar_ipm_departamentos(ipm_departamentos_final)

            # Transformar y modificar IPM regiones
            ipm_regiones_final = ipm.transformar_ipm_regiones(ipm_regiones, columnas, tipos, anios)
            ipm_regiones_final = ipm.modificar_ipm_regiones(ipm_regiones_final, Area_dicc, Region_dict)

            # Procesar variables departamentales
            variables_departamento = ipm.procesar_departamentos(ipm)
            variables_departamento = ipm.transformar_tabla(variables_departamento)

            # Procesar variables regionales
            variables_region = ipm.procesar_variables_region(ipm, region_dict, ruta_archivo_21, nombre_hoja2, skiprows_dict, nrows_dict, anios)

            # Procesar IPM nacional
            ipm_nacional = pd.read_excel(ruta_archivo2, sheet_name=nombre_hoja3, skiprows=range(14), nrows=3)
            ipm_nacional = ipm.procesar_ipm_nacional(ipm_nacional, Area_dicc_zona)

            # Procesar variables nacionales
            variables_nacional = ipm.procesar_variables_nacionales(ruta_archivo2, nombre_hoja4, skiprows_dict1)

            # Procesar contribución nacional
            contribucion_nacional = ipm.procesar_contribucion_nacional(ruta_archivo2, nombre_hoja5, skiprows_dict2, anios2, Area_dicc1)

            # Procesar contribución regional
            contribucion_regional = ipm.procesar_contribucion_regional(ruta_archivo2, nombre_hoja5, skiprows_dict3, anios2)

            # Procesar contribución departamento
            contribucion_departamento = pd.DataFrame()
            for ubicacion, skiprows in departamentos_tupla1:
                df = ipm.read_excel_with_location(ruta_archivo_21, nombre_hoja5, range(skiprows), 5, ubicacion)
                contribucion_departamento = pd.concat([contribucion_departamento, df], ignore_index=True)
            contribucion_departamento.rename(columns={'Dimensión': 'ID_Dimension_IPM'}, inplace=True)
            contribucion_departamento['ID_Tipo_ubicacion'] = 'Departamento'
            contribucion_departamento['ID_Area'] = 'Total'
            contribucion_departamento = contribucion_departamento.melt(id_vars=['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Dimension_IPM', 'ID_Area'],
                                                                    var_name="ID_Anio",
                                                                    value_name="Contribucion_IPM")
            contribucion_departamento = contribucion_departamento[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Dimension_IPM', 'ID_Area', 'Contribucion_IPM']]



        # Crear tabla con IPM a nivel Nacional, Regional y Departamental
            IPM = pd.concat([ipm_departamentos_final, ipm_regiones_final, ipm_nacional], axis=0, ignore_index=True)

            # Guardar dataframe en formato csv
            IPM.to_csv(sts.datamart+'IPM_Hechos.csv')

            # Crear tabla con contribuciones IPM a nivel Nacional, Regional y Departamental
            contribucion_IPM = pd.concat([contribucion_nacional, contribucion_regional, contribucion_departamento], axis=0, ignore_index=True)
            contribucion_IPM.drop(columns=['ID_Area'], inplace=True)

            # Guardar dataframe en formato csv
            contribucion_IPM.to_csv(sts.datamart+'Contribucion_IPM_Hechos.csv')

            # Se guardan todas las mediciones de IPM (totales y por variable) en una misma tabla
            IPM_aux_variable = IPM.copy()
            IPM_aux_variable['ID_Variable_IPM'] = 'Total'
            IPM_aux_variable = IPM_aux_variable[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Area', 'ID_Variable_IPM', 'IPM']]

            IPM_aux_contribucion = IPM.copy()
            IPM_aux_contribucion['ID_Dimension_IPM'] = 'Total'
            IPM_aux_contribucion = IPM_aux_contribucion[['ID_Ubicacion', 'ID_Tipo_ubicacion', 'ID_Anio', 'ID_Dimension_IPM', 'ID_Area', 'IPM']]
            drop_index = IPM_aux_contribucion[IPM_aux_contribucion['ID_Area'] != 'Total'].index
            IPM_aux_contribucion.drop(index=drop_index, inplace=True)
            IPM_aux_contribucion['Contribucion_IPM'] = 100

            # Crear tabla con variables IPM a nivel Nacional, Regional y Departamental
            Variables_IPM = pd.concat([IPM_aux_variable, variables_departamento, variables_region, variables_nacional],
                                    axis=0, ignore_index=True)

            # Guardar dataframe en formato csv
            Variables_IPM.to_csv(sts.datamart+'Variables_IPM_Hechos.csv')

            # Nombre del archivo Excel
            archivo = 'Regiones_Departamentos.xls'

            # Leer archivo
            departamentos_dim = pd.read_excel(dir_fuente_crudo + archivo)

            # Eliminar columna de referencia para manejo datos SECOP
            departamentos_dim.drop('Dpto_SECOP', axis=1, inplace=True)

            # Ajustar nombre de columnas
            departamentos_dim.rename(columns={'Departamento':'ID_Departamento', 'Region':'ID_Region'}, inplace=True)

            # Guardar dataframe en formato csv
            departamentos_dim.to_csv(sts.datamart+'Departamentos_Dim.csv')

            # Cargar regiones en lista y ordenar
            lista_regiones_dim = list(departamentos_dim.ID_Region.unique())
            lista_regiones_dim.sort()

            # Crear dataframe con regiones
            regiones_dim = pd.DataFrame(data={'ID_Region': lista_regiones_dim})

            # Asignar país
            regiones_dim['ID_Pais'] = 'Colombia'

            # Guardar dataframe en formato csv
            regiones_dim.to_csv(sts.datamart+'Regiones_Dim.csv')

            # Crear dataframe con regiones
            pais_dim = pd.DataFrame(data={'ID_Pais': ['Colombia', 'Dummy']})
            
            
                # Eliminar la última fila de pais_dim
            pais_dim.drop(pais_dim.index[-1], inplace=True)

            # Guardar dataframe en formato csv
            pais_dim.to_csv(sts.datamart+'Pais_Dim.csv')

            # Crear listas con ubicaciones y tipo de ubicación
            lista_departamentos = departamentos_dim.ID_Departamento.unique().tolist()
            lista_regiones = regiones_dim.ID_Region.unique().tolist()
            lista_pais = pais_dim.ID_Pais.unique().tolist()
            ID_Ubicacion = lista_departamentos + lista_regiones + lista_pais
            ID_Tipo_Ubicacion = ['Departamento' for _ in lista_departamentos] + ['Región' for _ in lista_regiones] + ['Nacional' for _ in lista_pais]

            # Crear DataFrame de Ubicación
            Ubicacion = pd.DataFrame(data={'ID_Ubicacion': ID_Ubicacion, 'ID_Tipo_Ubicacion': ID_Tipo_Ubicacion})

            # Guardar dataframe en formato csv
            Ubicacion.to_csv(sts.datamart+'Ubicacion_Dim.csv')

            # Crear lista con tipos de ubicación
            lista_tipo_ubicacion = Ubicacion.ID_Tipo_Ubicacion.unique().tolist()

            # Crear DataFrame de Tipo_Ubicacion
            Tipo_Ubicacion = pd.DataFrame(data={'ID_Tipo_Ubicacion': lista_tipo_ubicacion})

            # Guardar dataframe en formato csv
            Tipo_Ubicacion.to_csv(sts.datamart+'Tipo_ubicacion_Dim.csv')

            # Crear DataFrame de Área
            Area = pd.DataFrame(data={'ID_Area': ['Total', 'Cabecera', 'Centros Poblados y Rural Disperso'],
                                    'Descripcion': ['Toda el área', 'Ciudades principales', 'Pueblos y veredas']})

            # Guardar dataframe en formato csv
            Area.to_csv(sts.datamart+'Area_Dim.csv')

            # Crear lista de dimensiones IPM
            lista_dims_IPM = contribucion_IPM.ID_Dimension_IPM.unique().tolist()

            # Crear DataFrame de Dimensiones_IPM
            IPM_dim = pd.DataFrame(data={'ID_Dimension_IPM': lista_dims_IPM,
                                        'Descripcion': ['Condiciones educación',
                                                        'Condiciones de niñez y juventud',
                                                        'Acceso a trabajo',
                                                        'Acceso a salud',
                                                        'Condiciones de vivienda']})

            # Guardar dataframe en formato csv
            IPM_dim.to_csv(sts.datamart+'Dimensiones_IPM_Dim.csv')
        
        except ValueError as e:
            print('Error setting the ticket:')
            print(f'Exception type: {type(e).__name__}')
            print(f'Error message: {str(e)}')