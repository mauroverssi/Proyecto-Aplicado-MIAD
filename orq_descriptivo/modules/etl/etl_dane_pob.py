import findspark
findspark.init()

# Librerías básicas
import pandas as pd
import sys
import settings as sts

class PoblacionDANE:
    
    def __init__(self, df_2010_2019, df_2020_2030):
        self.df_2010_2019 = df_2010_2019
        self.df_2020_2030 = df_2020_2030
        self.poblacion_DANE = None
        self.dptos_poblacion = None
        
    def limpiar_archivo(self, df, inicio_anio, fin_anio):
        df.columns = df.iloc[10]
        df = df.drop(df[0:11].index)
        df.drop(columns=['DP'], inplace=True)
        df.rename(columns={'DPNOM':'Departamento', 'AÑO':'Anio', 'ÁREA GEOGRÁFICA':'Area'}, inplace=True)
        filas_drop = df[(df['Anio'] < inicio_anio) | (df['Anio'] > fin_anio)].index
        df.drop(filas_drop, inplace=True)
        df.dropna(inplace=True)
        return df
    
    def unir_tablas(self, df1, df2):
        df = pd.concat([df1, df2])
        df.reset_index(drop=True, inplace=True)
        df.columns.name = ''
        return df
    
    def cambiar_nombres_departamentos(self, df):
        df.loc[df['Departamento'] == 'Bogotá, D.C.', 'Departamento'] = 'Bogotá D.C.'
        df.loc[df['Departamento'] == 'Quindio', 'Departamento'] = 'Quindío'
        df.loc[df['Departamento'] == 'Archipiélago de San Andrés', 'Departamento'] = 'San Andrés'
        df = df.rename(columns={'Departamento':'ID_Departamento',
                                'Area': 'ID_Area',
                                'Anio': 'ID_Anio',
                                "Población": "Poblacion"})
        return df
    
    def obtener_nombres_departamentos(self, df):
        return df.Departamento.unique().tolist()   
    
   
    
    def poblacion(self, poblacion_dane,pobl_2010_2019,pobl_2020_2030,inicio_anio1=2010,inicio_anio2=2020,fin_anio1=2019,fin_anio2=2030):
        
        try:
            pobl_2010_2019=poblacion_dane.limpiar_archivo(pobl_2010_2019, inicio_anio1, fin_anio1)
            pobl_2020_2030=poblacion_dane.limpiar_archivo(pobl_2020_2030, inicio_anio2, fin_anio2)
            poblacion_DANE = poblacion_dane.unir_tablas(pobl_2010_2019, pobl_2020_2030)
            dptos_poblacion =poblacion_dane.obtener_nombres_departamentos(poblacion_DANE)
            poblacion_DANE =poblacion_dane.cambiar_nombres_departamentos(poblacion_DANE)
            # Guardar tabla
            return poblacion_DANE.to_csv(sts.datamart+"poblacion_DANE_Hechos.csv", index=False, sep = ';', encoding='utf-8')
        
        except ValueError as e:
            print('Error setting the ticket:')
            print(f'Exception type: {type(e).__name__}')
            print(f'Error message: {str(e)}')

    