import findspark
import os
import pandas as pd
import numpy as np
from sodapy import Socrata
import pyspark
import socket
import multiprocessing
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
findspark.init()


class Conexion:

    def __init__(self, app_name="my_app", driver_memory="24g", executor_memory="24g", partitions="7", api_token=None,spark_port=7077):
        self.app_name = app_name
        self.driver_memory = driver_memory
        self.executor_memory = executor_memory
        self.partitions = partitions
        self.api_token = api_token
        self.spark_port = spark_port
        self.spark_session = None

            
    def init_spark_session(self):
        # Obtener la dirección IP local
        ip_address = socket.gethostbyname(socket.gethostname())
        num_cores = multiprocessing.cpu_count()
        self.spark_session = SparkSession.builder.appName(self.app_name) \
            .config("spark.master", f"local[{num_cores}]") \
            .config("spark.driver.memory", self.driver_memory) \
            .config("spark.executor.memory", self.executor_memory) \
            .config("spark.sql.execution.arrow.enabled", "true") \
            .config("spark.sql.shuffle.partitions", self.partitions) \
            .config("spark.driver.host", ip_address) \
            .config("spark.driver.bindAddress", ip_address) \
            .config("spark.ui.reverseProxyUrl", f"http://localhost:{self.spark_port}") \
            .getOrCreate()
    
    def connect_to_secop_api(self, dataset_id="f789-7hwg", limit=1000000):
        client = Socrata("www.datos.gov.co", self.api_token)
        results = client.get_all(dataset_id, limit=limit)
        
        # Crear un DataFrame de Spark a partir del RDD
        schema = StructType([
            StructField("Anno Cargue SECOP", StringType(), True),
            StructField("Anno Firma del Contrato", StringType(), True),
            StructField("Nivel Entidad", StringType(), True),
            StructField("Orden Entidad", StringType(), True),
            StructField("Tipo de Proceso", StringType(), True),
            StructField("Estado del Proceso", StringType(), True),
            StructField("Objeto a Contratar", StringType(), True),
            StructField("Detalle del Objeto a Contratar", StringType(), True),
            StructField("Tipo de Contrato", StringType(), True),
            StructField("Cuantia Proceso", StringType(), True),
            StructField("Nombre Grupo", StringType(), True),
            StructField("Nombre Familia", StringType(), True),
            StructField("Nombre Clase", StringType(), True),
            StructField("Fecha Ini Ejec Contrato", StringType(), True),
            StructField("Plazo de Ejec del Contrato", StringType(), True),
            StructField("Rango de Ejec del Contrato", StringType(), True),
            StructField("Tiempo Adiciones en Dias", StringType(), True),
            StructField("Tiempo Adiciones en Meses", StringType(), True),
            StructField("Fecha Fin Ejec Contrato", StringType(), True),
            StructField("Cuantia Contrato", StringType(), True),
            StructField("Valor Contrato con Adiciones", StringType(), True),
            StructField("Objeto del Contrato a la Firma", StringType(), True),
            StructField("Origen de los Recursos", StringType(), True),
            StructField("Departamento Entidad", StringType(), True),
        ])
        
        results_rdd = self.spark_session.sparkContext.parallelize(results)
        results_df = self.spark_session.read.json(results_rdd, schema)
        
        # Guardar el DataFrame como archivo CSV
        results_df.write.csv("tablas/SECOP.csv", header=True, mode="overwrite")

