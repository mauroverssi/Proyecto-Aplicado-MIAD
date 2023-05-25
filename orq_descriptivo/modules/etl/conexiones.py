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
        schema = sts.schema
        results_rdd = self.spark_session.sparkContext.parallelize(results)
        results_df = self.spark_session.read.json(results_rdd, schema)
        
        # Guardar el DataFrame como archivo CSV
        results_df.write.csv(sts.crudos+"SECOP.csv", header=True, mode="overwrite")

