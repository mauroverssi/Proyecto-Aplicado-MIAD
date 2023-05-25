# Análisis de Inversión Pública y Pobreza Multidimensional - Framework Dockerizado
Bienvenidos al repositorio de nuestro proyecto, que tiene como objetivo proporcionar una herramienta que relaciona la inversión pública en infraestructura con las dimensiones del Índice de Pobreza Multidimensional (IPM) en Colombia.

Este proyecto está dockerizado para facilitar la configuración y despliegue del mismo, asegurando la replicabilidad del entorno y la portabilidad de la aplicación.

## Tabla de contenidos
1. Marco Conceptual
2. Acerca la herramienta
3. Acerca de la nube
4. Soporte
5. Licencia
## Instalación
Para utilizar este proyecto, necesitará tener Docker instalado en su sistema. Si aún no lo tiene, puede descargarlo aquí.

# Framework-Proyecto
#Una vista descriptiva a la Inversión en Infraestructura y a la Pobreza Multidimensional en Colombia 

## Marco Conceptual
#### Punto de partida:  
La pobreza en Colombia es una realidad que, aparte del aspecto monetario, se percibe en las dimensiones de salud, educación, niñez y juventud, vivienda y trabajo.​

#### Necesidad de gestión: 
Para entender mejor esta situación, los gobiernos departamentales y nacionales necesitan una forma de analizar cómo se distribuyen las inversiones públicas en relación con las dimensiones de la pobreza.​

#### Herramienta inexistente: 
Sin embargo, en la actualidad no hay una herramienta que permita visualizar de forma simultanea la información histórica de contratación pública, en particular la inversión en infraestructura, con la evolución de los indicadores de pobreza del país.​

#### Impacto de la brecha: 
Esta ausencia limita la capacidad de los funcionarios públicos para analizar y comprender las inversiones en infraestructura y la pobreza multidimensional.

#### Necesitamos una solución 
que asocie de manera descriptiva la inversión en infraestructura pública con las dimensiones de la pobreza, permitiendo un análisis y entendimiento más efectivo de esta información.​

#### ¿Quién puede requerir esta solución? 
Funcionarios públicos que participen en la planeación del desarrollo nacional o departamental.

# Acerca la herramienta
Para dar solución al problema planteado, desarrollamos una herramienta descriptiva que aplica técnicas de procesamiento de lenguaje natural al texto del objeto de los contratos de obra de SECOP I para identificar inversiones asociadas a las dimensiones de pobreza​. 

Adicionalmente, enriquecemos la información agregando los registros históricos del Índice de Pobreza Multidimensional, que en adelante llamaremos IPM, y datos de población.  

## Diagrama descriptivo
Desarrollamos una herramienta descriptiva que aplica técnicas de procesamiento de lenguaje natural al objeto contractual de los contratos de obra de SECOP I para identificar inversiones asociadas a algunas de las dimensiones de la Pobreza Multidimensional


   ```mermaid
graph LR
A[Recopilación de datos: SECOP I, DANE]
A --> B[Extracción de los datos: Spark, Python]
B --> C[Transformación de los datos: Selección de columnas relevantes, ajustes de formato y organización de la información]
C --> D[Análisis de los datos: Identificación de contratos asociados a las dimensiones de pobreza usando Procesamiento de Lenguaje Natural (PLN)]
D --> E[Aplicación de los datos: Tablero de control Power BI]
   ```


## Bases de datos:
En la primera version de la herramienta no se construyeron bases de datos debido al alcance inicial y al tiempo que se tenia para el primer entregable. Sin embargo, se estructuran dos carpetas dentro del Framework, 

1. crudos, en esta tabla deben de ir todas las tablas de entrada. En el link a continuacion puede descargar todos los insumos. 
[crudos](https://drive.google.com/drive/folders/1kjr7_0ZoEPWykEFWc-YmhdrP7PBIgjWi).
2. datamart, en esta van los resultados de los modelos de datos y del modelo LDA

## Estructura del Orquestador 
A continuacion se listan los archivos que se usan en el orquestador y como se usa cada uno:

Conexiones.py
Clase con dos atributos:
* Se configura la sesión de pyspark, de manera eficiente y parametrizada.
* Se configura la conexión a la API de SECOP I por medio de Socrata, con el fin de descargar el segmento de interes de los contratos.

etl_dane_imp.py
Clase IPM_DANE
Con atributos para:
* Recibir información procesar y obtener la información geográfica, el IPM por departamentos, por regiones, a nivel nacional, la contribución de las dimensiones del IPM a nivel nacional y regional
* Se envían los resultados al DataMart para ser ingestados al tablero

etl_dane_pob.py
Clase PoblacionesDANE
* Se obtiene la información poblacional de acuerdo al senso publicado en el DANE y se llevan los resultados al DataMart

etl_secop.py
Clase SECOP_ETL
Con atributos para:
* Función donde se recibe la información del SECOP I y se procesa con el fin de tener limpia y dispuesta la información y poderla ingestar en el modelo LDA para encontrar tópicos.
* Se enriquece la base cruzando con la información del DANE
* Se dispone la información final en una tabla en el DataMart

lda.py
En este archivo se disponen funciones y clases para identificar tópicos en el detalle del objeto del contrato con el fin de identificar tópicos que se puedan asociar con las dimensiones del IPM de acuerdo a la definición de cada una de estas.
Esta trae las siguientes funciones independientes, y cases con sus respectivos atributos:
* def iter_column, función que produce una lista de lemas
* def iter_column_Xgramas, función devuelve liste de unigramas y trigramas
* class MyCorpus_sample(), constructor del corpus
* def iter_csv_file, funcion que se usa para iterar en el detalle del objeto del contrato y devuelve una cadena de texto
* def find_optimal_number_of_topics_coherence, funcion que encuentra el numero optimo de topicos en un modelo LDA de acuerdo a la coherencia
* def find_optimal_number_of_topics_perplexity, funcion que encuentra el numero optimo de topicos en un modelo LDA de acuerdo a la perplejidad
* def assign_most_probable_topic, funcion que asigna un tema probable a una lista de lemas
* def execute_analysis, función que integra las clases y funciones anteriores con el fin de obtener el resultado final de los tópicos asociados a el IPM

settings.py
Este archivo es donde se parametrizan todos los datos estáticos del proyecto, con el fin de hacer mas eficiente el código 

__init__.py
En este archivo se crea la instancia de conexión a Spark, se conecta a las fuentes del DataMart, se instancian las ETLs el modelo y se ejecutan en cascada con el fin de tener los insumos para el front

## Front
La salida de estos modelos se llevan al DataMart el cual ingesta el Front del proyecto,[Iluminando Patrones: Una Vista Descriptiva a la Inversión en Infraestructura y la Pobreza Multidimensional en Colombia](https://app.powerbi.com/reportEmbed?reportId=b7ccd959-e9fb-4983-96c2-25affa1162c4&autoAuth=true&ctid=fabd047c-ff48-492a-8bbb-8f98b9fb9cca).

## Acerca de la nube
Este proyecto se construyo sobre Django con el fin de que fuera escalable, con la finalidad de poderlo desplegar en cualquier infraestructura nube como AWS, Azure o GCP por mencionar algunas. Para esto se deja el archivo requirements.txt y Dockerfile. A continuacion se describen los pasos para Dockerizar el framework y desplegarlo en la nube:

Para dockerizar una solución y desplegarla en la nube, sigue estos pasos generales:

1. **Preparación del entorno**: Asegúrate de tener Docker instalado en tu máquina local y una cuenta en la plataforma de nube en la que deseas desplegar la solución (por ejemplo, Amazon Web Services, Google Cloud Platform, Microsoft Azure, etc.).

2. **Dockerizar la aplicación**: Crea un archivo llamado `Dockerfile` en la raíz de tu proyecto. Este archivo contiene las instrucciones para construir la imagen de Docker de tu aplicación. Especifica el sistema operativo base, las dependencias necesarias y los comandos para configurar la aplicación dentro del contenedor.

3. **Construir la imagen de Docker**: Abre una terminal en la carpeta donde se encuentra el archivo `Dockerfile` y ejecuta el siguiente comando para construir la imagen de Docker:

   ```
   docker build -t nombre_imagen .
   ```

   Reemplaza `nombre_imagen` con el nombre que deseas darle a tu imagen de Docker.

4. **Probar la imagen de Docker**: Ejecuta un contenedor basado en la imagen recién creada para asegurarte de que todo funcione correctamente. Puedes hacerlo con el siguiente comando:

   ```
   docker run --rm -p puerto_local:puerto_contenedor nombre_imagen
   ```

   Reemplaza `puerto_local` con el puerto de tu máquina local en el que deseas acceder a la aplicación y `puerto_contenedor` con el puerto en el que la aplicación expone su servicio dentro del contenedor.

5. **Subir la imagen de Docker a un registro**: Para desplegar la imagen en la nube, debes subirla a un registro de contenedores (como Docker Hub, Amazon ECR, Google Container Registry, etc.). Esto te permitirá acceder a la imagen desde la nube. Sigue la documentación del registro de contenedores de tu elección para obtener instrucciones sobre cómo subir la imagen.

6. **Configurar el entorno en la nube**: En la plataforma de nube, crea una instancia de máquina virtual o un servicio de contenedores donde desees desplegar tu solución. Asegúrate de tener configurado el entorno con las dependencias y configuraciones necesarias.

7. **Desplegar la imagen en la nube**: En la plataforma de nube, utiliza las herramientas proporcionadas para desplegar la imagen de Docker desde el registro de contenedores. Esto puede implicar crear un clúster, definir un servicio, configurar reglas de enrutamiento, etc. Sigue la documentación de la plataforma de nube específica para obtener instrucciones detalladas sobre cómo desplegar contenedores.

8. **Acceder a la solución desplegada**: Una vez que la solución se haya desplegado correctamente, podrás acceder a ella a través de la dirección o URL proporcionada por la plataforma de nube. Verifica que todo funcione según lo esperado.

Recuerda que los pasos pueden variar dependiendo de la plataforma de nube que elijas y las especificidades de tu solución. Consulta la documentación correspondiente para obtener instrucciones más detalladas sobre cómo dockerizar y desplegar en la nube en tu entorno específico.

Una vez instalado Docker, clone este repositorio a su máquina local utilizando git clone. Luego, navegue hasta el directorio del proyecto en su terminal y ejecute el siguiente comando para construir la imagen de Docker:

Copy code
docker build -t nombre_imagen .
Por último, ejecute la imagen con:

arduino
Copy code
docker run -p 8888:8888 nombre_imagen
## Uso
Una vez que la imagen de Docker esté ejecutándose, puede acceder al Tablero de Control a través de su navegador web. Siga las instrucciones que aparecen en la consola para abrir la aplicación en su navegador.

Dentro de la aplicación, encontrará diversas visualizaciones de datos y filtros interactivos que le permitirán explorar la relación entre la inversión en infraestructura y las dimensiones del IPM en Colombia.

## Contribución
Nos encantaría recibir contribuciones a este proyecto. Por favor, lea las pautas de contribución en CONTRIBUTING.md antes de enviar cualquier pull request.

## Soporte
Si tiene alguna pregunta o problema con la instalación o uso de este proyecto, por favor, abra un issue en este repositorio y haremos todo lo posible por ayudarle.

## Licencia
Este proyecto está licenciado bajo los términos de la Licencia MIT. Por favor, consulte el archivo LICENSE.md para más detalles.
