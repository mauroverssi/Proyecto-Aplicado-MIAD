# Análisis de Inversión Pública y Pobreza Multidimensional - Framework Dockerizado
Bienvenidos al repositorio de nuestro proyecto, que tiene como objetivo proporcionar una herramienta que relaciona la inversión pública en infraestructura con las dimensiones del Índice de Pobreza Multidimensional (IPM) en Colombia.

Este proyecto está dockerizado para facilitar la configuración y despliegue del mismo, asegurando la replicabilidad del entorno y la portabilidad de la aplicación.

## Tabla de contenidos
1. Instalación
2. Uso
3. Contribución
4. Soporte
5. Licencia
## Instalación
Para utilizar este proyecto, necesitará tener Docker instalado en su sistema. Si aún no lo tiene, puede descargarlo aquí.

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
