#Básicas
import csv
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import settings as sts
import sys

#Gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.utils import deaccent
from gensim.models import LdaModel
from gensim.models import Phrases
from gensim.corpora import Dictionary
# Librerías para el cálculo de la coherencia
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaMulticore
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel
import csv
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
#Otras para análisis y procesamiento de texto
import nltk

import spacy
nlp = spacy.load("es_core_news_lg")

from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import os

# *********************************************************************************************************************

def iter_column(df, col_name,stoplist):
    """
    Esta función toma un DataFrame de pandas y el nombre de una columna y devuelve un iterador que
    produce una lista de lemas para cada línea en la columna especificada.

    Argumentos:
        df (pandas.DataFrame): El DataFrame de pandas a leer.
        col_name (str): El nombre de la columna en la que queremos iterar.

    Yields:
        Una lista de lemas para cada línea en la columna especificada.

    Ejemplo de uso:

    >>> for lemmas in iter_column(df, 'texto'):
            print(lemmas)

        ['comprar', 'manzana', 'pera', 'naranja']
        ['ir', 'cine', 'amigo']
        ['cocinar', 'comida', 'saludable', 'cena']
        ...
    """
    # Itera sobre cada línea en la columna especificada
    for line in df[col_name]:
        # Se eliminan los acentos de las palabras en la línea utilizando unidecode
        # Tokeniza la línea utilizando simple_preprocess
        tokens = simple_preprocess(line, deacc=True,min_len=3)
        # Se remueven las stopwords y las palabras que aparecen solo una vez antes de aplicar la lematización
        doc = [token for token in nlp(' '.join(tokens).lower()) if token.text not in stoplist]
        # Itera sobre cada token en el objeto Doc y devuelve su forma lematizada utilizando el atributo lemma_
        lemmas = [token.lemma_ for token in doc]
        lemmas = [deaccent(lemma) for lemma in lemmas]
        # Genera una lista de lemas para cada línea en la columna de entrada utilizando la sentencia yield
        yield lemmas


## Función para iterar una columna y devolver una lista de lemas se utiliza para generar el diccionario incluyendo bigramas 
        
def iter_column_Xgramas(df, col_name,stoplist):
    """
    Esta función toma un DataFrame de pandas y el nombre de una columna y devuelve un iterador que
    produce una lista de lemas (unigramas y bigramas) para cada línea en la columna especificada.

    Argumentos:
        df (pandas.DataFrame): El DataFrame de pandas a leer.
        col_name (str): El nombre de la columna en la que queremos iterar.

    Yields:
        Una lista de lemas para cada línea en la columna especificada.

    Ejemplo de uso:

    >>> for lemmas in iter_column(df, 'texto'):
            print(lemmas)

        ['comprar', 'manzana', 'pera', 'naranja']
        ['ir', 'cine', 'amigo']
        ['cocinar', 'comida', 'saludable', 'cena']
        ...
    """
    # Itera sobre cada línea en la columna especificada
    for line in df[col_name]:
        # Se eliminan los acentos de las palabras en la línea utilizando unidecode
        # Tokeniza la línea utilizando simple_preprocess
        tokens = simple_preprocess(line, deacc=True,min_len=3)
        # Se remueven las stopwords y las palabras que aparecen solo una vez antes de aplicar la lematización
        doc = [token for token in nlp(' '.join(tokens).lower()) if token.text not in stoplist]
        # Itera sobre cada token en el objeto Doc y devuelve su forma lematizada utilizando el atributo lemma_
        lemmas = [token.lemma_ for token in doc]
        lemmas = [deaccent(lemma) for lemma in lemmas]
        lemmas = lemmas + [token for token in bigramas[lemmas] if '_' in token] # Agregar bigramas y trigramas más frecuentes
        # Genera una lista de lemas para cada línea en la columna de entrada utilizando la sentencia yield
        yield lemmas
        

# Función para crear el corpus a partir de la muestra

class MyCorpus_sample():
    """
    Esta clase es una implementación de la interfaz de corpus de Gensim, que define cómo se accede a los documentos en un corpus de texto.
    """

    # Constructor de la clase MyCorpus
    def __init__(self, dictionary, df, column_name):
        """
        Constructor de la clase MyCorpus.

        Argumentos:
        dictionary (gensim.corpora.Dictionary): Objeto de diccionario de Gensim que se utilizará para crear bolsas de palabras.
        df (pandas.DataFrame): El DataFrame de pandas a leer.
        column_name (str): El nombre de la columna en el que queremos iterar.
        """
        self.dictionary = dictionary
        self.df = df
        self.column_name = column_name

    # Método que devuelve un generador que produce bolsas de palabras para cada línea en el dataframe
    def __iter__(self):
        """
        Método que devuelve un generador que produce bolsas de palabras para cada línea en el dataframe.

        Yields:
        Una bolsa de palabras para cada línea en el dataframe.
        """
        # Itera sobre cada línea en el dataframe utilizando el método iter_column_Xgramas
        for line in iter_column_Xgramas(self.df, self.column_name):
            # Convierte la lista de lemas en una bolsa de palabras utilizando el método doc2bow de self.dictionary
            yield self.dictionary.doc2bow(' '.join(line).split())
            
            
def iter_csv_file(filename, column_name,stoplist,bigram_transformer,trigram_transformer):
    """
    Esta función itera a través de un archivo CSV y procesa una columna específica en cada fila del archivo.
    
    :param filename: str, Nombre del archivo CSV.
    :param column_name: str, Nombre de la columna en el archivo CSV que se procesará.
    :return: str, Cadena de texto con los lemas (tokens, bigramas y trigramas) unidos, separados por un espacio.
    """
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row[column_name]
            tokens = [token for token in simple_preprocess(text, deacc=True, min_len=3) if token not in stoplist]
            lemmas = bigram_transformer[tokens]
            lemmas = trigram_transformer[lemmas]
            yield " ".join(lemmas) # unir la lista de lemas en una cadena de texto separada por un espacio
class MyCorpus():
    """
    La clase MyCorpus se utiliza para iterar a través de un archivo CSV y convertir las palabras en un documento
    en un vector de bolsa de palabras utilizando un diccionario de Gensim.
    """
    def __init__(self, dictionary, filename, column_name,stoplist,bigram_transformer,trigram_transformer):
        """
        Constructor de la clase MyCorpus.
        
        :param dictionary: gensim.corpora.Dictionary, Diccionario Gensim utilizado para convertir documentos en vectores.
        :param filename: str, Nombre del archivo CSV.
        :param column_name: str, Nombre de la columna en el archivo CSV que se procesará.
        """
        self.dictionary = dictionary
        self.filename = filename
        self.column_name = column_name
        self.stoplist=stoplist
        self.bigram_transformer=bigram_transformer
        self.trigram_transformer=trigram_transformer

    def __iter__(self):
        """
        Método iterador que se utiliza para recorrer el archivo CSV y convertir cada línea en un vector de bolsa de palabras.
        
        :yield: list of tuple, Vector de bolsa de palabras para cada documento en el archivo CSV.
        """
        for line in iter_csv_file(self.filename, self.column_name,self.stoplist,self.bigram_transformer,self.trigram_transformer):
            yield self.dictionary.doc2bow(line.split())
            
def iter_csv_file_dic(filename, column_name,stoplist,bigram_transformer,trigram_transformer):
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row[column_name]
            tokens = [token for token in simple_preprocess(text, deacc=True, min_len=3) if token not in stoplist]
            lemmas = bigram_transformer[tokens]
            lemmas = trigram_transformer[lemmas]
            yield lemmas  # devolver la lista de lemas dir
            

def find_optimal_number_of_topics_coherence(data, dictionary, preprocessed_docs, start=4, end=20, step=1, coherence_measure='c_v', coherence_topn=10, workers=None):
    """
    Encuentra el número óptimo de tópicos en un modelo LDA.

    Parámetros:
    - data: un objeto gensim corpus que contiene los documentos preprocesados.
    - dictionary: un objeto gensim Dictionary que contiene el vocabulario de todas las palabras en el corpus.
    - preprocessed_docs: una lista de listas de tokens lematizados para cada documento.
    - start: el número de tópicos mínimo a evaluar.
    - end: el número de tópicos máximo a evaluar.
    - step: el intervalo de valores entre cada número de tópicos a evaluar.
    - coherence_measure: un string con el nombre de la medida de coherencia a utilizar (por defecto 'c_v').
    - coherence_topn: un número entero que indica la cantidad de palabras más relevantes a considerar para la medida de coherencia.
    - workers: el número de núcleos a utilizar para el entrenamiento del modelo LDA (por defecto None, que utiliza todos los núcleos disponibles).

    Retorna:
    - Un gráfico de línea que muestra la medida de coherencia del modelo para cada valor de num_topics evaluado.
    - El modelo LDA óptimo.
    """
    
    coherence_scores = []
    models_list = []

    # Experimentar con diferentes valores de chunksize
    num_cores = workers if workers else os.cpu_count()
    chunk_sizes = [1000, 5000, 10000]
    chunk_size = chunk_sizes[0]
    for cs in chunk_sizes:
        if len(data) / cs > num_cores:
            chunk_size = cs
            break
    print("Chunk size: ", chunk_size)

    for num_topics in range(start, end+1, step):
        lda_model = LdaMulticore(corpus=data,
                                 id2word=dictionary,
                                 num_topics=num_topics,
                                 alpha=0.1,
                                 eta=0.5,
                                 random_state=42,
                                 workers=workers,
                                 chunksize=2000)

        coherence_model_lda = CoherenceModel(model=lda_model,
                                             texts=preprocessed_docs,
                                             dictionary=dictionary,
                                             coherence=coherence_measure,
                                             topn=coherence_topn)

        coherence_lda = coherence_model_lda.get_coherence()
        coherence_scores.append(coherence_lda)
        models_list.append(lda_model)

    # Plot coherence scores
    x = range(start, end+1, step)
    plt.plot(x, coherence_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Find the optimal number of topics
    optimal_num_topics = x[coherence_scores.index(max(coherence_scores))]

    # Print the optimal number of topics
    print("Optimal number of topics: ", optimal_num_topics)

    # Return the models list and coherence scores
    return models_list, coherence_scores, optimal_num_topics


def find_optimal_number_of_topics_perplexity(data, dictionary, preprocessed_docs, start=4, end=20, step=1, workers=None):
    """
    Encuentra el número óptimo de tópicos en un modelo LDA.

    Parámetros:
    - data: un objeto gensim corpus que contiene los documentos preprocesados.
    - dictionary: un objeto gensim Dictionary que contiene el vocabulario de todas las palabras en el corpus.
    - preprocessed_docs: una lista de listas de tokens lematizados para cada documento.
    - start: el número de tópicos mínimo a evaluar.
    - end: el número de tópicos máximo a evaluar.
    - step: el intervalo de valores entre cada número de tópicos a evaluar.
    - workers: el número de núcleos a utilizar para el entrenamiento del modelo LDA (por defecto None, que utiliza todos los núcleos disponibles).

    Retorna:
    - Un gráfico de línea que muestra la perplejidad del modelo para cada valor de num_topics evaluado.
    - El modelo LDA óptimo.
    """

    perplexity_scores = []
    models_list = []

    # Experimentar con diferentes valores de chunksize
    num_cores = workers if workers else os.cpu_count()
    chunk_sizes = [1000, 5000, 10000]
    chunk_size = chunk_sizes[0]
    for cs in chunk_sizes:
        if len(data) / cs > num_cores:
            chunk_size = cs
            break
    print("Chunk size: ", chunk_size)

    for num_topics in range(start, end+1, step):
        lda_model = LdaMulticore(corpus=data,
                                 id2word=dictionary,
                                 num_topics=num_topics,
                                 alpha=0.1,
                                 eta=0.5,
                                 random_state=42,
                                 workers=workers,
                                 chunksize=chunk_size)

        perplexity = lda_model.log_perplexity(data)
        perplexity_scores.append(perplexity)
        models_list.append(lda_model)

    # Plot perplexity scores
    x = range(start, end+1, step)
    plt.plot(x, perplexity_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity")
    plt.legend(("perplexity_values"), loc='best')
    plt.show()

    # Find the optimal number of topics
    optimal_num_topics = x[perplexity_scores.index(min(perplexity_scores))]

    # Print the optimal number of topics
    print("Optimal number of topics: ", optimal_num_topics)

    # Return the models list and perplexity scores
    return models_list, perplexity_scores, optimal_num_topics



def find_optimal_number_of_topics_coherence_mix(data, dictionary, preprocessed_docs, start=4, end=20, step=1, coherence_measure='c_v', coherence_topn=10, workers=None, alphas=[0.01, 0.1, 1], etas=[0.01, 0.1, 1]):
    """
    Encuentra el número óptimo de tópicos en un modelo LDA.

    ...
    - alphas: una lista de valores de alpha a probar.
    - etas: una lista de valores de eta a probar.

    Retorna:
    - Un diccionario de modelos y sus puntuaciones de coherencia.
    - El modelo LDA óptimo.
    """

    models_dict = {}
    optimal_model = None
    max_coherence = -1
    optimal_num_topics = -1
    optimal_alpha = -1
    optimal_eta = -1

    num_cores = workers if workers else os.cpu_count()
    chunk_sizes = [1000, 5000, 10000]
    chunk_size = chunk_sizes[0]
    for cs in chunk_sizes:
        if len(data) / cs > num_cores:
            chunk_size = cs
            break
    print("Chunk size: ", chunk_size)

    for alpha in alphas:
        for eta in etas:
            for num_topics in range(start, end+1, step):
                lda_model = LdaMulticore(corpus=data,
                                         id2word=dictionary,
                                         num_topics=num_topics,
                                         alpha=alpha,
                                         eta=eta,
                                         random_state=42,
                                         workers=workers,
                                         chunksize=2000)

                coherence_model_lda = CoherenceModel(model=lda_model,
                                                     texts=preprocessed_docs,
                                                     dictionary=dictionary,
                                                     coherence=coherence_measure,
                                                     topn=coherence_topn)

                coherence_lda = coherence_model_lda.get_coherence()

                # Store the model, parameters and coherence score in the dictionary
                models_dict[(alpha, eta, num_topics)] = {'model': lda_model, 'coherence': coherence_lda}

                if coherence_lda > max_coherence:
                    max_coherence = coherence_lda
                    optimal_model = lda_model
                    optimal_num_topics = num_topics
                    optimal_alpha = alpha
                    optimal_eta = eta

    print("Optimal number of topics: ", optimal_num_topics)
    print("Optimal alpha: ", optimal_alpha)
    print("Optimal eta: ", optimal_eta)

    # Return the models dictionary and optimal parameters
    return models_dict, optimal_model, max_coherence, optimal_num_topics, optimal_alpha, optimal_eta


def assign_most_probable_topic(lemmas,dictionary,tfidf_model,mejor_modelo):
    bow = dictionary.doc2bow(lemmas)
    tfidf_bow = tfidf_model[bow]
    topics = mejor_modelo.get_document_topics(tfidf_bow)
    most_probable_topic = max(topics, key=lambda x: x[1])[0]
    return most_probable_topic

def execute_analysis(datos):
    try:
        nltk.download('stopwords')
        lista_stopwords = stopwords.words("spanish")
        dictionary = None
        lista_stopwords2 = []
        with open(sts.planos+'stoplist.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lista_stopwords2.append(row[0])  # Asumiendo que cada línea contiene una sola stopword

        # Departamento tokenizado en minúsculas y sin acentos
        df_dpto = pd.read_csv(sts.planos+'Regiones_Departamentos.csv', sep=';')
        df_dpto = list(iter_column(df_dpto, 'Dpto_SECOP',lista_stopwords))
        Token_departamento = []
        for i in df_dpto:
            Token_departamento = Token_departamento + i
        Token_departamento = list(set(Token_departamento))

        # Generar única lista
        departamentos = list(set(Token_departamento + sts.lista_departamento))
        departamentos.sort()
        # Generar lista con municipios

        df_municipio = pd.read_csv(sts.planos+'Departamentos_y_municipios_de_Colombia.csv', sep=',')
        df_municipio = list(iter_column(df_municipio, 'MUNICIPIO',lista_stopwords))
        municipios = []
        for i in df_municipio:
            municipios = municipios + i
        municipios = list(set(municipios))
        municipios.sort()
        # Consolidar palabras en una lista y exportar a archivo csv
        stoplist = list(set(lista_stopwords + sts.stopwords_analisis + departamentos + municipios + lista_stopwords2))
        stoplist.sort()
        # Guardar Stoplist en archivo csv
        Stoplist_df = pd.DataFrame(stoplist)
        Stoplist_df.to_csv(sts.planos+'Stoplist.csv', index=False)

        stoplist = pd.read_csv(sts.planos+'Stoplist.csv')
        stoplist = list(stoplist['0'])

        #Leer el archivo CSV y obtener una lista de tokens para cada línea en la columna 'texto'
        with open(sts.datamart+'df_secop_obra.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            texts = [simple_preprocess(row['Detalle_Objeto_Contratar'], deacc=True, min_len=3) for row in reader]

        # Entrenar el modelo de bigrama
        bigram_model = Phrases(texts, min_count=5, threshold=10)
        bigram_model.save(sts.pick+'bigram_model.pkl')

        # Entrenar el modelo de trigrama
        trigram_model = Phrases(bigram_model[texts], min_count=5, threshold=10)
        trigram_model.save(sts.pick+'trigram_model.pkl')

        # Generar objetos bigram_transformer y trigram_transformer
        bigram_transformer = Phraser(bigram_model)
        trigram_transformer = Phraser(trigram_model)

        dictionary = corpora.Dictionary(iter_csv_file_dic(sts.datamart+'df_secop_obra.csv',
                                                        'Detalle_Objeto_Contratar',stoplist,bigram_transformer,trigram_transformer))

        # Depurar el diccionario

        # Retirar tokens con frecuencia = 1
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]

        # Retirar    # palabras en stoplist
        stop_ids = [dictionary.token2id[stopword]
                    for stopword in stoplist
                    if stopword in dictionary.token2id]

        dictionary.filter_tokens(once_ids + stop_ids)

        ## Crear corpus con la muestra
        corpus = MyCorpus(dictionary, sts.datamart+'df_secop_obra.csv', 'Detalle_Objeto_Contratar',stoplist,bigram_transformer,trigram_transformer)

        tfidf_model = TfidfModel(corpus)

        # Transformar el corpus en una matriz TF-IDF
        corpus_tfidf = tfidf_model[corpus]

        preprocessed_docs = list(iter_csv_file_dic(sts.datamart+'df_secop_obra.csv', 'Detalle_Objeto_Contratar',stoplist,bigram_transformer,trigram_transformer))

        alphas = [0.38, 0.40, 0.42]
        etas = [0.65, 0.70, 0.75]
        # Llama a la función
        models_dict, optimal_model, max_coherence, optimal_num_topics, optimal_alpha, optimal_eta = find_optimal_number_of_topics_coherence_mix(
            data=list(corpus_tfidf),
            dictionary=dictionary,
            preprocessed_docs=preprocessed_docs,
            start=6,
            end=12,
            step=1,
            coherence_measure='c_v',
            coherence_topn=5,
            workers=None,
            alphas=alphas,
            etas=etas)

        # Suponiendo que el mejor modelo obtenido es 'optimal_model'
        # y deseas guardarlo en un archivo llamado 'mejor_modelo.pkl'

        with open(sts.pick+'mejor_modelo.pkl', 'wb') as f:
            pickle.dump(optimal_model, f)

        ruta_archivo = sts.pick+"mejor_modelo.pkl"

        # Cargar el archivo pickle
        with open(ruta_archivo, "rb") as archivo:
            mejor_modelo = pickle.load(archivo, fix_imports=True)

            # Número de palabras más relevantes que quieres mostrar para cada tópico
        num_words = 10

        # Obtener los tópicos y sus palabras más relevantes
        topics = optimal_model.show_topics(num_topics=10, num_words=num_words, formatted=False)

        # Aplicar la función a 'preprocessed_docs'
        most_probable_topics = [assign_most_probable_topic(lemmas,dictionary,tfidf_model,mejor_modelo) for lemmas in preprocessed_docs]

        datos['ID_Tópico_LDA'] = most_probable_topics

        datos.to_csv(sts.datamart+'datos_con_topicos.csv', index=False)
        
    except ValueError as e:
        print('Error setting the ticket:')
        print(f'Exception type: {type(e).__name__}')
        print(f'Error message: {str(e)}')