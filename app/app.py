from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from pdf2image import convert_from_path
import cv2 as cv
import pytesseract
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
import spacy
from pyrae import dle
from gensim import corpora, models
from keras.utils import pad_sequences
import tensorflow as tf
import pandas as pd

UPLOAD_FOLDER = os.path.abspath("./documentos/")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

etiquetas = ['Investigación Científica Universitaria', 'Elecciones Universitarias', 'Administración de Personal Docente Universitario', 'Designación de Cargos Académicos Universitarios', 'Gestión de Viajes Universitarios']

@app.route('/')
def index():
    opciones = ['Analizador', 'Buscador']
    data = {
        'Titulo': 'ARU - Busqueda de Resoluciones Universitarias',
        'Bienvenida': 'Universidad Mayor de San Andrés',
        'Opciones': opciones,
        'Numero_op': len(opciones)
    }
    return render_template('index.html')

@app.route('/analizador')
def analizador():
    data = {
        'etiquetas': etiquetas,
        'titulo': '',
        'textoOCR': '',
        'nro_p': 0,
        'paginas': '',
        'preprocesado': '',
        'topico': '',
        'mayor': 0
    }
    return render_template('analizador.html', data=data)

@app.route('/procesar', methods=["GET", 'POST'])
def procesar():
    if request.method == "POST":
        documento = request.files['documento']
        if documento.filename == "":
            return "Archivo no encontrado."
        filename = documento.filename
        documento.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        texto, paginas, titulo = pdf_a_imagen(str(app.config['UPLOAD_FOLDER'] + '\\'+ filename))
        
        texto_pre = preprocesamiento(texto)
        res, m = prediccion(texto_pre)       
        
        data = {
            'etiquetas': etiquetas,
            'titulo': titulo,
            'textoOCR': str(texto),
            'nro_p': len(paginas),
            'paginas': paginas,
            'preprocesado': texto_pre,
            'topico': res,
            'mayor': m
        }
        return render_template('analizador.html', data=data)
        #return str(app.config['UPLOAD_FOLDER'] + '\\'+ filename)
        #return redirect(url_for("get_file", filename=filename))

#Conversion de PDF a texto con OCR
def pdf_a_imagen(ruta):
    pag = convert_from_path(ruta, 500, poppler_path=r".\config\poppler-0.68.0\bin")
    nombreArchivo = os.path.split(ruta)[1] 
    txt = ""
    paginas = []
    for i in range(len(pag)):
        nombreImagen = './app/static/extra/' + nombreArchivo[0:-4] + '_pag' + str(i) + '.jpg'
        pag[i].save(nombreImagen, 'JPEG')
        #Abrimos la imagen
        imagen = cv.imread(nombreImagen)
       
        #Filtro de aclaracioin de letras
        imagen = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        imagen = cv.GaussianBlur(imagen,(5,5),0)
        rt_imagen, imagen = cv.threshold(imagen, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        imagen = cv.imwrite(nombreImagen, imagen)
        txt = txt + " " + imagen_a_texto(nombreImagen)
        paginas.append(str(nombreArchivo[0:-4] + '_pag' + str(i) + '.jpg'))
        titulo = nombreArchivo[0:-4]
    return txt, paginas, titulo

def imagen_a_texto(nombreImagen):
    pytesseract.pytesseract.tesseract_cmd = r".\config\Tesseract-OCR\tesseract"
    imagenP = cv.imread(nombreImagen)
    # Le pasamos como argumento la imagen abierta   
    texto = pytesseract.image_to_string(imagenP, lang='spa')

    # Mostramos el resultado
    #print(texto)
    return texto

#Proceso de Preprocesamiento
def preprocesamiento(texto):
    texto_preprocesado = tokenizacion(texto)
    texto_preprocesado = normalizacion(texto_preprocesado)
    texto_preprocesado = remover_stopwords(texto_preprocesado)
    texto_preprocesado = lematizacion(texto_preprocesado)
    texto_preprocesado = remover_rae(texto_preprocesado)
    return texto_preprocesado

#Tokenización
def tokenizacion(texto):
    texto_tokenizado = word_tokenize(texto)
    return texto_tokenizado

#Normalización
def normalizacion(texto):
    texto_normalizado = minusculas(texto)
    texto_normalizado = remover_puntuacion(texto_normalizado)
    texto_normalizado = remover_numeros(texto_normalizado)
    return texto_normalizado

def minusculas(texto):
    nuevo_texto = []
    for palabra in texto:
        nuevo_texto.append(palabra.lower())
    return nuevo_texto

def remover_puntuacion(texto):
    nuevo_texto = []
    for palabra in texto:
        palabra_nueva = re.sub(r'[^\w\s]', '', palabra)
        if palabra_nueva != '':
            nuevo_texto.append(palabra_nueva)
    return nuevo_texto

def remover_numeros(texto):
    nuevo_texto = []
    for palabra in texto:
        if not palabra.isdigit():
            nuevo_texto.append(palabra)
    return nuevo_texto

#Stopwords
def remover_stopwords(texto):
    texto_sin_stopwords = filtrar_stopwords(texto)
    texto_sin_stopwords = remover_palabras3letras(texto_sin_stopwords)
    return texto_sin_stopwords

def filtrar_stopwords(texto):
    STOPWORDS = set(stopwords.words("spanish"))
    nuevo_texto = []
    stopwords_extra = ["umsa", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre", "universidad", "mayor", "andrés", "honorable", "consejo", "lite", "lito", "vistos", "considerando", "facultativo", "facultad", "ciencias", "puras", "naturales", "regístrese", "registrese", "comuniquese", "archivese", "fcpn", "bolivia", "resolución", "universitario", "sesión", "ordinaria", "fecha", "conformidad", "articulo", "inciso", "rectorado", "rectoral", "aprueba", "carpeta", "comité", "ejecutivo"]
    for palabra in texto:
        if palabra not in STOPWORDS:
            if palabra not in stopwords_extra:
                nuevo_texto.append(palabra)
    return nuevo_texto

def remover_palabras3letras(texto):
    nuevo_texto = []
    for palabra in texto:
        if len(palabra) > 3:
            nuevo_texto.append(palabra)
    return nuevo_texto

#Lematización
def lematizacion(texto):
    nuevo_texto = []
    nlp = spacy.load("es_core_news_sm")
    txt = " ".join(texto)
    doc = nlp(txt)
    for token in doc:
        nuevo_texto.append(token.lemma_)
    #nuevo_texto = [token.lemma_ for token in doc]
    return nuevo_texto

#RAE
def remover_rae(texto):
    ruido = []
    nuevo_texto = []
    for palabra in texto:
        res = dle.search_by_word(word=palabra)
        try:
            res = res.to_dict()
            if len(res) != 1 or (palabra=='cogobierno' and palabra=='semipresencial'):
                nuevo_texto.append(palabra)
            else:
                ruido.append(palabra)
        except AttributeError:
            print("Error inesperado.")
    return nuevo_texto

#Predicción del Tópico
def prediccion(texto):
    diccionario = corpora.Dictionary.load('./config/Diccionario.p')
    max_longitud = 540
    bow_texto = diccionario.doc2bow(texto)
    X_ids = obtener_id(bow_texto)
    X_ids =X_ids.reshape(1, X_ids.shape[0])
    X_ids = pad_sequences(X_ids, maxlen = max_longitud)
    modelo = tf.keras.models.load_model('./config/anailisis_resoluciones.h5')
    res = modelo.predict(X_ids)
    res, m = prediccion_topico(res[0])
    return res, m

def obtener_id(corpus):
    ids = np.array([corpus[k][0] for k in range(len(corpus))])
    return ids

def prediccion_topico(res):
    res_final = []
    max = 0
    j = 0
    for i in res:
        if i > max:
            max = i
            pos = j
        res_final.append(round(i * 100, 2))
        j += 1
    return res_final, pos

@app.route('/buscador')
def buscador():
    data = {
        'etiquetas': etiquetas,
        'tabla': '',
        'cantidad': '',
        'topico': '',
        'pos': '',
        'texto': ''
    }
    return render_template('buscador.html', data=data)

@app.route('/buscar', methods=["GET", 'POST'])
def buscar():
    if request.method == "POST":
        df = pd.read_csv('./config/Dataset_Res.csv', encoding='utf-8')
        texto = request.form['texto']
        texto_pre = preprocesamiento_busqueda(texto)
        topico = topico_busqueda(texto_pre)
        topico, pos = topico_representativo(topico)
        tabla = buscar(df, texto_pre, pos)
        cantidad = len(tabla)
        data = {
            'etiquetas': etiquetas,
            'tabla': tabla,
            'cantidad': cantidad,
            'topico': topico,
            'pos': pos,
            'texto': texto
        }
    return render_template('buscador.html', data=data)

def preprocesamiento_busqueda(texto):
    texto_preprocesado = tokenizacion(texto)
    texto_preprocesado = normalizacion(texto_preprocesado)
    texto_preprocesado = remover_stopwords(texto_preprocesado)
    texto_preprocesado = lematizacion(texto_preprocesado)
    return texto_preprocesado

def buscar(df, texto, top):
    res_doc = df[df['tokens'].apply(lambda sentence: any(palabra in sentence for palabra in texto))]
    res_doc = res_doc[res_doc.tópico == top]
    res_doc = res_doc.drop(['tokens'], axis=1)
    res_doc = res_doc.to_dict('records')
    return res_doc

def topico_busqueda(texto):
    diccionario = corpora.Dictionary.load('./config/Diccionario.p')
    corpus = corpora.MmCorpus('./config/BoW_corpus.mm')
    modeloLDA = models.LdaMulticore.load('./config/Modelo_LDA_TFIDF.model')
    tfidf = models.TfidfModel(corpus)
    bow_texto = diccionario.doc2bow(texto)
    tfidf_texto = tfidf[bow_texto]
    distribucion_res = modeloLDA.get_document_topics(tfidf_texto, minimum_probability=0)
    return distribucion_res

def topico_representativo(distribucion):
    nueva_distribucion = []
    max = 0
    for i, t in distribucion:
        nueva_distribucion.append(round(t * 100, 2))
        if t > max:
            max = t
            pos = i
    return nueva_distribucion, pos

@app.route('/vista/<filename>')
def get_file(filename):
    #return send_from_directory(os.path.abspath("./"), filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def pagina_no_encontrada(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug=True, port=5000)