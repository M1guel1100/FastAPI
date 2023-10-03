import os
import numpy as np
import pandas as pd
import psycopg2
import random
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Manejar los origenes que se permiten en el microservicio, ponienod la ip del servidor donde se aloja la página
origins = [
    "http://127.0.0.1:5500", 
]

# Permite acceso completo a los origenes especificados con anterioridad
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de la base de datos PostgreSQL
conexion = psycopg2.connect(
    host="pg-365cc6ea-modular.aivencloud.com",
    port="10496",
    database="modularbd",
    user="avnadmin",
    password="AVNS_TLo1tmhPI2GwYvBAN6o"
)

# Cargar datos desde la base de datos
preguntas_data = pd.read_sql_query("SELECT * FROM preguntas", conexion)
pregunta_carrera_data = pd.read_sql_query("SELECT * FROM preguntas_carreras", conexion)
carrera_data = pd.read_sql_query("SELECT * FROM carreras", conexion)

#Variables Globales
selected_careers = []
asked_questions = set()
MAX_QUESTIONS = 15
current_question_id = None
contador = 0
user_responses = []
carrera_recomendada = None
model_filename = "modelo.h5"  # Ruta completa para guardar el modelo

# Función para cargar o crear el modelo
def load_or_create_model(input_shape, output_shape):
    if os.path.exists(model_filename):
        return load_model(model_filename)
    else:
        # El modelo es del tipo secuencial (librería Keras)
        model = Sequential([
            # Primera capa de la red neuronal con 128 neuronas, que espera entradas con 16 caracteristicas
            Dense(128, activation='relu', input_shape=(16,)),
            # Segunda capa de la red neuronal con el mismo numero de neuronas como de carreras
            Dense(output_shape, activation='softmax')
        ])
        # ADAM algoritmo de optimización para entrenamiento de la red neuronal.
        # Se guarda el modelo en el directorio especificado
        model.save(model_filename)
        return model

# Cargar o crear el modelo
input_shape = len(carrera_data) #76 carreras
output_shape = len(carrera_data) #76 carreras
model = load_or_create_model(input_shape, output_shape)


# Función para entrenar el modelo con los datos de la tabla respuestas_usuario
def train_model_with_user_responses():
    global model
    # Cargar los datos de la tabla respuestas_usuario
    response_data = pd.read_sql_query("SELECT respuestas, carrera_recomendada_id FROM respuestas_usuario", conexion)

    # Procesar los datos con un procesamiento de texto separandolos en pares identificando comas, parentesis y signos.
    def process_responses(respuestas):
        response_list = respuestas.strip('()').split(',')
        processed_responses = [int(pair.split(':')[1].rstrip(')')) for pair in response_list]
        return processed_responses

    # Aquí se aplica la función process_responses a cada fila de la columna 'respuestas' del DataFrame response_data. 
    # Esto crea una nueva serie responses que contiene listas de respuestas procesadas para cada entrada en la columna 'respuestas'.
    responses = response_data['respuestas'].apply(process_responses)
    # Se calcila la  longitud máxima de las listas de respuestas en la serie responses. 
    # Esto se utilizará para asegurarse de que todas las listas tengan la misma longitud cuando se creen las matrices NumPy.
    max_length = max(len(resp) for resp in responses)
    # Creación de la matriz NumPy x (Fila corresponiente a las respuestas procesadas)
    X = np.array([resp + [0] * (max_length - len(resp)) for resp in responses])
    # Creación de la matrix Numpy y (Columa corresponiente a Carrera Recomendada)
    y = response_data['carrera_recomendada_id'].values
    # Ajuste del indice
    y = y - 1  

    # Codificar etiquetas de carrera como one-hot
    # Cada etiqueta se convierte en un vector binario donde un valor específico está en 1 y los demás están en 0, de acuerdo con la carrera recomendada.
    y_one_hot = to_categorical(y, output_shape)

    # Entrenar el modelo
    model.fit(X, y_one_hot, epochs=50, batch_size=32)

    try:
        # Guardar el modelo entrenado
        model.save(model_filename)
        print(f"Modelo guardado en: {model_filename}")
    except Exception as e:
        print(f"Error al guardar el modelo: {str(e)}")



# Ruta de entrenamiento del modelo (POSTMAN)
@app.post("/train_model")
def train_model():
    # Llama a la función para entrenar el modelo
    train_model_with_user_responses()
    
    return {"message": "Modelo entrenado correctamente"}

@app.get("/")
def index():
    return {"message": "Bienvenido al test de orientación vocacional"}

# Ruta para reiniciar la API
@app.post("/reset_api")
def reset():
    global selected_careers, asked_questions, current_question_id, contador, user_responses, carrera_recomendada, model

    # Cerrar el modelo si está abierto
    model = None
    # Formateo del modelo
    if os.path.exists(model_filename):
        os.remove(model_filename)

    # Crea un nuevo modelo
    model = load_or_create_model(input_shape, output_shape)
    # Reset a todas las variables globales
    selected_careers = []
    asked_questions = set()
    current_question_id = None
    contador = 0
    user_responses = []
    carrera_recomendada = None
    # Entrenamiento de la red neuronal
    train_model_with_user_responses()
    return {"message": "Estado reiniciado"}

# Ruta Para Obtener Pregunta
@app.get("/get_question")
def get_question():
    global asked_questions, current_question_id, contador

    # VALIDACION DE ERRORES
    # Validación para el contador en caso de superar el numero máximo de preguntas
    if contador == MAX_QUESTIONS:
        raise HTTPException(status_code=404, detail="No se encontraron más preguntas disponibles.")

    # Lista que contiene ID de preguntas que no han sido preguntadas
    available_questions = [qid for qid in preguntas_data['preguntaid'] if qid not in asked_questions]
    # Si la lista de preguntas disponibles esta vacía devuelve error
    if not available_questions:
        raise HTTPException(status_code=404, detail="No hay más preguntas disponibles.")

    # Eleccion de pregunta al azar
    # ID de la pregunta actual se almacena en current_question_id
    current_question_id = random.choice(available_questions)
    # Obtener el texto de la pregunta desde el DataFrame utilizando .loc[] y se almacena en next_question_text
    next_question_text = preguntas_data.loc[preguntas_data['preguntaid'] == current_question_id, 'textopregunta'].values[0]
    # Agregar pregunta a preguntas ya hechas para evitar preguntas repetidas
    asked_questions.add(current_question_id)
    # Se devuelve una respuesta JSON que contiene el id de la pregunta y el texto
    return {"pregunta_id": current_question_id, "question": next_question_text}

# Ruta Para Mandar Respuesta
@app.post("/submit_answer")
def submit_answer(answer: int = Form(...), pregunta_id: int = Form(...)):
    global selected_careers, asked_questions, current_question_id, user_responses, contador, model
    # Por cada pregunta contestada el contador aumenta
    contador += 1
    #  Se obtienen las carreras relacionadas con la pregunta actual desde pregunta_carrera_data y se agregan a la lista selected_careers
    #  Esto se hace para realizar un seguimiento de las carreras que podrían ser relevantes para el usuario en función de sus respuestas.
    if answer:
        related_careers = pregunta_carrera_data[pregunta_carrera_data['preguntaid'] == current_question_id]['carreraid']
        selected_careers.extend(related_careers)

    # Agrega las respuestas del usuario como una tupla a la lista user_responses para guardarse en la base de datos al final del test
    user_responses.append((pregunta_id, answer))

    # VALIDACIÓN DE PREGUNTAS CONTESTADAS
    if contador == MAX_QUESTIONS:

        """# Contar la frecuencia de las carreras seleccionadas
        career_counts = dict()
        for career_id in selected_careers:
            if career_id in career_counts:
                career_counts[career_id] += 1
            else:
                career_counts[career_id] = 1

        # Encontrar la carrera más común (la que tiene más acumulaciones)
        recommended_career_id = max(career_counts, key=career_counts.get)
        recommended_career = carrera_data[carrera_data['carreraid'] == recommended_career_id]['nombrecarrera'].values[0]
        carrera_recomendada = recommended_career_id"""


        # Realizar la predicción con el modelo
        # Se crea un array input_data lleno de ceros con forma (1, 16) para ajustarse al modelo
        input_data = np.zeros((1, 16))

        # Recorre la lista de las respuestas del usuario
        for pregunta_id, respuesta in user_responses:
            if pregunta_id <= 16: 
                input_data[0, pregunta_id - 1] = respuesta

        # Se utiliza la funcion predict para obtener resultados a partir del modelo de red neuronal.
        # Toma el array que se creo con las respuestas y predice la carrera recomendada
        predictions = model.predict(input_data)

        # Obtener la carrera recomendada con mayor probabilidad
        recommended_career_index_model = np.argmax(predictions)
        # Se obtiene el nombre de la carrera 
        recommended_career_model = carrera_data.loc[carrera_data.index == recommended_career_index_model + 1, 'nombrecarrera'].values[0]
        # Se almacena el id de la carrera para su posterior agregación a la base de datos
        carrera_recomendada = carrera_data.loc[carrera_data.index == recommended_career_index_model + 1, 'carreraid'].values[0]
        # Se convierte a int para que sea compatible con el tipo de dato en la base de datos
        carrera_recomendada = int(carrera_recomendada)

        # Obtener los centros relacionados como una lista para mostrarlos en la página web
        recommended_career_info = carrera_data[carrera_data['carreraid'] == recommended_career_index_model + 1].iloc[0]
        related_centers = recommended_career_info['centrosrelacionados']

        # Llama la función para guardar las respuestas y carrera en la bd, mandandole como parametros en arrelo con preguntas y carrera recomendada
        save_responses_to_database(user_responses, carrera_recomendada)

        print(recommended_career_model)

        return {"message": "Test completado", "recommended_career": recommended_career_model, "related_centers": related_centers,
                "recommended_career_model": recommended_career_model}

    return {"message": "Respuesta recibida"}

# Función para guardar en la base de datos
def save_responses_to_database(responses, carrera_recomendada):
    # Crear una cadena de texto con los pares (idpregunta, respuesta)
    response_text = ",".join(f"({pregunta_id}:{respuesta})" for pregunta_id, respuesta in responses)

    cursor = conexion.cursor()
    query = "INSERT INTO respuestas_usuario (respuestas, carrera_recomendada_id) VALUES (%s, %s)"
    # Carga las respuestas y carrera a la base de datos
    cursor.execute(query, (response_text, carrera_recomendada))
    conexion.commit()
    cursor.close()
