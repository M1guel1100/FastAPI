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
    "http://testvocacionaludeg.x10.mx",
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
        model = Sequential([
            Dense(128, activation='relu', input_shape=(16,)),
            Dense(output_shape, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

# Cargar o crear el modelo
input_shape = len(carrera_data)
output_shape = len(carrera_data)  # Debería ser 79 si tienes 79 carreras
model = load_or_create_model(input_shape, output_shape)

# ...




# Función para entrenar el modelo con los datos de la tabla respuestas_usuario
def train_model_with_user_responses():
    # Cargar los datos de la tabla respuestas_usuario
    response_data = pd.read_sql_query("SELECT respuestas, carrera_recomendada_id FROM respuestas_usuario", conexion)

    # Procesar los datos
    def process_responses(respuestas):
        response_list = respuestas.strip('()').split(',')
        processed_responses = [int(pair.split(':')[1].rstrip(')')) for pair in response_list]
        return processed_responses

    # Procesar los datos
    responses = response_data['respuestas'].apply(process_responses)
    max_length = max(len(resp) for resp in responses)
    X = np.array([resp + [0] * (max_length - len(resp)) for resp in responses])
    y = response_data['carrera_recomendada_id'].values
    y = y - 1  # Esto ajusta los índices a 0-78

    # Codificar etiquetas de carrera como one-hot
    y_one_hot = to_categorical(y, output_shape)

    # Entrenar el modelo
    model.fit(X, y_one_hot, epochs=50, batch_size=32)
    model.save(model_filename)
    print(f"Modelo guardado en: {model_filename}")


# Define una ruta para entrenar el modelo
@app.post("/train_model")
def train_model():
    # Llama a la función para entrenar el modelo
    train_model_with_user_responses()
    
    return {"message": "Modelo entrenado correctamente"}

@app.get("/")
def index():
    return {"message": "Bienvenido al test de orientación vocacional"}

@app.post("/reset_api")
def reset():
    global selected_careers, asked_questions, current_question_id, contador, user_responses, carrera_recomendada, model, carrera_recomendada_modelo
    selected_careers = []
    asked_questions = set()
    current_question_id = None
    contador = 0
    user_responses = []
    carrera_recomendada = None
    carrera_recomendada_modelo = None
    # model = load_or_create_model(input_shape, output_shape)
    # train_model()

    return {"message": "Estado reiniciado"}

@app.get("/get_question")
def get_question():
    global asked_questions, current_question_id, contador

    if contador == MAX_QUESTIONS:
        raise HTTPException(status_code=404, detail="No se encontraron más preguntas disponibles.")

    available_questions = [qid for qid in preguntas_data['preguntaid'] if qid not in asked_questions]

    if not available_questions:
        raise HTTPException(status_code=404, detail="No hay más preguntas disponibles.")

    current_question_id = random.choice(available_questions)
    next_question_text = preguntas_data.loc[preguntas_data['preguntaid'] == current_question_id, 'textopregunta'].values[0]
    asked_questions.add(current_question_id)
    return {"pregunta_id": current_question_id, "question": next_question_text}

@app.post("/submit_answer")
def submit_answer(answer: int = Form(...), pregunta_id: int = Form(...)):
    global selected_careers, asked_questions, current_question_id, user_responses, contador
    # Cargar el modelo dentro de la función
    # model = load_or_create_model(input_shape, output_shape)
    contador += 1
    if answer:
        related_careers = pregunta_carrera_data[pregunta_carrera_data['preguntaid'] == current_question_id]['carreraid']
        selected_careers.extend(related_careers)

    user_responses.append((pregunta_id, answer))

    if contador == MAX_QUESTIONS:
        # Contar la frecuencia de las carreras seleccionadas
        career_counts = dict()
        for career_id in selected_careers:
            if career_id in career_counts:
                career_counts[career_id] += 1
            else:
                career_counts[career_id] = 1

        # Encontrar la carrera más común (la que tiene más acumulaciones)
        recommended_career_id = max(career_counts, key=career_counts.get)
        recommended_career = carrera_data[carrera_data['carreraid'] == recommended_career_id]['nombrecarrera'].values[0]
        carrera_recomendada = recommended_career_id

        # Obtener los centros relacionados como una lista
        recommended_career_info = carrera_data[carrera_data['carreraid'] == recommended_career_id].iloc[0]
        related_centers = recommended_career_info['centrosrelacionados']

        # Realizar la predicción con el modelo
       # input_data = np.zeros((1, 16))  # Ajusta la forma de entrada a 16

       # for pregunta_id, respuesta in user_responses:
      #      if pregunta_id <= 16:  # Asegúrate de que no supere 16
      #          input_data[0, pregunta_id - 1] = respuesta
    
      #  predictions = model.predict(input_data)

        # Obtener la carrera recomendada con mayor probabilidad
     #   recommended_career_index_model = np.argmax(predictions)
       # recommended_career_model = carrera_data.loc[carrera_data.index == recommended_career_index_model + 1, 'nombrecarrera'].values[0]

        # Guardar las respuestas en la base de datos
        save_responses_to_database(user_responses, carrera_recomendada)
      #  print(recommended_career_model)
        return {"message": "Test completado", "recommended_career": recommended_career, "related_centers": related_centers}#, "recommended_career_model": recommended_career_model}

    # No obtengas la siguiente pregunta aquí, simplemente devuelve una respuesta
    return {"message": "Respuesta recibida"}


def save_responses_to_database(responses, carrera_recomendada):
    # Crear una cadena de texto con los pares (idpregunta, respuesta)
    response_text = ",".join(f"({pregunta_id}:{respuesta})" for pregunta_id, respuesta in responses)


    cursor = conexion.cursor()
    query = "INSERT INTO respuestas_usuario (respuestas, carrera_recomendada_id) VALUES (%s, %s)"
    cursor.execute(query, (response_text, carrera_recomendada))
    conexion.commit()
    cursor.close()
