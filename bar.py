import pandas as pd
import plotly.express as px
from data import load_emotions_dataset  # Importa la función o variable necesaria desde data.py

def barra():
# Cargar los datos utilizando la función show_data() de data.py
    train, validation, test = load_emotions_dataset()

# Suponiendo que deseas trabajar con el conjunto de entrenamiento (`train`), puedes hacer lo siguiente:
# Aquí agregamos un ejemplo simple de datos para ilustrar el proceso.
    data = {'label_name': ['joy', 'anger', 'joy', 'sadness', 'fear', 'joy', 'surprise', 'anger']}
    df = pd.DataFrame(data)

# Graficar usando Plotly Express
    fig = px.bar(df['label_name'].value_counts().sort_index(ascending=True), template='plotly_white')

# Mostrar la figura
    fig.show()

def qwords():
    # Llamar a la función load_emotions_dataset() desde data.py para obtener los datos
    emotions = load_emotions_dataset()

    # Obtener el conjunto de entrenamiento
    train_ds = emotions["train"]

    # Convertir el conjunto de entrenamiento a DataFrame de pandas
    train = train_ds.to_pandas()

    # Convertir las etiquetas numéricas a etiquetas de cadena
    def label_int2str(row):
        return train_ds.features["label"].int2str(row)

    train["label_name"] = train["label"].apply(label_int2str)

    # Verificar los datos cargados y transformados
    print(train.head())  # Asegúrate de que los datos y la columna 'label_name' estén presentes

    # Calcular el número de palabras por tweet
    train["Words Per Tweet"] = train["text"].str.split().apply(len)

    # Crear el gráfico de caja usando Plotly Express
    fig = px.box(train, y='Words Per Tweet', color='label_name', template='plotly_white')

    # Mostrar la figura
    fig.show()

# Llama a la función qwords para ejecutarla
