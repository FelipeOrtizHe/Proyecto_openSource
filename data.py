
import pandas as pd


def detect_text_column(data):
    # Asumimos que la columna con las frases tendrá la mayor longitud promedio de cadenas
    text_column = None
    max_avg_length = 0

    for column in data.columns:
        if data[column].dtype == object:  # Solo consideramos columnas de tipo string
            avg_length = data[column].str.len().mean()
            if avg_length > max_avg_length:
                max_avg_length = avg_length
                text_column = column

    return text_column