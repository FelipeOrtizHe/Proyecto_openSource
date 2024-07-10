# model.py

import matplotlib.pyplot as plt
from transformers import pipeline

def initialize_classifier(model_name='bhadresh-savani/distilbert-base-uncased-emotion'):
    classifier = pipeline("text-classification", model=model_name, top_k=None)
    return classifier

def plot_emotion_scores(prediction, filename, title):
    # Extraemos etiquetas y puntuaciones
    labels = [item['label'] for item in prediction[0]]
    scores = [item['score'] for item in prediction[0]]

    # Creamos la gráfica de barras
    plt.figure(figsize=(12, 6))  
    bars = plt.bar(labels, scores, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Scores')
    plt.title(title)  
    plt.xticks(rotation=45, ha='right') 


    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{score:.2f}', ha='center', va='bottom')


    plt.suptitle(title, y=1.05, fontsize=16, fontweight='bold', ha='center')

    # Ajustes de diseño
    plt.tight_layout()

    # Guardamos la gráfica en un archivo
    plt.savefig(filename)
    plt.close()
