import panel as pn
import pandas as pd
import hvplot.pandas
from model import load_model, classify_text
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

def main():
    pn.extension('hvplot')
    
    # Cargar modelo y hacer predicciones
    classifier = load_model()
    new_data = 'I watched a movie last night, it was quite brilliant'
    preds = classify_text(classifier, new_data)
    
    # Convertir predicciones a DataFrame para visualización
    df_preds = pd.DataFrame(preds[0])
    
    # Crear panel para visualización
    panel = pn.Column(
        pn.pane.Markdown(f"## Dato de prueba:\n{new_data}"),
        pn.pane.Markdown("## Predictions"),
        pn.pane.DataFrame(df_preds),
        pn.pane.Markdown("## Gráfico de barra de predicciones"),
        df_preds.hvplot.bar(x='label', y='score', rot=90, width=500, height=400)
    )
    panel.show()

if __name__ == "__main__":
    main()