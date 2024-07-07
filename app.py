import panel as pn
import pandas as pd
from model import load_model, classify_text

def main():
    pn.extension()
    
    # Cargar modelo y hacer predicciones
    classifier = load_model()
    new_data = 'I watched a movie last night, it was quite brilliant'
    preds = classify_text(classifier, new_data)
    
    # Convertir predicciones a DataFrame para visualización
    df_preds = pd.DataFrame(preds[0])
    
    # Crear panel para visualización
    panel = pn.Column(
        pn.pane.Markdown("## Predictions"),
        pn.pane.DataFrame(df_preds)
    )
    panel.show()

if __name__ == "__main__":
    main()
