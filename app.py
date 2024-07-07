import panel as pn
from data import load_emotions_dataset
from bar import barra,qwords

# Función para mostrar un panel con Tabulator
def show_panel(df):
    return pn.widgets.Tabulator(df.head(20),
                                show_index=False,
                                pagination='local', 
                                page_size=10)

# Función para convertir etiquetas de entero a cadena
def label_int2str(row, train_ds):
    return train_ds.features["label"].int2str(row)

# Función principal para ejecutar y mostrar el panel
def main():
    # Cargar y construir el DatasetDict desde data.py
    emotions = load_emotions_dataset()

    # Convertir el Dataset a DataFrame pandas y mostrar en el panel
    train_ds = emotions["train"]
    df = train_ds.to_pandas()
    df["label_name"] = df["label"].apply(lambda x: label_int2str(x, train_ds))
    panel = show_panel(df)
    
    # Mostrar el panel utilizando Panel
    pn.extension()
    pn.Column(panel.servable()).servable()

# barra()
qwords()
if __name__ == "__main__":
    main()


