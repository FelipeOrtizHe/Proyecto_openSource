import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

def load_emotions_dataset():
    # Cargar los datos CSV
    validation = pd.read_csv(r'C:\Users\felip\OneDrive\Escritorio\Programacion\Python\projectos\Hard\tweets\validation.csv')
    train = pd.read_csv(r'C:\Users\felip\OneDrive\Escritorio\Programacion\Python\projectos\Hard\tweets\training.csv')
    test = pd.read_csv(r'C:\Users\felip\OneDrive\Escritorio\Programacion\Python\projectos\Hard\tweets\test.csv')


    # Definir características (features) y etiquetas de clase
    class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    ft = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

    # Construir DatasetDict
    emotions = DatasetDict({
        "train": Dataset.from_pandas(train, features=ft),
        "test": Dataset.from_pandas(test, features=ft),
        "validation": Dataset.from_pandas(validation, features=ft)
    })

    return emotions

