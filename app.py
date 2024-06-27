import pandas as pd
import numpy as np
import panel as pn
import warnings; warnings.filterwarnings('ignore')
from datasets import Dataset,DatasetDict,Features,Value,ClassLabel
from google.colab import drive
import plotly.express as px
from transformers import AutoTokenizer


mount = drive.mount('/content/drive')


validation = pd.read_csv('/content/drive/MyDrive/DB/datasets_tweeter/validation.csv')
train = pd.read_csv('/content/drive/MyDrive/DB/datasets_tweeter/training.csv')
test = pd.read_csv('/content/drive/MyDrive/DB/datasets_tweeter/test.csv')

# EMOCIONES QUE SE VAN A UTILIZAR
class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# DEFINIMOS FORMALMENTE LA ESTRUCTURA QUE VA A TENER NUESTRI DATASET
feat = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

# CONVERTIMOS LOS DATAFRAME A DATASETS DE FORMATO DE HUGGING FACE (CONTENEDOR DE MULTIPLES DATASETS)
emotions = DatasetDict({
    "train": Dataset.from_pandas(train,features=feat),
    "test": Dataset.from_pandas(test,features=feat),
    "validation": Dataset.from_pandas(validation,features=feat)
    })



# DATA DE ENTRENAMIENTO
train_ds = emotions["train"]
train_ds[0:5]

# EN CASO DE QUE NECESITEMOS LA EN UNA DATAFRAME DE PANDAS (don't forget to reset)
emotions.set_format(type="pandas")
df = emotions["train"][:]

# CUANDO NECESITEMOS AGREGAR LOS LABES DE LA COLUMNA LABEL EN TEXTO
# Add label data to dataframe
def label_int_to_str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int_to_str)
df

# EN NUESTROS SET DE ENTRENAMIENTO TENEMOS 6 CLASES DE SENMIENTOS (TRISTESA, ALEGRIA, AMOR, IRA, MIEDO Y SORPRESA)
px.bar(df['label_name'].value_counts(ascending=True),template='plotly_dark')


# DISTILBERT SOLO RECIBE UN MAXIMO DE 512 TOKENS (division de texto en palabras iguales)
# LOS TWEETS REDONDEAN ENENTRE LAS 10 - 20 PALABRAS, POR LO TANTO SOBREPASARIA EL LIMITE

df["Words Per Tweet"] = df["text"].str.split().apply(len)
px.box(df,y='Words Per Tweet',
       color='label_name',
       template='plotly_dark')


# DistilBERT  NO PUEDE RECIBIR EL TWEET COMPLETO COM UN STRING, ASI QUE DEBE TOKENIZAR
# TENEMOS TOKENIZACION POR cCARACTERES, PALABRAS Y SUBPALABRAS

# TOKENIZANDO EL DATASET DE ENTRENAMIENTO


# SUBPALABRAS
text = 'Tokenisation of text is a core task of NLP.'

# CARGAMOS EL TOKENIZADOR DE DistilBERT
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# TOKENIZACION
print('Encoded text')
encoded_text = tokenizer(text)
print(encoded_text,'\n')

# CONVERSION DE LOS IDS A USUS RESPECTIVOS TOKENS (PALABRAS)
print('Tokens')
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens,'\n')

# RECONSTRUCCION DE LA FRASE INICIAL APARTIR DE LOS TOKENS
print('Convert tokens to string')
print(tokenizer.convert_tokens_to_string(tokens),'\n')


emotions.reset_format()

emotions

# FUNCION PARA TOKENIZAR
def tokenise(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# Esto tokeniza los primeros dos ejemplos del conjunto de entrenamiento.
ex_tokenised = tokenise(emotions["train"][:2])
ex_tokenised


# Show attention mask
ex_tokenised['attention_mask']