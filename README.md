clasificacion de texto 

construiremos un sistema que pueda identificar automaticamente los estados emocionale, que las empresas expresan en twitter.

para esto usaremos una variante de BERT: DistilBERT, es un modelo mas pequeño que BERT, pero esto es mas eficiente, ademas, usaremos tres bibliotecas principales del ecosistema
Hugging face: Datasets,Tokenizers y transformers.

para esto utilizaremos la clasificacion binaria, para tener precision al momento de la clasicacion

ls /kaggle/input/emotion-dataset/

se utilizara para acceder a los archivos necesarios 
en este caso seran

test.cv
training.csv
validation.csv
