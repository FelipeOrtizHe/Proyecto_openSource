import pandas as pd

def load_emotions_dataset():
    train_df = pd.read_csv(r'C:\Users\felip\OneDrive\Escritorio\Programacion\clasificador_tweets\training.csv')
    return train_df

def get_first_n_records(df, n=5):
    return df.head(n)
