from transformers import pipeline

def load_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def classify_text(classifier, text):
    return classifier(text, return_all_scores=True)