import os
import spacy
import pandas as pd

try:
    spacy_en = spacy.load("en_core_web_sm")
except:
    os.system('python -m spacy download en_core_web_sm')
    spacy_en = spacy.load("en_core_web_sm")

stops_spacy = sorted(spacy.lang.en.stop_words.STOP_WORDS)
stops_spacy.extend(["is", "to"])


def remove_punctuation(text):
    text = ''.join([char if char.isalnum() or char == ' ' else ' ' for char in text])
    text = ' '.join(text.split())  # remove multiple whitespace
    return text


def remove_stopwords_spacy(text, stopwords=stops_spacy):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


def concat_multiple_documents(list_of_dataframes):
    if len(list_of_dataframes) > 1:
        df = pd.concat(list_of_dataframes)
        return df
    return list_of_dataframes[0]


def preprocess(data):
    data["row"] = data["row"].apply(remove_punctuation)
    data["row"] = data["row"].apply(remove_stopwords_spacy)
    return data