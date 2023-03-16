import sys
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def train_model(X_train, y_train):

    nb = Pipeline([('tfidf', TfidfVectorizer(lowercase=False, token_pattern='\w+', ngram_range=(1, 2), min_df=3)),
                           ('clf', MultinomialNB())])
    nb.fit(X_train, y_train)
    print("Training of Naive Bayes done")

    return nb


def save_pipeline(pipeline: Pipeline, save_path: str) -> None:
    
    """Save the sklearn Pipeline object"""
    
    joblib.dump(pipeline, save_path)


def load_pipeline(file_path: str) -> Pipeline:
    
    """Load a saved Sklearn Pipeline object"""
    
    try:
        trained_model = joblib.load(filename=file_path)
    except Exception as e:
        print(e)
        print("Make sure you have the trained model in your current directory")
        sys.exit(1)

    return trained_model
