
from pathlib import Path
import joblib
import numpy as np

from gensim.models import Word2Vec

from preprocessing import clean_text


# -----------------------------------
# 1. Project Paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"


# -----------------------------------
# 2. Load Saved Models
# -----------------------------------

word2vec_model = Word2Vec.load(
    str(MODELS_DIR / "word2vec.model")
)

classifier = joblib.load(
    MODELS_DIR / "logistic_regression.pkl"
)


# -----------------------------------
# 3. AvgWord2Vec Function
# -----------------------------------

def average_word_vectors(tokens, model):

    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# -----------------------------------
# 4. Predict SMS Message
# -----------------------------------

def predict_message(message):

    tokens = clean_text(message)

    sentence_vector = average_word_vectors(
        tokens,
        word2vec_model
    )

    sentence_vector = sentence_vector.reshape(1, -1)

    prediction = classifier.predict(sentence_vector)[0]

    probabilities = classifier.predict_proba(
        sentence_vector
    )[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print("\nMessage:", message)
    print("Prediction:", result)
    print("Ham Probability:", round(probabilities[0] * 100, 2), "%")
    print("Spam Probability:", round(probabilities[1] * 100, 2), "%")


# -----------------------------------
# 5. User Input
# -----------------------------------

if __name__ == "__main__":

    message = input("\nEnter an SMS message: ")

    predict_message(message)
