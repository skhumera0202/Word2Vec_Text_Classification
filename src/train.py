
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import clean_text


# -----------------------------------
# 1. Project Paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------------
# 2. Load Dataset
# -----------------------------------

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    names=["label", "message"]
)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# -----------------------------------
# 3. Convert Labels into Numbers
# -----------------------------------

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# -----------------------------------
# 4. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# -----------------------------------
# 5. Text Preprocessing
# -----------------------------------

X_train_tokens = X_train.apply(clean_text)
X_test_tokens = X_test.apply(clean_text)

print("\nText Preprocessing Completed!")


# -----------------------------------
# 6. Train Word2Vec Model
# -----------------------------------

print("\nTraining Word2Vec Model...")

word2vec_model = Word2Vec(
    sentences=X_train_tokens.tolist(),
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=20,
    seed=42
)

print("Word2Vec Training Completed!")


# -----------------------------------
# 7. Convert Sentences into
#    AvgWord2Vec Vectors
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


X_train_vectors = np.array([
    average_word_vectors(tokens, word2vec_model)
    for tokens in X_train_tokens
])

X_test_vectors = np.array([
    average_word_vectors(tokens, word2vec_model)
    for tokens in X_test_tokens
])

print("\nAvgWord2Vec Conversion Completed!")

print("Training Vector Shape:", X_train_vectors.shape)
print("Testing Vector Shape:", X_test_vectors.shape)


# -----------------------------------
# 8. Train Classification Model
# -----------------------------------

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

classifier.fit(X_train_vectors, y_train)

print("Classifier Training Completed!")


# -----------------------------------
# 9. Make Predictions
# -----------------------------------

y_pred = classifier.predict(X_test_vectors)


# -----------------------------------
# 10. Model Evaluation
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("WORD2VEC TEXT CLASSIFICATION RESULTS")
print("===================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


# -----------------------------------
# 11. Confusion Matrix
# -----------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title("Word2Vec - AvgWord2Vec Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "word2vec_confusion_matrix.png"
)

plt.close()

print("\nConfusion Matrix Saved!")


# -----------------------------------
# 12. Save Models
# -----------------------------------

word2vec_model.save(
    str(MODELS_DIR / "word2vec.model")
)

joblib.dump(
    classifier,
    MODELS_DIR / "logistic_regression.pkl"
)

print("\nModels Saved Successfully!")

print("\nTraining Pipeline Completed!")
# -----------------------------------
# 13. Random Forest Classification
# -----------------------------------

print("\nTraining Random Forest Classifier...")
rf_classifier = RandomForestClassifier(
    random_state=42
)

rf_classifier.fit(X_train_vectors, y_train)

# Predictions
rf_predictions = rf_classifier.predict(X_test_vectors)

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================")

print("\nAccuracy:", rf_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Ham", "Spam"]
    )
)

# Save Random Forest model
joblib.dump(
    rf_classifier,
    MODELS_DIR / "random_forest.pkl"
)

print("\nRandom Forest Model Saved!")

# -----------------------------------
# 14.  Random Forest Confusion Matrix
# -----------------------------------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

rf_cm = confusion_matrix(y_test, rf_predictions)

rf_display = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Ham", "Spam"]
)

rf_display.plot(cmap="Oranges")

plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nRandom Forest Confusion Matrix Saved!")

#------------------------------------
# 15. Model Comparison Chart
# -----------------------------------

import matplotlib.pyplot as plt
import numpy as np

models = ["Logistic Regression", "Random Forest"]
accuracies = [accuracy, rf_accuracy]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for bar, score in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.2%}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "model_comparison.png",
    dpi=300
)

plt.show()

print("\nModel comparison chart saved!")