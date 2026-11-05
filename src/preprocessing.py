import re


def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize into words
    words = text.split()

    return words