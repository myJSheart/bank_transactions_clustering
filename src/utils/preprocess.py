import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import string

# In bank statements, "transfer" can be considered as a stop word
stop_words = list(stopwords.words('english'))
stop_words.extend(['transfer', 'trf'])


def remove_stop_words(text, stop_words=stop_words):
    """ Remove stop words from a string

    Parameters:
        text: a string
        stop_words: a list of strings

    Returns:
        a string which does not contain any stop words
    """
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)


def remove_punctuation(text):
    return ' '.join([token for token in word_tokenize(text) if token not in string.punctuation])


def remove_number(text):
    return re.sub(r'\d+', '', text)


def preprocess_text(text):
    text = remove_stop_words(text)
    text = remove_punctuation(text)
    text = remove_number(text)
    return text
