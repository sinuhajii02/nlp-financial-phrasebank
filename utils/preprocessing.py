import re

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {l : i for i, l in enumerate(LABELS)}
FILTERS = '!"#$&()*+,-./:;=?@[\\]^_`{|}~\t\n' 

MAX_LEN = 64

def encode_labels(labels):
    return labels.map(LABEL_TO_ID).to_numpy()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+([.,]\d+)*", " <num> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean(texts):
    return texts.apply(clean_text)

def fit_tokenizer(texts):
    tok = Tokenizer(oov_token="<OOV>", filters=FILTERS)
    tok.fit_on_texts(clean(texts))
    return tok

def encode_texts(tok, texts):
    return pad_sequences(tok.texts_to_sequences(clean(texts)), maxlen=MAX_LEN, padding="post")