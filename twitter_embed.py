import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import gc

EMBED_FILE = 'glove.6B.100d.txt'
MAX_FEAT = 20000
MAX_LEN = 100
DIM = 100

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=MAX_FEAT)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_LEN)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_LEN)

word_index = tokenizer.word_index
embed_index = {}
f = open(EMBED_FILE)
for line in f:
    value = line.split()
    embed_index[value[0]] = np.asarray(value[1:], dtype='float32')
f.close()

embed_matrix = np.zeros((len(word_index)+1, DIM))
for word, idx in word_index.items():
    vec = embed_index.get(word)
    if vec is not None:
        embed_matrix[idx, :] = vec
print(embed_matrix.shape)

embed_layer = Embedding(input_dim=len(word_index)+1,
                        output_dim=DIM,
                        weights=[embed_matrix],
                        trainable=False)
