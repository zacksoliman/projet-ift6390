import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
import string, re
import nltk
from sklearn.utils import shuffle
import os.path
from keras.utils.visualize_util import plot
from tweet_analysis.Tools import data_processing as dp


###################
#   DATA STUFF    #
###################


print('Preprocessing data...')

data = get_clean_data()
tweets = data['processed_text']
sentiment = data['sentiment']
sentiment_cats = pd.Categorical.from_array(sentiment)
sentiment = to_categorical(sentiment_cats.codes) # one-hot encoding for sentiment classes


vocab = set()
vocab.add('OOV') # For out of vocabulary words
# max number of timesteps (chars in our case)
max_len = 150

# learn vocab
for tweet in tweets:
    for c in tweet:
        vocab.add(c)

vocab = list(vocab)

char_id = {ch:i for i, ch in enumerate(vocab)}
id_char = {i:ch for i, ch in enumerate(vocab)}

# using bool to reduce memory usage
X = np.zeros((len(tweets), max_len, len(vocab)), dtype=np.bool)
y = np.zeros((len(tweets), len(sentiment_cats.categories)), dtype=np.bool)

print('Formating input and targets...')

# set the appropriate indices to 1 in each one-hot vector
for i, train_example in enumerate(tweets):
    for timestep, char in enumerate(train_example):
        X[i, timestep, char_id[char]] = 1 # one hot encodings of tweet characters
    #y[i, sentiment[i]] = 1 # one hot encoding for the sentiment categories

X, y = shuffle(X, y, random_state=0)
X_train = X[:8999]
X_test = X[8999:]

y_train = sentiment[:8999]
y_test = sentiment[8999:]

##############
#   MODEL    #
##############

print('Building model...')

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(max_len, len(vocab))))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
plot(model, to_file='model.png', show_shapes=True)
print('Start fitting...')

hist = model.fit(X_train, y_train, batch_size=100, nb_epoch=20, validation_data=(X_test, y_test))

model.save('my_model.h5')
