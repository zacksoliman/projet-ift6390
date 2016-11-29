import matplotlib.pyplot as plt
import pandas as pd
from Tools.data_processing import get_clean_data
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

data = get_clean_data()
categories = data.sentiment.unique()
categories  = categories.tolist() #neutre, positif, négatif

#set target y and predictor x
print 'set target and predictor'
x = data.processed_text.values
y_labelname = data.sentiment.values
y_categories = pd.Categorical.from_array(y_labelname)
y= y_categories.codes #transform sentiment labels into 0,1,2

# Déterminer au hasard des indices pour les exemples d'entrainement et de test
print 'creation des data train test'
n_train = (data.shape[0]*2) // 3
inds = [i for i in range(data.shape[0])]
random.shuffle(inds)
train_inds = inds[:n_train]
test_inds = inds[n_train:]

x_train = x[train_inds]	# garder les bonnes lignes
x_test = x[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]

#bag of words representation: vectorizing
#1-grams
print '1-grams'
count_vect = CountVectorizer() #counts occurences of each word
X_train_counts = count_vect.fit_transform(x_train).toarray()
X_test_counts = count_vect.transform(x_test).toarray()
#X_train_counts=X_train_counts.toarray()
#tfidf_transformer = TfidfVectorizer() #applies transformation to account for frequency
#X_train_tfidf = tfidf_transformer.fit_transform(x_train)
#X_test_tfidf= tfidf_transformer.fit_transform(x_test)

#including 2-grams
#bigram_vect = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
#X_train_bigramcounts = bigram_vect.fit_transform(x_train)

#%%
# notre exemple de test
x_train_keras= X_train_counts
x_test_keras=X_test_counts
y_train_keras = to_categorical(y_train,3)
y_test_keras = to_categorical(y_test,3)
# reseau de neurone avec keras


model = Sequential()
model.add(Dense(10, input_dim=X_test_counts.shape[1]))
model.add(Activation('tanh'))
model.add(Dense(3))
model.add(Activation('softplus'))
sgd = SGD(0.1)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#%%

logs = model.fit(x_train_keras, y_train_keras, nb_epoch=30,
                 validation_data=(x_test_keras, y_test_keras))


# graphique classe correct
plt.plot(logs.history['acc'], label='train')
plt.plot(logs.history['val_acc'], label='valid')
plt.legend()
plt.title('Pourcentage de classes correctes')
plt.show()
plt.plot(logs.history['loss'], label='train')
plt.plot(logs.history['val_loss'], label='valid')
plt.title('Fonction objectif sur l\'ensemble d\'entrainement')
plt.show()
