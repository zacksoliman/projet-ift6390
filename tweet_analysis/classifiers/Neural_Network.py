import matplotlib.pyplot as plt
import pandas as pd
from Tools.data_processing import get_clean_data
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer

#Importation des data
#Sentiment  election, Tweets  avion
namedata='Sentiment'
data = get_clean_data(namedata)


#set target y and predictor x
print 'set target and predictor'
if namedata=='Tweets':
    y_labelname = data.airline_sentiment.values
    categories = data.airline_sentiment.unique()
else:
    y_labelname = data.sentiment.values
    categories = data.sentiment.unique()

categories  = categories.tolist()

x = data.processed_text.values
y_categories = pd.Categorical.from_array(y_labelname)

y= y_categories.codes  #transform sentiment labels into 0,1,2



# Determiner au hasard des indices pour les exemples d'entrainement et de test
print 'creation des data train test'
n_train = (data.shape[0]*2) // 3
inds = [i for i in range(data.shape[0])]
np.random.shuffle(inds)
train_inds = inds[:n_train]
test_inds = inds[n_train:]

# garder les bonnes lignes
x_train = x[train_inds]
x_test = x[test_inds]
y_train = y[train_inds]
y_test = y[test_inds]



#bag of words representation: vectorizing
#1-grams
print '1-grams'
#counts occurences of each word
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train).toarray()
X_test_counts = count_vect.transform(x_test).toarray()
#X_train_counts=X_train_counts.toarray()
#tfidf_transformer = TfidfVectorizer() #applies transformation to account for frequency
#X_train_tfidf = tfidf_transformer.fit_transform(x_train)
#X_test_tfidf= tfidf_transformer.fit_transform(x_test)

#including 2-grams
#bigram_vect = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
#X_train_bigramcounts = bigram_vect.fit_transform(x_train)



#%% Keras
#notre exemple de test modifier au besoin de keras
x_train_keras= X_train_counts
x_test_keras=X_test_counts
y_train_keras = to_categorical(y_train,3)
y_test_keras = to_categorical(y_test,3)

#creation du reseau de neurone avec keras
#importation du reseau de neurone nn12
from Tools.KerasNN import NN1
model=NN1(np.shape(X_train_counts)[1])
#model=NN4()

#Entrainement du reseau de neurones
logs = model.fit(x_train_keras, y_train_keras, nb_epoch=8,
                 validation_data=(x_test_keras, y_test_keras))

#essai plot keras
#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')
#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot

#SVG(model_to_dot(model).create(prog='dot', format='svg'))

# graphique classe correct et fonction objective
#plt.plot(logs.history['acc'], label='train')
#plt.plot(logs.history['val_acc'], label='valid')
#plt.legend()
#plt.title('Pourcentage de classes correctes')
#plt.show()
#plt.plot(logs.history['loss'], label='train')
#plt.plot(logs.history['val_loss'], label='valid')
#plt.title('Fonction objectif sur l\'ensemble d\'entrainement')
#plt.show()
