import matplotlib.pyplot as plt
import pandas as pd
from tweet_analysis.Tools.data_processing import get_clean_data
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
import time
from keras.callbacks import EarlyStopping


##%Load and set the data
#Importation des data
#Sentiment  election, Tweets  avion
namedata='Tweets'

#data = get_clean_data(namedata)
data=get_clean_data(namedata)

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
x_exist_emot=data.exist_emot.values
x_sent_emot=data.sent_emot.values

y_categories = pd.Categorical.from_array(y_labelname)

y= y_categories.codes  #transform sentiment labels into 0,1,2


#Determiner au hasard des indices pour les exemples d'entrainement et de test
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

# garder les bonnes lignes, EMOTICONES
x_train_exist_emot=x_exist_emot[train_inds]
x_test_exist_emot=x_exist_emot[test_inds]
x_train_sent_emot=x_sent_emot[train_inds]
x_test_sent_emot=x_sent_emot[test_inds]

#bag of words representation: vectorizing
#1-grams
print '1-grams'
#counts occurences of each word
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train).toarray()
X_test_counts = count_vect.transform(x_test).toarray()
#tfidf_transformer = TfidfVectorizer() #applies transformation to account for frequency
#X_train_tfidf = tfidf_transformer.fit_transform(x_train)
#X_test_tfidf= tfidf_transformer.fit_transform(x_test)

#2-grams
#bigram_vect = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
#X_train_bigramcounts = bigram_vect.fit_transform(x_train)



#%% Keras
#notre exemple de test modifier au besoin de keras, sans considerer emoticone
x_train_keras= X_train_counts
x_test_keras=X_test_counts
y_train_keras = to_categorical(y_train,3)
y_test_keras = to_categorical(y_test,3)


# Arrete le code lorsque le model n'est plus capable d'optimiser plus.
early_stopping = EarlyStopping(monitor='val_loss', patience=2)


#creation du reseau de neurone avec keras
#importation du reseau de neurone nn12
nepoche = 100
optimizerlist=['sgd']#,'rmsprop','adagrad','adadelta','adam','adamax','nadam']  #,'tfoptimizer'] #tester ac leur param par def
'''
for i,opt in enumerate(optimizerlist):
    from Tools.KerasNN import NN1
    model=NN1(np.shape(x_train_keras)[1],opt)
    #Entrainement du reseau de neurones
    logs = model.fit(x_train_keras, y_train_keras, nb_epoch=nepoche,validation_data=(x_test_keras, y_test_keras))

    #enregistrement des resultats

    file = open("Tweets_NN1.txt", "a")
    print >> file, 'NN1'
    print >> file, opt
    print >> file, 'train'
    print >> file, logs.history['acc'][-1]
    print >> file, 'valid'
    print >> file, logs.history['val_acc'][-1]
    print >> file, 'historique'
    print >> file, logs.history
    print >> file, ''
    file.close()
'''

for i,opt in enumerate(optimizerlist):

    from tweet_analysis.Tools.KerasNN import NN2
    model=NN2(np.shape(x_train_keras)[1],opt,'tanh',10)  # nn2,3 : et hidden layer, tanh ou relu.
    #Entrainement du reseau de neurones

    start = time.time()
    logs = model.fit(x_train_keras, y_train_keras, nb_epoch=nepoche,validation_data=(x_test_keras, y_test_keras),callbacks=[early_stopping])

    end = time.time()
    duree=end-start
    print duree

    #enregistrement des resultats

    file = open("Tweets_without_emot.txt", "a")
    print >> file, 'NN2'
    print >> file, opt
    print >> file, 'tanh'
    print >> file, '10'
    print >> file, 'train'
    print >> file, logs.history['acc'][-1]
    print >> file, 'valid'
    print >> file, logs.history['val_acc'][-1]
    print >> file, 'historique'
    print >> file, logs.history
    print >> file, duree
    print >> file, ''
    file.close()


#modification du xtrain, ajout de 2 colonne, EMOTICONES.
train_emot=np.transpose(np.vstack((x_train_exist_emot,x_train_sent_emot)))
test_emot=np.transpose(np.vstack((x_test_exist_emot,x_test_sent_emot)))

x_train_keras=np.hstack((X_train_counts,train_emot))
x_test_keras=np.hstack((X_test_counts,test_emot))

for i,opt in enumerate(optimizerlist):

    from tweet_analysis.Tools.KerasNN import NN2
    model=NN2(np.shape(x_train_keras)[1],opt,'tanh',10)  # nn2,3 : et hidden layer, tanh ou relu.
    #Entrainement du reseau de neurones

    start = time.time()
    logs = model.fit(x_train_keras, y_train_keras, nb_epoch=nepoche,validation_data=(x_test_keras, y_test_keras),callbacks=[early_stopping])

    end = time.time()
    duree=end-start
    print duree

    #enregistrement des resultats

    file = open("Tweets_with_emot.txt", "a")
    print >> file, 'NN2'
    print >> file, opt
    print >> file, 'tanh'
    print >> file, '10'
    print >> file, 'train'
    print >> file, logs.history['acc'][-1]
    print >> file, 'valid'
    print >> file, logs.history['val_acc'][-1]
    print >> file, 'historique'
    print >> file, logs.history
    print >> file, duree
    print >> file, ''
    file.close()

'''
# graphique classe correct et fonction objective
plt.plot(logs.history['acc'], label='train')
plt.plot(logs.history['val_acc'], label='valid')
plt.legend()
plt.title('Pourcentage de classes correctes')
plt.show()
plt.plot(logs.history['loss'], label='train')
plt.plot(logs.history['val_loss'], label='valid')
plt.title('Fonction objectif sur l\'ensemble d\'entrainement')
plt.show()
'''
