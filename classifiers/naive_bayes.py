from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer, CountVectorizer
from sklearn import naive_bayes,metrics, linear_model, svm, grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import random
from Tools.data_processing import get_clean_data

data = get_clean_data()
categories = data.sentiment.unique()
categories  = categories.tolist() #neutre, positif, négatif

#set target y and predictor x
x = data.processed_text.values
y_labelname = data.sentiment.values
y_categories = pd.Categorical.from_array(y_labelname)
y= y_categories.codes #transform sentiment labels into 0,1,2

# Déterminer au hasard des indices pour les exemples d'entrainement et de test
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
count_vect = CountVectorizer() #counts occurences of each word
X_train_counts = count_vect.fit_transform(x_train)
tfidf_transformer = TfidfVectorizer() #applies transformation to account for frequency
X_train_tfidf = tfidf_transformer.fit_transform(x_train)
#including 2-grams
bigram_vect = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
X_train_bigramcounts = bigram_vect.fit_transform(x_train)

#First attempt at classification
clf = MultinomialNB().fit(X_train_counts, y_train)
clf_bigram = MultinomialNB().fit(X_train_bigramcounts, y_train)
clf_tfidf = MultinomialNB().fit(X_train_tfidf, y_train)
#check with test set
X_test_counts = count_vect.transform(x_test)
predicted = clf.predict(X_test_counts)
print("Means for counts: " + str(np.mean(predicted == y_test)))

X_test_bigramcounts = bigram_vect.transform(x_test)
predicted2 = clf_bigram.predict(X_test_bigramcounts)
print("Means for bigrams: " + str(np.mean(predicted2 == y_test)))

X_test_tfidf = count_vect.transform(x_test)
predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
print("Means for tfidf: " + str(np.mean(predicted_tfidf == y_test)))

classifiers = [clf, clf_bigram, clf_tfidf]

predicted_train_counts = clf.predict(X_train_counts)
predicted_train_bigram = clf_bigram.predict(X_train_bigramcounts)
predicted_train_tfidf = clf_tfidf.predict(X_train_tfidf)

print("Means: " + str(np.mean(predicted_train_counts == y_train)))
print("Means: " + str(np.mean(predicted_train_bigram == y_train)))
print("Means: " + str(np.mean(predicted_train_tfidf == y_train)))

