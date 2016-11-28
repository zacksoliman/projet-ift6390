import pandas as pd
import numpy as np
import string, re
import nltk
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer, CountVectorizer
from sklearn import naive_bayes,metrics, linear_model,svm, grid_search
import time,random
import operator
#from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.snowball import SnowballStemmer

stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation +["rt", 'url']
stemmer = SnowballStemmer("english")
HillaryWords = ['hillary clinton','hillaryclinton','hilaryclinton','hillari clinton','hilari clinton','hilary clinton','hillary','clinton']
DonaldWords = ['donald trump','donaldtrump','donald','trump','realdonaldtrump']
CarsonWords = ['realbencarson','bencarson','carson']
BushWords = ['jebbush','bush']
hillary_re = re.compile('|'.join(map(re.escape, HillaryWords)))
donald_re = re.compile('|'.join(map(re.escape, DonaldWords)))

data = pd.read_csv("../data/Tweets.csv")
classifier =[]
def preprocess(tweet):
    if type(tweet)!=type(2.0):
        tweet = tweet.lower()
        tweet = " ".join(tweet.split('#'))
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = " ".join(tweet.split('@'))
        tweet = re.sub(r'@([^\s]+)', r'\1', tweet)
        tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))','URL',tweet)
        tweet = re.sub("http\S+", "URL", tweet)
        tweet = re.sub("https\S+", "URL", tweet)
        tweet = " ".join(tweet.split(':'))
        #removes @username from text entirely
        #tweet = re.sub('@[^\s]+','AT_USER',tweet)
        #tweet = tweet.replace("AT_USER","")
        tweet = tweet.replace("URL","")
        tweet = tweet.replace(".","")
        tweet = tweet.replace('\"',"")
        tweet = tweet.replace('&amp',"")
        #remove punctuation words
        tweet = " ".join([word for word in tweet.split(" ") if word not in stop_list])
        #remove words ending with special character
        tweet = " ".join([word for word in tweet.split(" ") if re.search('^[a-z]+$', word)])
        #remove common words such as "the"
        tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split(" ")])
        #stem similar words such as "hurt" and "hurting"
        tweet = " ".join([stemmer.stem(word) for word in tweet.split(" ")])
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = tweet.strip('\'"')
        #manually stem similar political words
        tweet = hillary_re.sub("hillary", tweet)
        tweet = donald_re.sub("donald", tweet)
    else:
        tweet=''
    return tweet

#text processing
data['processed_text'] = data.text.apply(preprocess)

#remove duplicate tweets
data = data.drop_duplicates(subset='processed_text', keep='last')

categories = data.airline_sentiment.unique()
categories  = categories.tolist() #neutre, positif, négatif

#set target y and predictor x
x = data.processed_text.values
y_labelname = data.airline_sentiment.values
y_categories = pd.Categorical.from_array(y_labelname)
y= y_categories.codes #transform sentiment labels into 0,1,2

#Some commands
#data.head() to display first 5 lines of each column
#test2 = [i for i, w in enumerate(data.processed_text) if re.search('(hillari)', w)]
#test = [w for w in data.processed_text if re.search('(clinton|hillari)', w)]

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
