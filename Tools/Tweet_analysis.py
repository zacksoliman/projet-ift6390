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

stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation +["rt", 'url']

data = pd.read_csv("../data/Sentiment.csv")
classifier =[]
def preprocess(tweet):
    if type(tweet)!=type(2.0):
        tweet = tweet.lower()
        tweet = " ".join(tweet.split('#'))
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))','URL',tweet)
        tweet = re.sub("http\S+", "URL", tweet)
        tweet = re.sub("https\S+", "URL", tweet)
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
        tweet = tweet.replace("AT_USER","")
        tweet = tweet.replace("URL","")
        tweet = tweet.replace(".","")
        tweet = tweet.replace('\"',"")
        tweet = tweet.replace('&amp',"")
        tweet  = " ".join([word for word in tweet.split(" ") if word not in stop_list])
        tweet  = " ".join([word for word in tweet.split(" ") if re.search('^[a-z]+$', word)])
        tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split(" ")])
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = tweet.strip('\'"')
    else:
        tweet=''
    return tweet

data['processed_text'] = data.text.apply(preprocess)
categories = data.sentiment.unique()
categories  = categories.tolist()

x = data.processed_text.values
y = data.sentiment.values