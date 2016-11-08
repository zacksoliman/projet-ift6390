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

data = pd.read_csv("../data/Sentiment.csv")
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

categories = data.sentiment.unique()
categories  = categories.tolist()

#set target and predictor
x = data.processed_text.values
y = data.sentiment.values

#Some commands
#data.head() to display first 5 lines of each column
#test2 = [i for i, w in enumerate(data.processed_text) if re.search('(hillari)', w)]
#test = [w for w in data.processed_text if re.search('(clinton|hillari)', w)]
