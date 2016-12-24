import pandas as pd
import numpy as np
import string, re
import nltk
import time,random
import operator
#from tabulate import tabulate
from nltk.stem.snowball import SnowballStemmer

import os.path

stop_list = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
punctuation = list(string.punctuation)
stop_list = stop_list + punctuation +["rt", 'url']
stemmer = SnowballStemmer("english")
HillaryWords = ['hillary clinton','hillaryclinton','hilaryclinton','hillari clinton','hilari clinton','hilary clinton','hillary','clinton']
DonaldWords = ['donald trump','donaldtrump','donald','trump','realdonaldtrump']
CarsonWords = ['realbencarson','bencarson','carson','drbencarson']
BushWords = ['jebbush','bush','jeb']
HuckabeeWords = ['mike huckabee','huckabee','govmikehuckabee','huck']
RubioWords = ['marcorubio','marco rubio','rubio']
KasichWords = ['john kasich','johnkasich','kasich']
CruzWords = ['ted cruz','tedcruz','cruz']
hillary_re = re.compile('|'.join(map(re.escape, HillaryWords)))
donald_re = re.compile('|'.join(map(re.escape, DonaldWords)))
bush_re = re.compile('|'.join(map(re.escape, BushWords)))
carson_re = re.compile('|'.join(map(re.escape, CarsonWords)))
huckabee_re = re.compile('|'.join(map(re.escape, HuckabeeWords)))
rubio_re = re.compile('|'.join(map(re.escape, RubioWords)))
kasich_re = re.compile('|'.join(map(re.escape, KasichWords)))
cruz_re = re.compile('|'.join(map(re.escape, CruzWords)))

classifier =[]


# emoticon, retourne un array text.shape,2
def emoticon(textdata):

    # on part du principe que si il y a plusieurs emoticon ils iront tous dans le mm sens
    # on ajoute 2 valeur a l input, 0 ou 1 qui precise l'existence d'un emot.
    #                               0 ou 1 qui precise si l'emoticon est, respectivement positif ou negatif
    x_emot=np.zeros((textdata.shape[0], 2))
    emoticon=[':)',':-)',':p',':(',':/',';)',':>',':-(',':*',':<',':-*',':-x','<3']   #celui ci me bug ':'('
    emoticon_val=[1,1,1,0,0,1,1,0,1,0,1,1,1] # 1 positif, 0 negatif
    for i,row in enumerate(textdata):
        for j,emot in enumerate(emoticon):
            if row.find(emot) !=-1:
                x_emot[i,0]=1
                x_emot[i, 1] = emoticon_val[j]
            else:
                x_emot[i,0]=0
                x_emot[i, 1] = 0
    return x_emot


def preprocess(tweet, lemmatize):

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
        if lemmatize:
            #remove common words such as "the"
            tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split(" ")])
            #stem similar words such as "hurt" and "hurting"
            tweet = " ".join([stemmer.stem(word) for word in tweet.split(" ")])
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = tweet.strip('\'"')
        #manually stem similar political words
        tweet = hillary_re.sub("hillary", tweet)
        tweet = donald_re.sub("donald", tweet)
        tweet = carson_re.sub("carson", tweet)
        tweet = bush_re.sub("bush", tweet)
        tweet = huckabee_re.sub("huckabee", tweet)
        tweet = cruz_re.sub("cruz", tweet)
        tweet = kasich_re.sub("kasich", tweet)
        tweet = rubio_re.sub("rubio", tweet)
    else:
        tweet=''
    return tweet

def clean_data(data, lemmatize):

    #text processing
    data['processed_text'] = data.text.apply(preprocess, lemmatize=lemmatize)
    #text processing
    emot=emoticon(data.text)
    data['exist_emot'] = emot[:,0]
    data['sent_emot']=emot[:,1]
    # drop duplicates
    data = data.drop_duplicates(subset='processed_text', keep='last')
    return data

def get_clean_data(dataset="Sentiment", lemmatize=True):

    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, "..", "data", dataset + ".csv"))
    data = pd.read_csv(filepath, encoding = 'unicode_escape')

    return clean_data(data, lemmatize)
