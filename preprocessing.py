## TODO: PREPROCESS DATA FROM TRAINSET, LEMMATIZE, STOPWORD REMOVAL AND MAKING DICTIONAIRY
import os, sys
import pandas as pd
import numpy as np
import nltk
import math
from textblob import TextBlob as tb
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


## Declaring variables and functions
c_stopWords = stopwords.words("english")
ps = PorterStemmer()

## Functions
def tokenizeWords(input):
    tokenized_words = word_tokenize(input)
    return tokenized_words

def removeStopwords(input):
    text = input
    text = ' '.join([word for word in text.split() if word not in c_stopWords])
    return text



def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

## Check if file exists and load it
assert os.path.exists('./emails.train.csv'), "[Dataset File Not Found] Please download dataset first."
df = pd.read_csv('./emails.train.csv')
df_pos = df[df['spam']==1]

## Random text for testing
random_text = np.random.choice(df_pos['text'])

## Removing stopwords, tokenizing and stemming
test = removeStopwords(random_text)
tokenize = tokenizeWords(test)
print(tokenize)

##for word in tokenize:
    ##ps.stem(word)
    ##print(tokenize)





## Testing documents for TF-IDF
document1 = tb("blabla")
document2 = tb(np.random.choice(df_pos['text']))
document3 = tb(np.random.choice(df_pos['text']))


## Moet variablen veranderen omdat dit gekopieerd is.
bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:5]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))