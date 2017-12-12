from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import pickle
import re


stemmer=PorterStemmer()


def load_train_data():
    emb = pickle.load(open('', 'rb'))
    tag = pickle.load(open('', 'rb'))
    print 'training data loaded.........'
    return emb, tag

def load_test_data():
    emb = pickle.load(open('', 'rb'))
    tag = pickle.load(open('', 'rb'))
    print 'test data loaded.............'
    return emb, tag

def load_validation_data():
    emb = pickle.load(open('', 'rb'))
    tag = pickle.load(open('', 'rb'))
    print 'validation data loaded........'
    return emb, tag


def remove_non_ascii(text):
    return re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\xff]', ' ', text)

def clean_str(msg):
    #msg = re.sub(r"\'s", " \'s", msg, flags=re.IGNORECASE)
    #msg = re.sub(r"\'ve", " \'ve", msg, flags=re.IGNORECASE)
    #msg = re.sub(r"\'t", " \'t", msg, flags=re.IGNORECASE)
    #msg = re.sub(r"\'re", " \'re", msg, flags=re.IGNORECASE)
    #msg = re.sub(r"\'d", " \'d", msg, flags=re.IGNORECASE)
    #msg = re.sub(r"\'ll", " \'ll", msg, flags=re.IGNORECASE)
    msg = re.sub(r'\(', ' ', msg)
    msg = re.sub(r'\)', ' ', msg)
    msg = re.sub(r'\[', ' ', msg)
    msg = re.sub(r'\]', ' ', msg)
    msg = re.sub(r'\{', ' ', msg)
    msg = re.sub(r'\}', ' ', msg)
    msg = re.sub(r'\#', ' ', msg)
    msg = re.sub(r'\$', ' ', msg)
    msg = re.sub(r'\*', ' ', msg)
    msg = re.sub(r'\"', ' ', msg)
    msg = re.sub(r'\/', ' ', msg)
    msg = re.sub(r'-', ' ', msg)
    msg = re.sub(r'\+', ' ', msg)
    msg = re.sub(r',', ' ', msg)
    msg = re.sub(r'\:', ' ', msg)
    msg = re.sub(r'\;', ' ', msg)
    msg = re.sub(r'&nbsp', ' ', msg)
    msg = re.sub(r'\&', ' ', msg)
    msg = re.sub(r'\s+', ' ', msg)
    return remove_non_ascii(msg.strip())

def get_stop_words():
    stop = "i\'m you\'re i\'ll ourselves hers yourself that\'s there once during out they own an some for its yours such into of itself other is am or who as from him each the themselves until are we these your his don nor me were her himself this our their while to ours she all when at any before them same and in will on does yourselves then that because what over why so can now under he you herself has just where only myself which those i after few whom being if theirs my a by doing it how further was here"
    return stop.lower().split()

stop_words = get_stop_words()
def sentence_to_token_list(sentence, remove_stopwords=True,stemming_flag=False):
    words = word_tokenize(sentence)
    if remove_stopwords: 
        words_n = []
        for word in words:
            if word.lower() not in stop_words:
                words_n.append(word)
        words = words_n
    if stemming_flag:
        stemmed_words = []
        for word in words:
            stemmed_words.append(stemmer.stem(word))
        return  stemmed_words
    return words

def pos(tag):
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])












