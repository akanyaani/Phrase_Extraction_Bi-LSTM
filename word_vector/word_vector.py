from gensim.models.word2vec import Word2Vec
import pickle as pkl
import re
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import os
import time
import logging
from utils.util import *

class Word_Vector:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    stemmer=PorterStemmer()
    stop_words = get_stop_words()

    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.wvec_model[word]
        except KeyError:
            return self.rand_model[word]


    def message_to_tokenized_sentences(self,message,remove_stopwords=False):
    	try:
            raw_sentences = sent_tokenize(clean_str(message))
            sentences = []
            #print raw_sentences
            for raw_sentence in raw_sentences:
                if len(raw_sentence) > 0:
                    # print raw_sentence
                    sentences.append(self.sentence_to_token_list(raw_sentence))
            return sentences
            
        except Exception,e:
            print "Exception in sentence tokenizer............................"
            print e


    def train(self):
        print('processing corpus..............................')
        processed_sentences = []
        corpus = open(corpusPath, 'r').read()
        messages = corpus.split('\n')
        s = set()
        for message in messages:
            message = self.clean_msg(message)
            #print message
            processed_sentences += self.message_to_tokenized_sentences(message)
        #print processed_sentences[0]
        wvec_model = Word2Vec(sentences=processed_sentences, size=100, window=5, workers=16, sg=1, min_count=1,iter = 20)
        return wvec_model



    def train(self, processed_sentences, dimension, window, threads, mn_count, itr):
        print('Training Word2Vec Model..............................')
        wvec_model = Word2Vec(sentences=processed_sentences, size=dimension, window=window, workers=threads, sg=1, min_count=mn_count,iter = itr)
        return wvec_model

class MySentences(object):
    def __init__(self, dirname, wv):
        self.dirname = dirname
        self.wv = wv
    
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if "crc" in fname:
                continue
            print fname
            for line in open(os.path.join(self.dirname, fname)):
                #print "---------------------------------------"
                #print line
                sent_list = self.wv.message_to_tokenized_sentences(line)
                if sent_list == None:
                    continue
                for sent in sent_list:
                    yield sent
                



if __name__ == '__main__':

    wv = Word_Vector()
    sentences = MySentences('/home/centos/sentiment_lstm/Bi-Lstm/messages', wv) 

    model = wv.train(sentences, 100, 5, 16, 1, 20)
    model.save('/home/centos/sentiment_lstm/Bi-Lstm/model/wordvector_model_100_sep_10')
    print "Done..................."



















