from gensim.models.word2vec import Word2Vec
import pickle as pkl
import argparse
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from Stemming_Helpers import stem, original_form
from glove import Corpus, Glove
from util import *


class Glove_Word_Vector:
    tweet_tokenizer = TweetTokenizer()
    stemmer=PorterStemmer()
    corpus = Corpus()
    glove = Glove(no_components=100, learning_rate=0.05)
    
    def __init__(self, corpusPath):
        print('processing corpus..............................')
        processed_sentences = []
        corpus = open(args.corpus, 'r').read()
        messges = corpus.split('\n')
        for message in messages:
            processed_sentences.append(message_to_sentences(message))
          
        self.corpus.fit(processed_sentences, window=10)

        self.glove.fit(corpus_model.matrix, epochs=30, no_threads=4, verbose=True)

        self.glove.add_dictionary(corpus_model.dictionary)

        self.glove.save('glove.model')


    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.wvec_model[word]
        except KeyError:
            return self.rand_model[word]

    def sentence_to_token_list(sentence, remove_stopwords=False, stemming_flag):
    	sentence_text = re.sub(r'[^\w\s]','', sentence)
    	words = self.tweet_tokenizer.tokenize(sentence_text)
        if stemming_flag:
            stemmed_words = []
            for word in words:
                stemmed_words.append(self.stemmer.stem(word))
    	return(words)

    def message_to_sentences(message, tokenizer, remove_stopwords=False ):
    	try:
            raw_sentences = sent_tokenize(message.strip())
            sentences = []
            for raw_sentence in raw_sentences:
                if len(raw_sentence) > 0:
                    sentences.append(sentence_to_token_list(raw_sentence))
            return sentences
        except:
            print('nope')

if __name__ == '__main__':
    corpusPath = ""
    Word_Vector(corpusPath)
    #pkl.dump(model, open('wordvector_model_300' + '.pkl', 'wb'))
    print "Done................................"




