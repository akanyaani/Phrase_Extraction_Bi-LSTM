import numpy as np
import pickle as pkl
import gensim
import sys
from nltk.stem import PorterStemmer
from config.config import Config
from utils.util import *


stemmer=PorterStemmer()
conf = Config("config/system.config")

def find_max_sen_length(file_name):
    temp_len = 0
    max_length = 0
    for line in open(file_name):
        if line in ['\n', '\r\n']:
            if temp_len > max_length:
                max_length = temp_len
            temp_len = 0
        else:
            temp_len += 1
    return max_length


def get_input(model, word_dim, input_file, output_embed, output_tag, sentence_length=-1):
    count = 0
    word = []
    tag = []
    sentence = []
    sentence_tag = []
    if sentence_length == -1:
        max_sentence_length = find_max_sen_length(input_file)
    else:
        max_sentence_length = sentence_length
    sentence_length = 0
    print "max sentence length is %d" % max_sentence_length
    for line in open(input_file):
        #print line
        if line in ['\n', '\r\n']:
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.array([0] * 3))
                temp = np.array([0 for _ in range(word_dim + 6)])
                print len(temp)
                word.append(temp)
            sentence.append(word)
            sentence_tag.append(np.array(tag))
            #print word
            #print tag
            sentence_length = 0
            word = []
            tag = []
        else:
            assert (len(line.split()) == 3)
            sentence_length += 1
            try:
                temp = model[stemmer.stem(line.split()[0].strip().lower())]
            except:
                count = count + 1
                #print (line.split()[0])
                temp = np.random.uniform(low=.25, high=.50, size=(word_dim,))
            assert len(temp) == word_dim
            temp = np.append(temp, pos(line.split()[1]))  # adding pos one hot encode
            temp = np.append(temp, capital(line.split()[0]))  # adding one hot encode
            print len(temp)
            word.append(temp)
            t = line.split()[2]
            #print line.split()
            if t.endswith('B-L'):
                tag.append(np.array([1, 0, 0]))
            elif t.endswith('I-L'):
                tag.append(np.array([0, 1, 0]))
            elif t.endswith('O'):
                tag.append(np.array([0, 0, 1]))
    assert (len(sentence) == len(sentence_tag))

    pkl.dump(sentence, open(output_embed, 'wb'))
    pkl.dump(sentence_tag, open(output_tag, 'wb'))
    print ("Done................................")


word_vec_dim = conf.getConfig("MODEL_PARAMS", "word_vec_dim")
sen_len = conf.getConfig("MODEL_PARAMS", "sentence_length")
word_vec_model = conf.getConfig("PATHS", "word_vec_model")
train_file= conf.getConfig("PATHS", "train_csv_path")
test_file_one = conf.getConfig("PATHS", "test_csv_path")
train_embedding = conf.getConfig("PATHS", "train_embedding")
train_tag = conf.getConfig("PATHS", "train_tag")
test_embedding = conf.getConfig("PATHS", "test_embedding")
test_tag = conf.getConfig("PATHS", "test_tag")

word2vec_model = gensim.models.Word2Vec.load(word_vec_model)

get_input(word2vec_model, word_vec_dim, train_file, train_embedding, train_tag, sentence_length=sen_len)
get_input(word2vec_model, word_vec_dim, test_file_one, test_embedding, test_tag, sentence_length=sen_len)








