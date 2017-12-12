import numpy as np
import pickle as pkl
from nltk.stem import PorterStemmer
import gensim
import sys


stemmer=PorterStemmer()

def find_max_length(file_name):
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


def get_input(model, word_dim, input_file, output_embed, output_tag, sentence_length=-1):
    count = 0
    print('processing %s' % input_file)
    word = []
    tag = []
    sentence = []
    sentence_tag = []
    if sentence_length == -1:
        max_sentence_length = find_max_length(input_file)
    else:
        max_sentence_length = sentence_length
    sentence_length = 0
    print("max sentence length is %d" % max_sentence_length)
    for line in open(input_file):
        #print line
        if line in ['\n', '\r\n']:
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.array([0] * 3))
                temp = np.array([0 for _ in range(word_dim + 6)])
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
                print (line.split()[0])
                temp = np.random.uniform(low=.25, high=.50, size=(word_dim,))
            assert len(temp) == word_dim
            temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
            #temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
            temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
            #print (temp)
            word.append(temp)
            t = line.split()[2]
            #print line.split()
            if t.endswith('B-L'):
                tag.append(np.array([1, 0, 0]))
            elif t.endswith('I-L'):
                tag.append(np.array([0, 1, 0]))
            elif t.endswith('O'):
                tag.append(np.array([0, 0, 1]))
            else:
                print("error in input tag {%s}" % t)
                sys.exit(0)
    assert (len(sentence) == len(sentence_tag))
    print (count)
    pkl.dump(sentence, open(output_embed, 'wb'))
    pkl.dump(sentence_tag, open(output_tag, 'wb'))
    print ("Done................................")


model_dim = 100
sen_len = 35
word_vec_model = ""
train_file= ""
test_file_one = ""
train_embd = ""
train_tag = ""
test_embd = ""
test_tag = ""

word2vec_model = gensim.models.Word2Vec.load(word_vec_model)

get_input(word2vec_model, model_dim, train_file, train_embd, train_tag, sentence_length=sen_len)
get_input(word2vec_model, model_dim, test_file_one, test_embd, test_tag, sentence_length=sen_len)
#get_input(trained_model, args.model_dim, args.test_b, 'test_b_embed.pkl', 'test_b_tag.pkl',sentence_length=sen_len)







