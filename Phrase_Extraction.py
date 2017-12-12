from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from architecture.lstm_model import Model
import tensorflow as tf
import gensim
import numpy as np
from utils.util import *
from config.config import Config



class Phrase_Extraction:

   
    def __init__(self):
        stemmer=PorterStemmer()
        stop_words = get_stop_words()
        conf = Config("config/system.config")
        lstm_model_path = conf.getConfig("PATHS", "tf_model")
        word2_vec_model = conf.getConfig("PATHS", "word_vec_model")
        self.max_seq = conf.getConfig("MODEL_PARAMS", "sentence_length")
        self.vec_dim = conf.getConfig("MODEL_PARAMS", "word_vec_dim")
        self.input_dim = conf.getConfig("MODEL_PARAMS", "feature_vector_dim")
        num_layers = conf.getConfig("MODEL_PARAMS", "num_layers")
        class_size = conf.getConfig("MODEL_PARAMS", "class_size")
        rnn_size = conf.getConfig("MODEL_PARAMS", "rnn_size")
        word_model = gensim.models.Word2Vec.load(word2_vec_model)
        self.model = Model(self.max_seq, self.input_dim, num_layers, class_size, rnn_size)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, lstm_model_path)


    def get_feature_matrix(self,sentence):
        pos_tags = pos_tag(sentence)
        matrix = []
        for pos_tuple in pos_tags:
            try:
                temp = word_model[stemmer.stem(pos_tuple[0].lower().strip())]
            except:
                temp = np.random.uniform(low=.25, high=.50, size=(self.vec_dim,)) #Random vector for new word, Can also use dummy vector
            temp = np.append(temp, pos(pos_tuple[1]))
            temp = np.append(temp, capital(pos_tuple[0]))
            matrix.append(temp)
        for _ in range(self.max_seq - len(sentence)):
            temp = np.array([0 for _ in range(self.input_dim)])
            matrix.append(temp)
        print len(matrix)

        return matrix


    def get_input_x(self,msg):
        sentences = sent_tokenize(clean_str(msg))
        tensor = []
        for sen in sentences:
            tokens = sentence_to_token_list(sen)
            tensor.append(self.get_feature_matrix(tokens))
        return [tensor,sentences]


    def get_extracted_phrase(self,sequence,lables_seq):
        extracted_phrase = []
        for index, seq in enumerate(sequence):
            tokens = sentence_to_token_list(seq)
            phrase = ""
            for count, tag in enumerate(lables_seq[index]):
                if count >= len(tokens):
                    break
                if tag == 0:
                    phrase = phrase + " " + tokens[count]
                elif tag == 1:
                    phrase = phrase + " " + tokens[count]
                else:
                    if phrase != "":
                        extracted_phrase.append(phrase.strip())
                    phrase = ""
        return extracted_phrase


    def get_phrase(self,message):
        sen_and_inputx = tgr.get_input_x(msg)
        input_x = sen_and_inputx[0]
        sentences = sen_and_inputx[1]
        pred = self.sess.run(self.model.prediction, {self.model.input_data: input_x})
        prediction = np.argmax(pred, 2)
        return self.get_extracted_phrase(sentences, prediction)























