import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from config.config import Config
import random
from nltk import pos_tag
from utils.util import *
import csv


stemmer=PorterStemmer()
conf = Config("config/system.config")
training_data = conf.getConfig("PATHS", "training_data_path")
test_data = conf.getConfig("PATHS", "test_data")
train_csv_path = conf.getConfig("PATHS", "train_csv_path")
test_csv_path = conf.getConfig("PATHS", "test_csv_path")
max_seq_length = conf.getConfig("MODEL_PARAMS", "sentence_length")


def message_to_sentences(message,remove_stopwords=False):
    raw_sentences = sent_tokenize(clean_str(message))
    sentences = []
    for raw_sentence in raw_sentences:
        if "<" in raw_sentence and ">" in raw_sentence:
            tokens = sentence_to_token_list(raw_sentence)
            if len(tokens) <= max_seq_length:
                sentences.append(tokens)
        """
        elif "<" not in raw_sentence and ">" not in raw_sentence:
            tokens = sentence_to_token_list(raw_sentence)
            if len(tokens) <= 35:
                sentences.append(tokens)
        """
    return sentences
    
def get_pos(sent):
    posTag = pos_tag(sent)
    return posTag

def format_label(tagged_sen):
    tokens = []
    labels = []
    label = "O"
    for token in tagged_sen:
        if token == "<":
            label = "B-P"
        elif token == ">":
            label = "O"
        else:			
            tokens.append(token)
            labels.append(label)
            if label == "B-P":
                label = "I-P"
    pos_tag = get_pos(tokens)
    return [pos_tag, labels]

def write_to_csv(sen_list, file_name):
    with open(file_name, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for sen in sen_list:
            processed_sen = format_label(sen)
            for count, elem in enumerate(processed_sen[0]):
                writer.writerow([elem[0], elem[1], processed_sen[1][count]])
            writer.writerow([])	

def prepare_train_test_data(training_file_name, path):
    corpus = open(training_file_name, 'r').read()
    messages = corpus.split('\n')
    sen_list = []
    for msg in messages:
        sentences = message_to_sentences(msg)
        if len(sentences) > 0:
            sen_list.extend(sentences)
    print "Total number of sentences are - " + str(len(sen_list))		
    random.shuffle(sen_list)
    write_to_csv(sen_list, path)



prepare_train_test_data(training_data,train_csv_path)
prepare_train_test_data(test_data,test_csv_path)






