import tensorflow as tf
import numpy as np
from utils.util import *
from architecture.lstm_model import Model
from config.config import Config



def f1(prediction, target, length):
    tp = np.array([0] * (class_size + 1))
    fp = np.array([0] * (class_size + 1))
    fn = np.array([0] * (class_size + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    print (prediction)

    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
                if target[i, j] == 3:
                    print "------------------------------------"
                    print target[i, j]
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
            #rint tp
    unnamed_entity = class_size - 1
    for i in range(class_size):
        if i != unnamed_entity:
            tp[class_size] += tp[i]
            #print tp
            fp[class_size] += fp[i]
            fn[class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(class_size + 1):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print ("--------Precison---------")    
    print (precision) 
    print ("--------Recall---------")    
    print (recall) 
    print ("--------Fscore---------")       
    print(fscore)
    #return fscore[class_size]


def train(batch_size, sentence_length, input_dim, num_layers, class_size, rnn_size):
    train_inp, train_out = load_train_data()
    test_a_inp, test_a_out = load_test_data()
    test_b_inp, test_b_out = load_validation_data()
    model = Model(sentence_length, input_dim, num_layers, class_size, rnn_size)
    maximum = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        
        for e in range(epoch):
            for ptr in range(0, len(train_inp), batch_size):

                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + batch_size],
                                          model.output_data: train_out[ptr:ptr + batch_size]})
            if e % 10 == 0:
                save_path = saver.save(sess, model_path)
                print("model saved in file: %s" % save_path)
            pred, length = sess.run([model.prediction, model.length], {model.input_data: test_a_inp,model.output_data: test_a_out})
            print("epoch %d:" % e)
            print('test_a score:')
            m = f1(pred, test_a_out, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, model_path)
                print("max model saved in file: %s" % save_path)
                pred, length = sess.run([model.prediction, model.length], {model.input_data: test_b_inp,
                                                                           model.output_data: test_b_out})
                print("test_b score:")
                f1(pred, test_b_out, length)


conf = Config("config/system.config")
model_path = conf.getConfig("PATHS", "tf_model")
input_dim = conf.getConfig("MODEL_PARAMS", "feature_vector_dim")
sentence_length = conf.getConfig("MODEL_PARAMS", "sentence_length")
class_size = conf.getConfig("MODEL_PARAMS", "class_size")
rnn_size = conf.getConfig("MODEL_PARAMS", "rnn_size")
num_layers = conf.getConfig("MODEL_PARAMS", "num_layers")
batch_size = conf.getConfig("MODEL_PARAMS", "batch_size")
epoch = conf.getConfig("MODEL_PARAMS", "epoch")


train(batch_size, sentence_length, input_dim, num_layers, class_size, rnn_size)

print "Training Done..............."








