import tensorflow as tf
import numpy as np



class Model:

    def __init__(self, sentence_length, input_dim, num_layers, class_size, rnn_size, learning_rate):

        self.input = tf.placeholder(tf.float32, [None, sentence_length, input_dim])
        self.output = tf.placeholder(tf.float32, [None, sentence_length, class_size])

        #Forward LSTM cell
        forward_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        #Dropout Wrapper
        forward_cell = tf.contrib.rnn.DropoutWrapper(forward_cell, output_keep_prob=0.4)
        #Backward LSTM Cell
        backward_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        #Dropout Wrapper
        backward_cell = tf.contrib.rnn.DropoutWrapper(backward_cell, output_keep_prob=0.4)
        #Multi RNN Cell Wrapper
        forward_cell = tf.contrib.rnn.MultiRNNCell([forward_cell] * num_layers, state_is_tuple=True)
        #Multi RNN Cell Wrapper
        backward_cell = tf.contrib.rnn.MultiRNNCell([backward_cell] * num_layers, state_is_tuple=True)

        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input), reduction_indices=2))

        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        #Bidirectional RNN
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(forward_cell, backward_cell,
                                               tf.unstack(tf.transpose(self.input, perm=[1, 0, 2])),
                                               dtype=tf.float32, sequence_length=self.length)

        #weight, bias = self.get_weight_and_bias(2 * rnn_size, class_size)
        output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2 * rnn_size])
        
        #Weight and Bais for tanH Layer
        weight, bias = self.get_weight_and_bias(2 * rnn_size, rnn_size)

        #prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        ####################################################################
        #Weight Matrix multiplication and Tanh Activation function
        output = tf.tanh(tf.matmul(output, weight) + bias)  ## Tanh Layer

        #Weight and Bais for Softmax Layer
        weight2, bias2 = self.get_weight_and_bias(rnn_size, class_size)
        #Weight Matrix multiplication and Softmax
        prediction = tf.nn.softmax(tf.matmul(output, weight2) + bias2)
        

        #prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)

        #####################################################################
        self.prediction = tf.reshape(prediction, [-1, sentence_length, class_size])
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def get_weight_and_bias(input_size, output_size):
        weight = tf.truncated_normal([input_size, output_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[output_size])
        return tf.Variable(weight), tf.Variable(bias)




