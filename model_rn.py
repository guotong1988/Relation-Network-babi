from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
try:
    import tfplot
except:
    pass

from ops import conv2d, fc

from vqa_util import question2str, answer2str


class Model(object):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 batch_size,
                 sentence_size,
                 sentence_number,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.max_sentence_size = sentence_size
        self.sentence_number = sentence_number
        # create placeholders for the input
        self.input_context = tf.placeholder(
            name='context', dtype=tf.int32,
            shape=[self.batch_size, self.sentence_number, self.max_sentence_size],
        )
        self.q = tf.placeholder(
            name='q', dtype=tf.int32, shape=[self.batch_size, self.max_sentence_size],
        )
        self.a = tf.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.vocab_size],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build_text(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build_text(self, is_train=True):

        def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
                g_2 = fc(g_1, 256, name='g_2')
                g_3 = fc(g_2, 256, name='g_3')
                g_4 = fc(g_3, 256, name='g_4')
                return g_4

        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope) as scope:
                fc_1 = fc(g, 256, name='fc_1')
                fc_2 = fc(fc_1, 256, name='fc_2')
                fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, self.vocab_size, activation_fn=None, name='fc_3')
                return fc_3

        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy

        with tf.device('/cpu:0'), tf.name_scope("embedding_q"):
            W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            embedded_char = tf.nn.embedding_lookup(W, self.q)

        with tf.variable_scope("rnn_q"):
            cell = tf.nn.rnn_cell.LSTMCell(
                10,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell,
                embedded_char,
                dtype=tf.float32)
            self.encoding_question_input = rnn_states.h



        inputs = tf.unstack(self.input_context,axis=1)
        self.embedded_chars = []
        self.encoding_context_inputs = []
        for i in range(self.sentence_number):
            with tf.device('/cpu:0'), tf.name_scope("embedding"+str(i)):
                W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W")
                embedded_char = tf.nn.embedding_lookup(W, inputs[i])
                self.embedded_chars.append(embedded_char)

            with tf.variable_scope("rnn"+str(i)):
                cell = tf.nn.rnn_cell.LSTMCell(
                    32,
                    forget_bias=2.0,
                    use_peepholes=True,
                    state_is_tuple=True)

                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    cell,
                    self.embedded_chars[i],
                    dtype=tf.float32)
                encoding_context = rnn_states.h
                self.encoding_context_inputs.append(encoding_context)

        all_g = []
        for i in range(self.sentence_number):
            for j in range(self.sentence_number):
                if i == 0 and j == 0:
                    g_i_j = g_theta(self.encoding_context_inputs[i], self.encoding_context_inputs[j], self.encoding_question_input, reuse=False)
                else:
                    g_i_j = g_theta(self.encoding_context_inputs[i], self.encoding_context_inputs[j], self.encoding_question_input, reuse=True)
                all_g.append(g_i_j)

        all_g = tf.stack(all_g, axis=0)
        all_g = tf.reduce_mean(all_g, axis=0, name='all_g')

        logits = f_phi(all_g, scope='f_phi')
        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = build_loss(logits, self.a)

