
from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from model_rn import Model
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
parser.add_argument('--prefix', type=str, default='default')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--learning_rate', type=float, default=2.5e-4)
parser.add_argument('--lr_weight_decay', action='store_true', default=False)
config = parser.parse_args()

tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 99999, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "/home/gt/Relation-Network-Tensorflow/tasks_1-20_v1-2/en-10k/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)

# Add time words/indexes
for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position
sentence_size += 1  # +1 for time words

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)


# params
n_train = trainS.shape[0]
n_val = valS.shape[0]
n_test = testS.shape[0]

tf.set_random_seed(FLAGS.random_state)
batch_size = 64

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

batches_val = zip(range(0, n_val-batch_size, batch_size), range(batch_size, n_val, batch_size))
batches_val = [(start, end) for start, end in batches_val]

batches_test = zip(range(0, n_test-batch_size, batch_size), range(batch_size, n_test, batch_size))
batches_test = [(start, end) for start, end in batches_test]

with tf.Session() as sess:
  model = Model(vocab_size,16,batch_size,sentence_size,memory_size)
  global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
  optimizer = tf.contrib.layers.optimize_loss(
      loss=model.loss,
      global_step=global_step,
      learning_rate=config.learning_rate,
      optimizer=tf.train.AdamOptimizer,
      clip_gradients=20.0,
      name='optimizer_loss'
  )
  sess.run(tf.global_variables_initializer())
#  saver = tf.train.Saver(max_to_keep=10)

#  saver.restore(sess, os.path.join("./", 'model'))

  for t in range(1, FLAGS.epochs + 1):

      np.random.shuffle(batches)
      total_cost = 0.0
      for start, end in batches:
          s = trainS[start:end]
          q = trainQ[start:end]
          a = trainA[start:end]
          cost_t,accuracy_print = sess.run([optimizer, model.accuracy],
                             feed_dict={model.input_context:s,
                                        model.q:q,
                                        model.a:a
                                        })
          total_cost += cost_t
      print(total_cost,accuracy_print)
      if t % 100 == 1:
          for start, end in batches_val:
              s = valS[start:end]
              q = valQ[start:end]
              a = valA[start:end]
              accuracy_print_test = sess.run([model.accuracy],
                                            feed_dict={model.input_context: s,
                                                       model.q: q,
                                                       model.a: a,
                                                       model.is_training:False
                                                       })
              print(accuracy_print_test)
          for start, end in batches_test:
              s = testS[start:end]
              q = testQ[start:end]
              a = testA[start:end]
              accuracy_print_test = sess.run([model.accuracy],
                                            feed_dict={model.input_context: s,
                                                       model.q: q,
                                                       model.a: a,
                                                       model.is_training: False
                                                       })
              print(accuracy_print_test)
#      if t % 1000 == 0:
#          saver.save(sess,os.path.join("./", 'model'),global_step=global_step)

