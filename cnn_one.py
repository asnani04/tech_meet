from __future__ import print_function
import numpy as np
import tensorflow as tf
# import numpy as np
import os
import math
# import pvec_we
import get_data_mod as get_data
from time import time
import pickle as pkl
import gensim

# ===============================================================
# Data Loading and Preparation

print("Loading data...")
x_train, ind_train, change_train, sentences_padded_train, vocabulary, vocabulary_inv = get_data.load_data(50000)
# Randomly shuffle data
max_len = len(x_train[1])
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(x_train)))
x_train = x_train[shuffle_indices]
ind_train = ind_train[shuffle_indices]
change_train = change_train[shuffle_indices]
# y_train = y_train[shuffle_indices]

# shuffle_indices = np.random.permutation(np.arange(len(y_test)))
# x_test = x_test[shuffle_indices]
# y_test = y_test[shuffle_indices]

print("Vocabulary Size: {:d}".format(len(vocabulary)))

# ===============================================================
# Function to compute Error rate / Accuracy

def error_rate(predictions, labels):
    cor = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i], 0) == labels[i][0]:
            cor = cor + 1
    return 100.0 - (100.0 * cor / predictions.shape[0])

# Hyper Parameters

batch_size = 1
embedding_size = 300
vocabulary_size = len(vocabulary)
context_size = 2
num_classes = 1
num_labels = 30
train_size = len(x_train)
# test_size = len(x_test)
num_filters = 128
no_of_epochs = 15
h1 = 75
freq = 5

# Model
train_data = tf.placeholder(tf.int32, shape=[batch_size, max_len])
train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
test_data = tf.placeholder(tf.int32, shape=[batch_size, max_len])
test_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])

word_embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size * (2*context_size)],
                            stddev=1.0 / math.sqrt(embedding_size * 4)))
softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
conv1_weights = tf.Variable(
    tf.truncated_normal([5, embedding_size, 1, num_filters],  #5x300 filter, depth 32.
                          stddev=0.01))
conv1_biases = tf.Variable(tf.zeros([num_filters]))
fc1_weights = tf.Variable(  #fully connected, depth 512.
    tf.truncated_normal(
        [num_filters, h1],
        stddev=0.01))
fc1_biases = tf.Variable(tf.constant(0.01, shape=[h1]))
fc2_weights = []
fc2_biases = []

for i in range(num_labels):
    fc2_weights.append(tf.Variable(tf.truncated_normal([h1, 1])))
    fc2_biases.append(tf.Variable(tf.zeros([1])))


embed = tf.nn.embedding_lookup(word_embeddings, train_data) # Shape: (batch_size, max_len, embedding_size)
embed_shape = embed.get_shape().as_list()
embed = tf.reshape(embed, [embed_shape[0], embed_shape[1], embed_shape[2], 1])

embed_test = tf.nn.embedding_lookup(word_embeddings, test_data) # Shape: (batch_size, max_len, embedding_size)
embed_test_shape = embed_test.get_shape().as_list()
embed_test = tf.reshape(embed_test, [embed_test_shape[0], embed_test_shape[1], embed_test_shape[2], 1])

# Graph Computations - Training

conv1 = tf.nn.conv2d(embed,
         conv1_weights,
         strides = [1, 1, 1, 1],
        padding = "VALID",
         name="conv")
relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
pooled = tf.nn.max_pool(relu,
            ksize=[1, max_len - 5 + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
pool_shape = pooled.get_shape().as_list()
pooled = tf.reshape(pooled, [pool_shape[0], pool_shape[3]])

fc1 = tf.nn.relu(tf.matmul(pooled, fc1_weights) + fc1_biases)
fc2, error, optimizer = [], [], []

for i in range(num_labels):
    fc2.append(tf.matmul(fc1, fc2_weights[i]) + fc2_biases[i])
    error.append(tf.reduce_sum(tf.square(fc2[i] - train_labels)))
    optimizer.append(tf.train.AdamOptimizer(0.0001).minimize(error[i]))

    

# Graph Computations - Testing


saver = tf.train.Saver({'word_embeddings': word_embeddings, 'softmax_weights': softmax_weights, 'softmax_biases': softmax_biases})
saver_weights = tf.train.Saver({'conv1_weights': conv1_weights, 'conv1_biases': conv1_biases, 'fc1_weights': fc1_weights, 'fc1_biases': fc1_biases})

# Function that evaluates test data in batches

def eval_in_batches(data, sess):
    global batch_size
    size = data.shape[0]
    if size < batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, 2), dtype=np.float32)
    for begin in xrange(0, size, batch_size):
        end = begin + batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={test_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={test_data: data[-batch_size:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

# Test the learned model

def test_learned(sess):
    test_error = error_rate(eval_in_batches(x_test, sess), y_test)
    print('Test error: %.1f%%' % test_error)
    return

# Use a tensorflow session to carry out required training and evoke testing routines
y_train = change_train
stock = ind_train

start = time()

features = []
labels = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "./pvec_class_300_2.ckpt")
    print("Initialized")
    if os.path.exists("./cnn_doc_10000_2.ckpt"): # if Weights have been used
        saver_weights.restore(sess, "./cnn_doc_10000_3.ckpt")
        test_learned(sess)
    else:
        cum_loss = 0.0
        num_steps = int(no_of_epochs * train_size) // batch_size
        for step in range(int(no_of_epochs * train_size) // batch_size):
            offset = (step * batch_size) % (train_size - batch_size)
            batch_data = x_train[offset:(offset + batch_size), ...]
            batch_labels = [y_train[offset:(offset + batch_size)]]
            feed_dict = {train_data: batch_data,
                         train_labels: batch_labels}
            loss_train = sess.run([error[stock[step % train_size]], optimizer[stock[step % train_size]]], feed_dict=feed_dict)

            if step >= num_steps - train_size:
                featlist = sess.run([pooled, fc2[stock[step % train_size]]], feed_dict=feed_dict)
                features.append(featlist[0])
                
                labels.append(featlist[1])
                
            
            cum_loss += loss_train[0]
            if step % train_size == 0:
                end = time()
                delta_time = end-start 
                print(cum_loss, delta_time)
                start = time()
                cum_loss = 0.0


pkl.dump({'f':features}, open("feat.p", "wb"))
pkl.dump({'l':labels}, open("lab.p", "wb"))

