import numpy as np
import tensorflow as tf


# =========================
# Fully connected network to predict change in stock prices
# =========================

embed_dim = 300
h1 = 60
num_labels = 30
num_epochs = 3


train_data = tf.placeholder(tf.float32, [None, embed_dim])
train_labels = tf.placeholder(tf.float32, [None, num_labels])
layer1_w = tf.Variable(tf.truncated_normal([embed_dim, h1]))
layer1_b = tf.Variable(tf.zeros([h1]))
layer2_w = tf.Variable(tf.truncated_normal([h1, num_labels]))
layer2_b = tf.Variable(tf.zeros([num_labels]))

layer1_acts = tf.nn.relu(tf.matmul(train_data, layer1_w) + layer1_b)
layer2_acts = tf.matmul(layer1_acts, layer2_w) + layer2_b

error = tf.reduce_sum(tf.square(layer2_acts - train_labels))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(error)

data = np.random.rand(10, 300)
labels = np.random.rand(10, 30)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        loss, opt = sess.run([error, optimizer], feed_dict={train_data: data, train_labels: labels})
        print("epoch = %d, loss = %f" % (epoch, loss))
    
