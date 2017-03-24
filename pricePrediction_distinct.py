import numpy as np
import tensorflow as tf


# =========================
# Fully connected network to predict change in stock prices
# =========================

embed_dim = 300
h1 = 60
num_labels = 30
num_epochs = 3
batch_size = 10

error = tf.zeros([1], dtype=tf.float32)

train_data = tf.placeholder(tf.float32, [1, embed_dim])
    
train_labels = tf.placeholder(tf.float32, [1, 1])
# the rise or fall in the stock associated
    
init = 0

layer1_w = tf.Variable(tf.truncated_normal([embed_dim, h1]))
layer1_b = tf.Variable(tf.zeros([h1]))
layer2_w = []
layer2_b = []

for i in range(num_labels):
    layer2_w.append(tf.Variable(tf.truncated_normal([h1, 1])))
    layer2_b.append(tf.Variable(tf.zeros([1])))
    

def model(stock):
        
    train_stocks = stock
    # the particular stock being considered
        
    layer1_acts = tf.nn.relu(tf.matmul(train_data, layer1_w) + layer1_b)
    layer2_acts = tf.matmul(layer1_acts, layer2_w[train_stocks]) + layer2_b[train_stocks]

    global error
    error = error + tf.reduce_sum(tf.square(layer2_acts - train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(error)

    return error, optimizer


# optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(error)

    
data = np.random.rand(10, 300)
labels = np.random.rand(10, 1)
stock = [0, 1, 5, 4, 7, 17, 12, 25, 14, 2]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        for i in range(batch_size):
            loss = sess.run([model(stock[i])], feed_dict={train_data: [data[i]], train_labels: [labels[i]]})
            print(loss)
        # opt = sess.run([optimizer])
        # print("epoch = %d, loss = %f" % (epoch, loss))
    
