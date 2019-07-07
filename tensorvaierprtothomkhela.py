import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE


ratings = pd.read_csv('ratings.dat', sep="::", header=None, engine='python')

ratings_pivot = pd.pivot_table(ratings[[0, 1, 2]], values=2, index=0, columns=1).fillna(0)

X_train, X_test = train_test_split(ratings_pivot, train_size=0.8)

n_nodes_inpl = 3706
n_nodes_hl1 = 256
n_nodes_outl = 3706
hidden_1_layer_vals = {'weights': tf.Variable(tf.random_normal([n_nodes_inpl, n_nodes_hl1]))}

output_layer_vals = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1+1, n_nodes_outl]))}

input_layer = tf.placeholder('float', [None, 3706])

layer_1 = tf.nn.sigmoid(tf.matmul(input_layer,hidden_1_layer_vals['weights']))
layer1_const = tf.fill([tf.shape(layer_1)[0], 1],1.0)
layer_concat = tf.concat([layer_1, layer1_const], 1)
output_layer = tf.matmul( layer_concat,output_layer_vals['weights'])
output_true = tf.placeholder('float', [None, 3706])
meansq = tf.reduce_mean(tf.square(output_layer - output_true))
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 100  # how many images to use together for training
hm_epochs = 200    # how many times to go through the entire dataset
tot_users = X_train.shape[0] # total number of images

for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0

    for i in range(int(tot_users/batch_size)):
        epoch_x = X_train[i*batch_size: (i+1)*batch_size], c = sess.run([optimizer, meansq], feed_dict={input_layer: epoch_x, output_true: epoch_x})
        epoch_loss += c

    output_train = sess.run(output_layer, feed_dict={input_layer:X_train})
    output_test = sess.run(output_layer, feed_dict={input_layer:X_test})

    print('MSE train', MSE(output_train, X_train),'MSE test', MSE(output_test, X_test))
    print('Epoch', epoch, '/', hm_epochs, 'loss:', epoch_loss)

    sample_user = X_test.iloc[99, :]
    sample_user_pred = sess.run(output_layer, feed_dict={input_layer: [sample_user]})
