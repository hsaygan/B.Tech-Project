import tensorflow as tf
import pickle
import os
import numpy as np
from create_sentiment_featuresets import create_feature_sets_and_labels

#with open("Data/sentiment_set.pickle",'rb') as f:
#    train_x, train_y, test_x, test_y = pickle.load(f)
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('../Data/pos.txt', '../Data/neg.txt')

print (test_y)

n_nodes_hl1 = 350
n_nodes_hl2 = 100

hm_epochs = 25
n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}


    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                i += batch_size

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

def get_files():
    with open("train_x.txt", "w+") as f:
        for item in train_x:
            f.write("%s\n" % item)

    with open("train_y.txt", "w+") as f:
        for item in train_y:
            f.write("%s\n" % item)

    with open("test_x.txt", "w+") as f:
        for item in test_x:
            f.write("%s\n" % item)

    with open("test_y.txt", "w+") as f:
        for item in test_y:
            f.write("%s\n" % item)

get_files()
train_neural_network(x)
