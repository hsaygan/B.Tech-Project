import tensorflow as tf
import pickle
import os
import numpy as np
from create_sentiment_featuresets import create_feature_sets_and_labels, create_lexicon
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import random
import pickle                                   #Save data
from collections import Counter                 #Count stuff

lemmatizer = WordNetLemmatizer()
pos = "../Data/pos.txt"
neg = "../Data/neg.txt"

#with open("Data/sentiment_set.pickle",'rb') as f:
#    train_x, train_y, test_x, test_y = pickle.load(f)
train_x, train_y, test_x, test_y = create_feature_sets_and_labels(pos, neg)

n_nodes_hl1 = 350
n_nodes_hl2 = 100

hm_epochs = 25
n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output

saver = tf.train.Saver()

def train_neural_network(x):
    print ("\n\n================ Creating Neural Network Model")
    prediction = neural_network_model(x)
    print ("\n\n================ Training Neural Network")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            if epoch != 0:
                saver.restore(sess,"./temp/model.ckpt")
                print ("\n\tLoading session from model.ckpt ...")
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

            saver.save(sess, "./temp/model.ckpt")
            print ("\n\tSaving session to model.ckpt ...\n")
            print('Epoch', epoch+1, 'completed out of',hm_epochs,' | Loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('\nAccuracy:',accuracy.eval({x:test_x, y:test_y}))

def use_neural_network(input_data):
    print ("\n\n================ Testing Neural Network Model")
    prediction = neural_network_model(x)
    my_lexicon = create_lexicon(pos, neg)
    # with open('./temp/lexicon.pickle','rb') as f:
    #     my_lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"./temp/model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(my_lexicon))

        for word in current_words:
            if word.lower() in my_lexicon:
                index_value = my_lexicon.index(word.lower())
                features[index_value] += 1

        features = np.array(list(features))

        #print ("\n FEATURES : \n\tShape = ", features.shape, "\n\tValues = ", features, "\n\n")
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
            print('\tPositive:',input_data)
        elif result[0] == 1:
            print('\tNegative:',input_data)

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

train_neural_network(x)
use_neural_network("That's a silly criticism")          #Negative
use_neural_network("DiCaprio has an amazing charm")     #Positive
use_neural_network("I'm not Happy")                     #Not Obvious
#get_files()
