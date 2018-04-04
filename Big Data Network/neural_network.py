import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from preprocessing import get_train_and_test_data, initialize, create_lexicon
import csv
lemmatizer = WordNetLemmatizer()

Training_Data_Source = "../../Large Files/More/training.1600000.processed.noemoticon.csv"   # "Data/train_source.csv"
Testing_Data_Source = "../../Large Files/More/testdata.manual.2009.06.14.csv"               # "Data/test_source.csv"

input_nodes = 675
n_nodes_hl1 = 500
n_nodes_hl2 = 200
n_classes = 2

batch_size = 100
total_batches = int(1600000/batch_size)
hm_epochs = 10

line_start = 0
line_end = 2001

# with open("Temp/feature_set_and_labels-0-101.pickle",'rb') as f:
#     train_x, train_y, test_x, test_y = pickle.load(f)

create_custom_lexicon(Training_Data_Source, line_start, line_end, "Temp/train_initalized.csv", "Temp/lexicon-"+str(line_start)+"-"+str(line_end)+".pickle")
train_x, train_y, test_x, test_y = get_train_and_test_data(Training_Data_Source, Testing_Data_Source, line_start, line_end)

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([input_nodes, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
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
            saver.save(sess, "model.ckpt")

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        saver.restore(sess,"model.ckpt")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)

# def train_neural_network(x):
#     prediction = neural_network_model(x)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         try:
#             epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
#             print('STARTING:',epoch)
#         except:
#             epoch = 1
#
#         while epoch <= hm_epochs:
#             if epoch != 1:
#                 saver.restore(sess,"model.ckpt")
#             epoch_loss = 1
#             with open('Temp/lexicon-0-1001.pickle','rb') as f:
#                 lexicon = pickle.load(f)
#             with open('Temp/train_data.csv', buffering=20000, encoding='latin-1') as f:
#
#
#                 feature_sets = []
#                 labels = []
#                 counter = 0
#                 with open('Temp/test_vector.csv', buffering=20000) as f:
#                     for line in f:
#                         try:
#                             features = list(eval(line.split('::')[0]))
#                             label = list(eval(line.split('::')[1]))
#                             feature_sets.append(features)
#                             labels.append(label)
#                             counter += 1
#                         except:
#                             pass
#                 train_x = np.array(feature_sets)
#                 train_y = np.array(labels)
#
#                 ##################################################
#                 reader = list(csv.reader(f))
#                 batch_x = []
#                 batch_y = []
#                 batches_run = 0
#                 for line in reader:
#                     line = list(line)
#                     label = line[0]
#                     tweet = line[1]
#                     current_words = word_tokenize(tweet.lower().strip())
#                     current_words = [lemmatizer.lemmatize(i) for i in current_words]
#
#                     features = np.zeros(len(lexicon))
#
#                     for word in current_words:
#                         if word.lower() in lexicon:
#                             index_value = lexicon.index(word.lower())
#                             features[index_value] += 1
#                     line_x = list(features)
#                     line_y = eval(label)
#                     batch_x.append(line_x)
#                     batch_y.append(line_y)
#                     if len(batch_x) >= batch_size:
#                         _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
#                                                                   y: np.array(batch_y)})
#                         epoch_loss += c
#                         batch_x = []
#                         batch_y = []
#                         batches_run +=1
#                         print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)
#
#             saver.save(sess, "model.ckpt")
#             print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
#             with open(tf_log,'a') as f:
#                 f.write(str(epoch)+'\n')
#             epoch +=1
#
#
# def test_neural_network():
#     prediction = neural_network_model(x)
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         for epoch in range(hm_epochs):
#             try:
#                 saver.restore(sess,"model.ckpt")
#             except Exception as e:
#                 print(str(e))
#             epoch_loss = 0
#
#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#         feature_sets = []
#         labels = []
#         counter = 0
#         with open('Temp/test_data.csv', buffering=20000) as f:
#             for line in f:
#                 try:
#                     features = list(eval(line.split('::')[0]))
#                     label = list(eval(line.split('::')[1]))
#                     feature_sets.append(features)
#                     labels.append(label)
#                     counter += 1
#                 except:
#                     pass
#         print('Tested',counter,'samples.')
#         test_x = np.array(feature_sets)
#         test_y = np.array(labels)
#         print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
#
#
# train_neural_network(x)
# test_neural_network()
