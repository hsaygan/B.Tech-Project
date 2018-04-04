import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
import random
import pickle                               #save data
from collections import Counter                 #Count stuff

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos,neg):            #Creating array for all the words found in both files
    lexicon = []
    for current_file in [pos,neg]:
        with open(current_file, "r") as f:
            contents = f.readlines()

            #Random Print
            line = contents[0]
            all_words_in_line = word_tokenize(line.lower())
            all_words_in_line = list(all_words_in_line)
            lehicon = [lemmatizer.lemmatize(i) for i in all_words_in_line]
            w_counters = Counter(lehicon)
            print("\n\tLine (1 line of the ",current_file," dataset): \n", line, "\tall_words_in_line (extrating words from lines): \n", all_words_in_line, "\n\tlexicon (Lemmatized, ie. only one form of a word): \n", lehicon, "\n\tw_counters (counts the total occurence of lexicon words):\n", w_counters,  "\n\n")
            #Random Print

            for line in contents[:hm_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
            #Lemmatizer (Lematize): Converting a work to a token word
            #lexicon = ["positive", "cat", "python", "dog"] etc
    w_counts = Counter(lexicon)
            #w_counts contains the no. of times each word occurs in lexicon
            #w_counts = {"the":52521, "and":25242}
    lexicon2  = []

    for word in w_counts:
        if 1000 > w_counts[word] > 50:
            lexicon2.append(word)

    print("Length of Lexicon (entire dataset): ", len(lexicon2))
    return lexicon2

def sample_handling(sample, lexicon, classification):
    featureset = []
    # Hot array per line. Per line, shows if the lexicon word is present or not. Along with SENTENCE's Class
    # Class : [1,0] --> Positive  ,  [0,1] --> Negative
    # featureset[] = [
    #                   [[0,1,0,1,1,0,1,0], [1,0]],
    #                   [][0,1,0,1,1,0,1,0], [0,1]],
    #                       ...     ]

    with open(sample, 'r') as f:
        contents = f.readlines()
        for line in contents[:hm_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    #Random Print
    print ("\n\nFeature Set[0] for ", sample, " (This is a vector which represents lexicon words occurences, along with their sentiment): \n", featureset[0])
    #Random Print

    return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)
        #Final Question : does tf.argmax([output]) == tf.argmax([expectations])
        #with shuffle : tf.argmax([52351,11293]) == tf.argmax([1,0])
        #without      : tf.argmax([9999999999,-999999999]) == tf.argmax([1,0])
        #since it first goes all in for positive and then for negative.
    features = np.array(features)
    testing_size = int(test_size*len(features))
        #train_x = list(features[:,0])
        #[[5,8], [7,9]]
        #Gets all the 0th element ie. returns [5,7]
        #[[features, labels]] ie [[0 1 1 1 0 1], [0,1]]
        #in here, we get features
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    #Random Print
    print ("\nAfter That, we shuffle adjust +ve and -ve datasets, and separate the O-H-A as train_y, and array as train_x.\n\n\ntrain_x[5]:\n", train_x[5], "\n\ntrain_y[5]:\n", train_y[0])
    #Random Print

    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    if not os.path.exists(r"Data"):
        print("\nCreating New Folder 'Data'...")
        os.makedirs(r"Data")
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('../Data/pos.txt', '../Data/neg.txt')


    with open('Data/sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)

create_feature_sets_and_labels
