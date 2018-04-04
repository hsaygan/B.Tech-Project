import csv
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
import random

standard_pickle = "Temp/lexicon-0-2001.pickle"      #"Temp/lexicon-"+str(line_start)+"-"+str(line_end)+".pickle"
lemmatizer = WordNetLemmatizer()
if not os.path.exists(r"Temp"):
    print("\nCreating New Folder 'Temp'...")
    os.makedirs(r"Temp")

'''
0: -ve
2: neutral
4: +ve

[1,0]: -ve
[0,1]: +ve
'''

#Converts CSV file to format we desire
def initialize(source_file, starting_line, ending_line, output_file):
    with open(output_file, 'w+') as output_obj:
        output_writer = csv.writer(output_obj, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        with open(source_file, 'r', encoding='latin-1') as input_obj:
            reader_entire_file = list(csv.reader(input_obj))
            for line in reader_entire_file[starting_line:ending_line]:
                polarity = [0,0]
                line = list(line)

                if (int(line[0]) == 0):
                    polarity[0] = 1
                elif (int(line[0]) == 4):
                    polarity[1] = 1
                else:
                    continue

                tweet = str(re.sub(r'(\s)@\w+', r'\1',  " "+line[-1]))
                line = [polarity, tweet]

                output_writer.writerow(line)
                #print(line)
        print ("\n\t\tInitialization of the following file is completed: ", source_file)


#Creates a dictionary of words appearing in [line_start, line_end] lines
def create_lexicon(source_file, starting_line, ending_line, output_file):
    with open(output_file, 'wb') as output_obj:
        lexicon = []

        with open(source_file, 'r') as input_obj:
            reader_entire_file = list(csv.reader(input_obj))
            counter = 1
            content = ''

            for line in reader_entire_file[starting_line:ending_line]:
                #print ("\nLine ", counter, ":")
                counter += 1
                #if (counter/2500.0).is_integer():
                line = list(line)
                tweet = line[1]
                content += ' ' + tweet

                words = word_tokenize(content)
                words = [lemmatizer.lemmatize(i) for i in words]
                #print("\n\tNeglecting: ",end=''),
                words = [word if (len(word) > 1) else print(end='')  for word in words] #To display removed words: add print("word ",end='')
                lexicon = list(set(lexicon + words))
                #print ("\n\tLength of Lexicon: ",len(lexicon))

            print("\nLexicon is of Length:", len(lexicon), "\n", lexicon)
            pickle.dump(lexicon, output_obj)


#Vectorizes. Creates array of Zeros and increments the array if respective word appears for per Line of the source
def create_featuresets_and_labels(source_file, lexicon_pickle, output_file, save_separate_files = True):
    #print ("\nLoading Pickle: " + lexicon_pickle)
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
        feature_set = []
        labels = []
        if save_separate_files == True:
            with open("Temp/x_" + output_file, "w+") as x_file:
                with open("Temp/y_" + output_file, "w+") as y_file:
                    with open(source_file, buffering=20000, encoding='latin-1') as input_obj:
                        reader_entire_file = list(csv.reader(input_obj))

                        for line in reader_entire_file:
                            #print ("\nLine ", counter, ":")
                            line = list(line)
                            label = line[0].split(",")
                            label = [int((label[0])[1]), int((label[0])[1])]
                            tweet = line[1]

                            current_words = word_tokenize(tweet.lower())
                            current_words = [lemmatizer.lemmatize(i) for i in current_words]
                            features = np.zeros(len(lexicon), dtype=int)

                            for word in current_words:
                                if word.lower() in lexicon:
                                    index_value = lexicon.index(word.lower())
                                    features[index_value] += 1

                            #line = str(features)+':::'+str(label)+'\n'
                            feature_set.append(features)
                            labels.append(label)

            print ("\n\t\tFeaturesets and Labels for the following file is completed: ", source_file)
            return feature_set, labels

        elif save_separate_files == False:
            with open(output_file,'a') as output_obj:
                with open(source_file, buffering=20000, encoding='latin-1') as input_obj:
                    reader_entire_file = list(csv.reader(input_obj))
                    counter = 1

                    for line in reader_entire_file:
                        print ("\nLine ", counter, ":")
                        counter += 1
                        line = list(line)
                        label = line[0]
                        tweet = line[1]

                        current_words = word_tokenize(tweet.lower())
                        current_words = [lemmatizer.lemmatize(i) for i in current_words]
                        features = np.zeros(len(lexicon), dtype=int)

                        for word in current_words:
                            if word.lower() in lexicon:
                                index_value = lexicon.index(word.lower())
                                features[index_value] += 1
                        features = list(features)
                        line = str(features)+':::'+str(label)+'\n'

                        output_obj.write(line)
                        #print("\n\t", line)


#Delete Temp Files
def delete_file(path):
    os.remove(path)


#Shuffles lines for better Neural Network Learning
def shuffle(source_file, output_file):
    with open(source_file, "r") as read_file:
        li = read_file.readlines()
        random.shuffle(li)
        with open(output_file, "w+") as write_file:
            write_file.writelines(li)


#Initialization and Creating lexicon pickle
def create_standard_lexicon(source_file, starting_line, ending_line, output_file, lexicon_path):
    initialize(source_file, starting_line, ending_line, output_file)
    create_lexicon(output_file, starting_line, ending_line, lexicon_path)


#Retrieve Testing Featuresets and Labels
def get_test_data(Training_Data_Source, Testing_Data_Source, line_start, line_end):
    initialize(Testing_Data_Source, 0, -1, "Temp/test_initialized.csv")
    test_x, test_y = create_featuresets_and_labels("Temp/test_initialized.csv", standard_pickle, "test.csv")

    print ("\n\tPreprocessing for Testing Data Completed!")

    #Delete Temporary Files
    delete_file("Temp/test_initialized.csv")

    print ("\n\t\tTemporary Files are Deleted!")

    #Dumping all training and testing entities into pickle
    # with open("Temp/testing_featuresets_and_labels-"+str(line_start)+"-"+str(line_end)+".pickle", 'wb') as f:
    #     pickle.dump([test_x, test_y], f)

    return test_x, test_y


#Retrieve Training Featuresets and Labels
def get_train_data(Training_Data_Source, Testing_Data_Source, line_start, line_end):
    initialize(Training_Data_Source, line_start, line_end, "Temp/train_initalized.csv")
    shuffle("Temp/train_initalized.csv", "Temp/train_data.csv")
    train_x, train_y = create_featuresets_and_labels("Temp/train_data.csv", standard_pickle, "train.csv")

    print ("\n\tPreprocessing for Training Data Completed!")

    #Delete Temporary Files
    delete_file("Temp/train_initalized.csv")
    delete_file("Temp/train_data.csv")

    print ("\n\tTemporary Files are Deleted!")

    #Dumping all training and testing entities into pickle
    # with open("Temp/training_featuresets_and_labels-"+str(line_start)+"-"+str(line_end)+".pickle", 'wb') as f:
    #     pickle.dump([train_x, train_y], f)

    return train_x, train_y


if __name__ == "__main__":
    Training_Data_Source = "../../Large Files/More/training.1600000.processed.noemoticon.csv"   # "Data/train_source.csv"
    Testing_Data_Source = "../../Large Files/More/testdata.manual.2009.06.14.csv"               # "Data/test_source.csv"
    line_start = 0
    line_end = 2001

    # create_standard_lexicon(Training_Data_Source, line_start, line_end, "Temp/train_initalized.csv", standard_pickle)
    # get_train_data(Training_Data_Source, Testing_Data_Source, line_start, line_end)
    # get_test_data(Training_Data_Source, Testing_Data_Source, line_start, line_end)

#Length of Lexicon for [0, 2001) is 5499
#Length of Lexicon for [0, 200) is 1083
#Length of Lexicon for [0, 101) is 675
#180 lines for lexicon per minute, 1.6 Million Lines: 148.14 hours to form the complete lexicon
