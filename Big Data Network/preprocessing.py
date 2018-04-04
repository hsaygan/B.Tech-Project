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


#Creates a dictionary of words appearing in [line_start, line_end] lines
def create_lexicon(source_file, starting_line, ending_line, output_file):
    with open(output_file, 'wb') as output_obj:
        lexicon = []

        with open(source_file, 'r') as input_obj:
            reader_entire_file = list(csv.reader(input_obj))
            counter = 1
            content = ''

            for line in reader_entire_file[starting_line:ending_line]:
                print ("\nLine ", counter, ":")
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
                print ("\n\tLength of Lexicon: ",len(lexicon))

            #print(lexicon)
            pickle.dump(lexicon, output_obj)


#Vectorizes. Creates array of Zeros and increments the array if respective word appears for per Line of the source
def create_featuresets_and_labels(source_file, lexicon_pickle, output_file, save_separate_files = True):
    print ("\nLoading Pickle: " + lexicon_pickle)
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
        feature_set = []
        labels = []
        if save_separate_files == True:
            with open("Temp/x_" + output_file, "w+") as x_file:
                with open("Temp/y_" + output_file, "w+") as y_file:
                    with open(source_file, buffering=20000, encoding='latin-1') as input_obj:
                        reader_entire_file = list(csv.reader(input_obj))
                        counter = 1

                        for line in reader_entire_file:
                            print ("\nLine ", counter, ":")
                            counter += 1
                            line = list(line)
                            label = line[0].split(",")
                            tweet = line[1]
                            label = [int((label[0])[1]), int((label[0])[1])]
                            print ("\nlabel: ", label)

                            current_words = word_tokenize(tweet.lower())
                            current_words = [lemmatizer.lemmatize(i) for i in current_words]
                            features = np.zeros(len(lexicon), dtype=int)

                            for word in current_words:
                                if word.lower() in lexicon:
                                    index_value = lexicon.index(word.lower())
                                    features[index_value] += 1

                            # x_file.write("%s\n" % str(features))
                            # y_file.write("%s\n" % str(label))
                            line = str(features)+':::'+str(label)+'\n'
                            feature_set.append(features)
                            labels.append(label)

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


#Shuffles lines for better Neural Network Learning
def shuffle(source_file, output_file):
    # data = pd.read_csv(source_file, error_bad_lines=False)
    # data = data.iloc[np.random.permutation(len(data))]
    # print(data.head())
    # data.to_csv(output_file, index=False)

    fid = open(source_file, "r")
    li = fid.readlines()
    fid.close()

    random.shuffle(li)
    fid = open(output_file, "w")
    fid.writelines(li)
    fid.close()


#Delete Temp Files
def delete_file(path):
    os.remove(path)


#NOT DONE YET!
def create_test_data_pickle(fin):
	feature_sets = []
	labels = []
	counter = 0
	with open(fin, buffering=20000) as f:
		for line in f:
			try:
				features = list(eval(line.split('::')[0]))
				label = list(eval(line.split('::')[1]))

				feature_sets.append(features)
				labels.append(label)
				counter += 1
			except:
				pass

	print(counter)
	feature_sets = np.array(feature_sets)
	labels = np.array(labels)


#Initialization and Creating lexicon pickle
def create_custom_lexicon(source_file, starting_line, ending_line, output_file, lexicon_path):
    initialize(source_file, starting_line, ending_line, output_file)
    create_lexicon(output_file, starting_line, ending_line, lexicon_path)


#Main Function
def get_train_and_test_data(Training_Data_Source, Testing_Data_Source, line_start, line_end):
    #For Training Data
    initialize(Training_Data_Source, line_start, line_end, "Temp/train_initalized.csv")
    #create_lexicon("Temp/train_initalized.csv", line_start, line_end, "Temp/lexicon-"+str(line_start)+"-"+str(line_end)+".pickle")
    shuffle("Temp/train_initalized.csv", "Temp/train_data.csv")
    train_x, train_y = create_featuresets_and_labels("Temp/train_data.csv", "Temp/lexicon-"+str(line_start)+"-"+str(line_end)+".pickle", "train.csv")

    print ("\n\n\t Preprocessing for Training Data Completed!\n\n")

    #For Testing Data
    initialize(Testing_Data_Source, 0, -1, "Temp/test_initialized.csv")
    test_x, test_y = create_featuresets_and_labels("Temp/test_initialized.csv", "Temp/lexicon-"+str(line_start)+"-"+str(line_end)+".pickle", "test.csv")

    print ("\n\n\t Preprocessing for Testing Data Completed!\n\n")

    #Delete Temporary Files
    delete_file("Temp/train_initalized.csv")
    delete_file("Temp/test_initialized.csv")
    delete_file("Temp/train_data.csv")

    print ("\n\n\t Temporary Files are Deleted!\n\n")

    #Dumping all training and testing entities into pickle
    # with open("Temp/feature_set_and_labels-"+str(line_start)+"-"+str(line_end)+".pickle", 'wb') as f:
    #     pickle.dump([train_x, train_y, test_x, test_y], f)

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    Training_Data_Source = "../../Large Files/More/training.1600000.processed.noemoticon.csv"   # "Data/train_source.csv"
    Testing_Data_Source = "../../Large Files/More/testdata.manual.2009.06.14.csv"               # "Data/test_source.csv"
    line_start = 0
    line_end = 2001

    get_train_and_test_data(Training_Data_Source, Testing_Data_Source, line_start, line_end)

#Length of Lexicon for [0, 200) is 1083
#Length of Lexicon for [0, 101) is 675
#180 lines for lexicon per minute, 1.6 Million Lines: 148.14 hours to form the complete lexicon
