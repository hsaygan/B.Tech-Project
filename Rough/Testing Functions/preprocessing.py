import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()

'''
0: -ve
2: neutral
4: +ve

[1,0]: -ve
[0,1]: +ve
'''

#Converts CSV file to format we desire
def initialize(source_file, starting_line, ending_line, output_file):
    with open(output_file, 'a') as output_obj:
        output_writer = csv.writer(output_obj, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        with open(source_file, 'r') as input_obj:
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

                tweet = re.sub(r'(\s)@\w+', r'\1',  " "+line[-1])
                line = [polarity, tweet]

                output_writer.writerow(line)
                print(line)


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
def create_featuresets(source_file, lexicon_pickle, output_file):
    print ("\nLoading Pickle: " + lexicon_pickle)
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)

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
                    print("\n\t", line)


#Shuffles lines for better Neural Network Learning
def shuffle(source_file, output_file):
    data = pd.read_csv(source_file, error_bad_lines=False)
    data = data.iloc[np.random.permutation(len(data))]
    print(data.head())
    data.to_csv(output_file, index=False)

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


if __name__ == "__main__":
    line_start = 0
    line_end = 10

    #For Training Data
    initialize("train_source.csv", 0, -1, "train_initalized.csv")
    lexicon_count = create_lexicon("train_initalized.csv", 0, 50, "lexicon-"+str(line_start)+"-"+str(line_end)+".pickle")
    shuffle("train_initalized.csv", "train_shuffled.csv")

    #For Testing Data
    initialize("test_source.csv", 0, -1, "test_initialized.csv")
    create_featuresets("test_initialized.csv", "lexicon-"+str(line_start)+"-"+str(line_end)+".pickle", "test_vector.csv")
    create_test_data_pickle("test_vector.csv")
