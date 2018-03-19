# B.Tech-Project
All the updates and current work-in-progress for my B.Tech Project, along with [Namish Narayan](https://github.com/Namish123) under the guidance of Kshitiz Verma Sir.

## Current Work:
  * Sentiment Data Analysis
    - Creating Featuresets from Abstract Data
    - Feeding vectorized data into Neural Network
    - Modifying parameters to improve accuracy
    - Inputting large amount of data in the network

## Sentiment Data Analysis Overview:
  * Creating Organized and Vectorized Data
    - Sentiment data is extracted from 2 txt file, pos.txt and neg.txt that contains respective type of sentences.
    - The program reads the files line by line, and lemmatizes the words for each line.
    - Repeating the process for every line from data, gives lexicon of the data.
    - Lexicon contents are shuffled (in order to maintain randomness of +ve and -ve data)
    - Training Data, Testing Data, Training Labels and Testing Labels are created, by pairing the vectorized Lexicon's count of occurence for each line, alone with a one-hot-array to denote positive or negative sentiment of that specific line.
    - This data is then stored as pickle, and can be accessed externally.

  * Data Feed into Neural Network
<p align="center">
<img src="https://i.imgur.com/wMjbY4X.png">
</p>
    - This calculates the total accuracy of the model.

 ## How To Build:
   - Install pip3 (for Python 3)
   - Install Essential libraries like [TensorFlow](https://www.tensorflow.org/), [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/)
```
     pip3 install tensorflow
```
   - Install NLTK Library for Python
 ```
     pip3 install nltk
     #Open Python in Commandline
     python3
     #Download all the required data of NLTK
     nltk.download('all')
```
   - Clone this repository and run
```  
     python3 neural_net.py
```

Please leave all our suggestions and updates in issues. Thank You.
