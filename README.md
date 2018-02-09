# B.Tech-Project
All the updates and current work-in-progress for my B.Tech Project, along with Namish Narayan.

## Current Work:
  * Sentiment Data Analysis
    - Creating Featuresets from Abstract Data
    - Feeding vectorized data into Neural Network
    - Modifying parameters to improve accuracy

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
 
