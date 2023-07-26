# Emoji-Predictor

glove.6B.50d.txt Dataset:-

https://www.kaggle.com/datasets/watts2/glove6b50dtxt?resource=download

Colab :- https://colab.research.google.com/drive/1hsqJKyd1SYHO0FLoJc8QdQpCwgQ2t2DI?usp=sharing



Importing Libraries: We start by importing the necessary libraries, such as pandas and numpy, which help with data manipulation and analysis.

Loading Data: We load the training and test data from CSV files using pandas. The data contains text and corresponding emoji labels.

Preparing the Data: We remove unnecessary columns from the training data and install the emoji library to work with emojis in Python.

Defining Emoji Dictionary: We create a dictionary that maps numeric labels to corresponding emojis. This will help us display the predicted and actual emojis later.

Preprocessing: We extract the text and labels from the training data and convert them into numpy arrays for further processing. We also define a function to retrieve word embeddings using a pre-trained word vector model called GloVe.

Embedding the Text: We use the GloVe word embeddings to convert the text data into a numerical representation that can be fed into a recurrent neural network (RNN) model.

Converting Labels: We convert the emoji labels into categorical format using one-hot encoding.

Building the Model: We import the necessary modules from Keras and TensorFlow to build our RNN model. The model consists of LSTM layers, dropout layers, and a dense layer with softmax activation.

Compiling and Training the Model: We compile the model with an optimizer and a loss function, and then train it on the training data. The training process is repeated for a certain number of epochs.

Visualizing Accuracy: We plot a graph showing the accuracy scores of the model during training and validation.

Evaluating the Model: We evaluate the model's performance on the training data and print the accuracy score.

Preparing Test Data: We preprocess the test data in a similar manner to the training data, including removing unnecessary characters.

Evaluating on Test Data: We evaluate the trained model on the test data and print the accuracy score.

Making Predictions: We use the trained model to predict emojis for each text in the test data. We print the text, predicted emoji, and actual emoji for comparison.

Overall, this project demonstrates the process of training an RNN model to predict emojis based on text input using word embeddings.


....................................................................................................................................................................................................................................................................................................
About the word embedding:-
In the context of natural language processing (NLP), "embedding" refers to the process of converting words or phrases into dense vectors of numerical values. Pre-trained word embeddings are word representations that have been learned on a large corpus of text data before being used in specific NLP tasks.

One popular method for creating word embeddings is called GloVe (Global Vectors for Word Representation). GloVe is an unsupervised learning algorithm that aims to capture the semantic meaning of words by analyzing their co-occurrence patterns in a large corpus of text. The underlying idea is that words that often appear together in similar contexts are likely to have similar meanings.

Here's how the process works:

Data Collection: A vast amount of text data is gathered from various sources, such as books, articles, websites, etc. This corpus contains numerous words and their contextual relationships.

Co-occurrence Matrix: A co-occurrence matrix is constructed, where each cell represents the number of times a word appears in the context of another word within a specified window of words.

Training GloVe: The co-occurrence matrix is used to train the GloVe model. The model learns to represent words as dense vectors in a high-dimensional space, capturing their semantic relationships based on the co-occurrence patterns.

Creating Word Embeddings: Once the GloVe model is trained, each word in the vocabulary is associated with a fixed-size dense vector, typically containing hundreds of dimensions. These vectors are the pre-trained word embeddings.

The advantage of using pre-trained word embeddings like GloVe is that they capture meaningful semantic relationships between words, even for rare or unseen words, which makes them beneficial for a wide range of NLP tasks. By using these pre-trained embeddings, you can leverage the knowledge gained from large text corpora without needing to train a word embedding model from scratch on your specific dataset, which can be computationally expensive and may require a massive amount of text data. This makes it easier to work with smaller datasets and still achieve meaningful results in various NLP applications, such as sentiment analysis, text classification, machine translation, and more.
