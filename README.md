# Emoji-Predictor

glove.6B.50d.txt:-

https://www.kaggle.com/datasets/watts2/glove6b50dtxt?resource=download


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
