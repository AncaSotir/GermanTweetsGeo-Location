###########################################################
###  PRACTICAL MACHINE LEARNING, AI UNIBUC              ###
###  KAGGLE COMPETITION: GEO-LOCATION OF GERMAN TWEETS  ###
###  STUDENT: SOTIR ANCA-NICOLETA (GROUP 407)           ###
###########################################################

#=========================================================#
#                                                         #
#  APPROACH: NEURAL NETWORK                               #
#  FEATURES: EMBEDDING LAYER, VOCABULARY INDEXING         #
#            (TWEET ENTRIES ARE PREPROCESSED FOR THIS)    #
#                                                         #
#=========================================================#



# ---------------------------------------------------------
# SETUP

!pip install emot

import os
import numpy as np

from emot.emo_unicode import  UNICODE_EMO, EMOTICONS
import re
import nltk; nltk.download('punkt'); nltk.download('stopwords')
import string
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
%load_ext tensorboard
import datetime, os



# ---------------------------------------------------------
# functions for TEXT PREPROCESSING (same as for the SVR)

def convert_emotes(text, instruction = 'remove'):
  '''
  Removes both emojis and emoticons from the text or replaces them with their description.
  Instruction should be either 'remove' or 'replace'.
  '''
  if instruction == 'remove': # default handling is to remove emojis and emoticons
    # replace all emojis and emoticon occurrences with an empty string
    for emot in UNICODE_EMO:
      text = text.replace(emot, '') 
    for emot in EMOTICONS:
      text = re.sub(u'('+emot+')', '', text) 
  if instruction == 'replace':
    # replace emoji/emoticon with its description (while also removing punctuation)
    for emot in UNICODE_EMO: 
      text = text.replace(emot, UNICODE_EMO[emot].replace(':','').replace('_',''))
    for emot in EMOTICONS:
      text = re.sub(u'('+emot+')', EMOTICONS[emot].replace(" ",'').replace(',',''), text)  
  return text


def preprocess_text(text):
  '''
  Returns a list of words obtained through preprocessing the text.
  '''
  text = convert_emotes(text) # handle emojis and emoticons - default is to remove them
  text = text.lower() # lowercase all text
  text = ''.join([char for char in text if char not in string.punctuation]) # remove punctuation
  text = nltk.word_tokenize(text) # make a list of words
  stop_words = nltk.corpus.stopwords.words('german') # retrieves a list of german stop words 
                                                     # that do not provide meaning to the text
  text = [word for word in text if word not in stop_words] # remove german stopwords
  porter = nltk.stem.porter.PorterStemmer()
  text = [porter.stem(word) for word in text] # keep only the stem of each word
  return text



# ---------------------------------------------------------
# READING THE DATA (as oposed to the code for the SVR, the 
# text preprocessing is done while reading the data, inside 
# the split_entry function called by read_files)

data_files_path = '/content/drive/MyDrive/PML/Project1/pml-2020-unibuc/'

def read_file(file_path):
  '''
  Returns a list of lines read from the file given by its path.
  '''
  with open(file_path, encoding = 'utf8') as datafile:
    data = datafile.read().split('\n')
  data.pop() # remove last element which is an empty line
  return data


def split_entry(entry, set_type):
  '''
  Split the entry string provided and return the id, latitude, longitude 
  (if the entry contains the coordinates) and the preprocessed tweet text.
  Possible set_type values: 'training', 'validation' or 'test'.
  '''
  if set_type in ['training', 'validation']: # the entry contains the coordinates
    id, latitude, longitude, text_content = entry.split(',', 3)
    return [id, float(latitude), float(longitude), preprocess_text(text_content)] # preprocess_text is called
  if set_type == 'test':
    id, text_content = entry.split(',', 1)
    return [id, preprocess_text(text_content)] # preprocess_text is called


def read_files(files_path = data_files_path):
  '''
  Read the training, validation and test files located at files_path.
  '''
  training_data_fpath = os.path.join(files_path, 'training.txt')
  validation_data_fpath = os.path.join(files_path, 'validation.txt')
  test_data_fpath = os.path.join(files_path, 'test.txt')
  
  # get the entries (lines) from the file
  training_data = read_file(training_data_fpath)
  print(f'Found {len(training_data)} training entries.')
  validation_data = read_file(validation_data_fpath)
  print(f'Found {len(validation_data)} validation entries.')
  test_data = read_file(test_data_fpath)
  print(f'Found {len(test_data)} test entries.')

  # split each line to obtain the id, coordinates and the preprocessed text of the tweets
  training_data = [split_entry(entry, 'training') for entry in training_data]
  validation_data = [split_entry(entry, 'validation') for entry in validation_data]
  test_data = [split_entry(entry, 'test') for entry in test_data]
  
  return training_data, validation_data, test_data

training_data, validation_data, test_data = read_files()



# ---------------------------------------------------------
# defining the FEATURES

# get only the preprocessed text of the tweets
train_tweets = [entry[3] for entry in training_data]
valid_tweets = [entry[3] for entry in validation_data]
test_tweets = [entry[1] for entry in test_data]

# a list containing the number of words in each training tweet
tweet_lenghts = [len(tweet) for tweet in train_tweets]

# get the maximum number of words of a tweet from the training set
max_tweet_length = max(tweet_lenghts)
print(max_tweet_length) # 100


# build the vocabulary on the training tweets only
vocabulary = {} # using a dictionary (keys are words, values are positive integers)
for entry in train_tweets:
  for word in entry:
    if word not in vocabulary:
      vocabulary[word] = len(vocabulary) + 1 # the first word will have index 1

vocabulary_size = len(vocabulary)
print(vocabulary_size) # 97354


def build_tweet_repres(tweet, vocabulary, max_length):
  '''
  These features will enter the embedding layer.
  tweet is a list of words (it is preprocessed as such)

  Each word in a tweet will be replaced by its index in the vocabulary
  (if the word is not in the vocabulary, it will be replaced by 0).
  Also adds a padding of zeros untill tweet reaches length of max_Length
  (if length is greater than max_length, only the first max_length words are considered).

  Each tweet will be represented by a list of positive indexes of the same length max_length.
  '''
  indexed_tweet = [vocabulary.get(word, 0) for word in tweet] # get function returns 0 if word not in vocabulary
  return (indexed_tweet + [0] * max_length)[:max_length] # adds padding of zeros and takes only first max_length words


# encode the training, validation and test tweets and define tensors containing those features
encoded_train_tweets = tf.convert_to_tensor([build_tweet_repres(tweet, vocabulary, max_tweet_length) for tweet in train_tweets])
encoded_valid_tweets = tf.convert_to_tensor([build_tweet_repres(tweet, vocabulary, max_tweet_length) for tweet in valid_tweets])
encoded_test_tweets = tf.convert_to_tensor([build_tweet_repres(tweet, vocabulary, max_tweet_length) for tweet in test_tweets])

# define tensors containing the target coordinates for the training and validation entries
train_coords = tf.convert_to_tensor([[entry[1], entry[2]] for entry in training_data])
valid_coords = tf.convert_to_tensor([[entry[1], entry[2]] for entry in validation_data])



# ---------------------------------------------------------
# defining the MODEL

# this directory will contain a 'checkpoints' folder and a 'logs' folder
# (to save the model state at different epochs and to visualize a TensorBoard)
misc_dir = '/content/drive/MyDrive/PML/Project1/neural net'

# how to save the model checkpoints at different epochs
checkpoint_callback = ModelCheckpoint(
    filepath = os.path.join(misc_dir, 'checkpoints/model.{epoch:05d}.hdf5')
)

# how to save data to be plotted by the TensorBoard
log_dir = os.path.join(misc_dir, 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)



initializer = tf.keras.initializers.GlorotUniform() # weight initialization for the middle dense layers

# defining the neural network
model = Sequential([
  # input is of size (batch x max_tweet_length), 
  # the embedding layer will have an output of size (batch x max_tweet_length x output_dim)
  # (32 x 100) -> (32 x 100 x 25)
  Embedding(input_dim = vocabulary_size + 1, output_dim = 25, input_length = max_tweet_length),
  Flatten(), # flatten the features for the next dense layers
  Dense(512, activation = 'relu', kernel_initializer = initializer),  # 512 neurons
  Dense(1024, activation = 'relu', kernel_initializer = initializer), # 1024 neurons
  Dense(512, activation = 'relu', kernel_initializer = initializer),  # 512 neurons
  Dropout(0.4), # regularization: dropout rate is 0.4
  Dense(2) # final dense layer with two neurons to predict the latitude and longitude respectively
])



# ---------------------------------------------------------
# OPTIMIZING THE MODEL

# stochastic gradient descent optimizer
optimizer = SGD(learning_rate = 0.01, momentum = 0.9) 

# MAE and MSE metrics to see the progress
model.compile(optimizer = optimizer, loss = 'MAE', metrics = ['MeanAbsoluteError', 'MeanSquaredError'])

# training the model for 200 epocs, in batch sizes of 32
model.fit(encoded_train_tweets, train_coords,
          epochs=200, batch_size=32, initial_epoch=0,
          callbacks = [checkpoint_callback, tensorboard_callback],
          validation_data = (encoded_valid_tweets, valid_coords))

# Epoch 1/200
# 706/706 [==============================] - 24s 33ms/step - loss: 6.7779 - mean_absolute_error: 6.7779 - mean_squared_error: 200.0665 - val_loss: 1.0434 - val_mean_absolute_error: 1.0434 - val_mean_squared_error: 1.4733
# Epoch 2/200
# 706/706 [==============================] - 23s 32ms/step - loss: 1.9312 - mean_absolute_error: 1.9312 - mean_squared_error: 6.5934 - val_loss: 1.3486 - val_mean_absolute_error: 1.3486 - val_mean_squared_error: 2.4930
# Epoch 3/200
# 706/706 [==============================] - 23s 32ms/step - loss: 1.8641 - mean_absolute_error: 1.8641 - mean_squared_error: 6.0707 - val_loss: 1.5481 - val_mean_absolute_error: 1.5481 - val_mean_squared_error: 3.1321
# Epoch 4/200
# 706/706 [==============================] - 23s 33ms/step - loss: 1.7478 - mean_absolute_error: 1.7478 - mean_squared_error: 5.2473 - val_loss: 0.8603 - val_mean_absolute_error: 0.8603 - val_mean_squared_error: 1.1302
# Epoch 5/200
# 706/706 [==============================] - 23s 33ms/step - loss: 1.6274 - mean_absolute_error: 1.6274 - mean_squared_error: 4.5650 - val_loss: 0.9363 - val_mean_absolute_error: 0.9363 - val_mean_squared_error: 1.3195
# ...
# ...
# ...
# Epoch 195/200
# 706/706 [==============================] - 22s 31ms/step - loss: 0.0785 - mean_absolute_error: 0.0785 - mean_squared_error: 0.0135 - val_loss: 0.5962 - val_mean_absolute_error: 0.5962 - val_mean_squared_error: 0.6753
# Epoch 196/200
# 706/706 [==============================] - 21s 30ms/step - loss: 0.0775 - mean_absolute_error: 0.0775 - mean_squared_error: 0.0125 - val_loss: 0.5984 - val_mean_absolute_error: 0.5984 - val_mean_squared_error: 0.6833
# Epoch 197/200
# 706/706 [==============================] - 22s 30ms/step - loss: 0.0778 - mean_absolute_error: 0.0778 - mean_squared_error: 0.0126 - val_loss: 0.5918 - val_mean_absolute_error: 0.5918 - val_mean_squared_error: 0.6738
# Epoch 198/200
# 706/706 [==============================] - 22s 31ms/step - loss: 0.0782 - mean_absolute_error: 0.0782 - mean_squared_error: 0.0128 - val_loss: 0.5950 - val_mean_absolute_error: 0.5950 - val_mean_squared_error: 0.6757
# Epoch 199/200
# 706/706 [==============================] - 21s 30ms/step - loss: 0.0786 - mean_absolute_error: 0.0786 - mean_squared_error: 0.0129 - val_loss: 0.5961 - val_mean_absolute_error: 0.5961 - val_mean_squared_error: 0.6729
# Epoch 200/200
# 706/706 [==============================] - 21s 30ms/step - loss: 0.0789 - mean_absolute_error: 0.0789 - mean_squared_error: 0.0133 - val_loss: 0.6029 - val_mean_absolute_error: 0.6029 - val_mean_squared_error: 0.6886


# visualize the model's progress with TensorBoard
%tensorboard --logdir '/content/drive/MyDrive/PML/Project1/neural net/logs'


# used for writing the submission file
test_ids = [entry[0] for entry in test_data]

# predict the coordinates on the test features
predictions = model.predict(encoded_test_tweets)

# write the submission file
with open('/content/drive/MyDrive/PML/Project1/submissions/submission_nn.txt', 'w') as submission_file:
  submission_file.write('id,lat,long\n')
  for i in range(len(predictions)):
    submission_file.write(test_ids[i] + ',' + str(predictions[i][0]) + ',' + str(predictions[i][1]) + '\n')

