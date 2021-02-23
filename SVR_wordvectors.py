###########################################################
###  PRACTICAL MACHINE LEARNING, AI UNIBUC              ###
###  KAGGLE COMPETITION: GEO-LOCATION OF GERMAN TWEETS  ###
###  STUDENT: SOTIR ANCA-NICOLETA (GROUP 407)           ###
###########################################################

#=========================================================#
#                                                         #
#  APPROACH: SVR                                          #
#  FEATURES: WORD VECTORS USING SPACY PRE-TRAINED MODEL   #
#                                                         #
#=========================================================#




# ---------------------------------------------------------
# setup

!pip install emot

import os

from emot.emo_unicode import  UNICODE_EMO, EMOTICONS
import re
import nltk; nltk.download('punkt'); nltk.download('stopwords')
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



# ---------------------------------------------------------
# reading the data

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
  Split the entry string provided and return the id,  latitude, longitude 
  (if the entry contains the coordinates) and tweet text.
  Possible set_type values: 'training', 'validation' or 'test'.
  '''
  if set_type in ['training', 'validation']: # the entry contains the coordinates
    id, latitude, longitude, text_content = entry.split(',', 3)
    return [id, float(latitude), float(longitude), text_content]
  if set_type == 'test':
    return entry.split(',', 1)


def read_files(files_path):
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

  # split each line to obtain the id, coordinates and the text of the tweets
  training_data = [split_entry(entry, 'training') for entry in training_data]
  validation_data = [split_entry(entry, 'validation') for entry in validation_data]
  test_data = [split_entry(entry, 'test') for entry in test_data]
  
  return training_data, validation_data, test_data



# ---------------------------------------------------------
# text preprocessing

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


class Tweet():
  '''
  Encapsulates the attributes of a tweet:
    id
    the set type ('training', 'validation', 'test')
    coordinates (if set type is 'test' then they are None)
    content (preprocessed - a list of stemmed words)
  '''

  def __init__(self, set_type, tweet_info):
    # tweet_info is an array of size 4 or 2 (based on set type)
    self.set_type = set_type
    self.id = tweet_info[0]
    if self.set_type in ['training', 'validation']:
      self.latitude = tweet_info[1]
      self.longitude = tweet_info[2]
      raw_content = tweet_info[3]
    if self.set_type == 'test':
      self.latitude = None 
      self.longitude = None
      raw_content = tweet_info[1]
    self.content = preprocess_text(raw_content)


raw_training_data, raw_validation_data, raw_test_data = read_files(data_files_path)
# Found 22583 training entries.
# Found 3044 validation entries.
# Found 3138 test entries.

training_data = [Tweet('training', entry) for entry in raw_training_data]
validation_data = [Tweet('validation', entry) for entry in raw_validation_data]
test_data = [Tweet('test', entry) for entry in raw_test_data]

print(raw_test_data[0][1])
print(test_data[0].content)
# "👩min vibi funktionkert nöd... 👧hesch d'batterie dri gsteckt? 👩ja, aber das isch nöd sgliiche..."
# ['min', 'vibi', 'funktionkert', 'nöd', 'hesch', 'dbatteri', 'dri', 'gsteckt', 'ja', 'isch', 'nöd', 'sgliich']



# ---------------------------------------------------------
# data visualization

def plot_coords(data):
  '''
  data is an array of size 2: [training_data, validation_data] (each of those containing Tweet objects)
  a subplot will be generated for each element in data
  '''
  fig, axs = plt.subplots(1, 2, figsize = (30, 10)) # create the subplots
  fig.subplots_adjust(wspace = 0.3) # add some vertical space between the subplots
  fig.patch.set_facecolor('white') # background color
  for i in range(2):
    x = []
    y = []
    for tweet in data[i]:
      x.append(tweet.latitude)
      y.append(tweet.longitude)
    axs[i].scatter(x, y)
    axs[i].set_xlabel('Latitude')
    axs[i].set_ylabel('Longitude')
    # change the ticks frequency on the x axis
    start, end = axs[i].get_xlim()
    axs[i].xaxis.set_ticks(np.arange(start, end, 0.4))
    # change the ticks frequency on the y axis
    start, end = axs[i].get_ylim()
    axs[i].yaxis.set_ticks(np.arange(start, end, 0.4))
    # format the ticks values as floats with precision 2
    axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  axs[0].set_title('Tweets from training set')
  axs[1].set_title('Tweets from validation set')


plot_coords([training_data, validation_data])



# ---------------------------------------------------------
# defining the features

!pip install spacy==2.3.5
!python -m spacy download de_core_news_lg

import spacy
import de_core_news_lg # pre-trained for German written text

nlp = de_core_news_lg.load()

# visualize an example of a vector for a tweet
doc = nlp('Ich: Also langsam söti scho schlafe Au Ich: Hesch di scho mal gwundered ob all mensche all farve glich gsehnd?')
len(doc.vector) # 300


def get_features(data, set_type):
  '''
  set type is either 'training', 'validation' or 'test'
  data is a list of Tweet objects
  '''
  X = [] # this holds the features
  if set_type in ['training', 'validation']:
    y = [] # this holds the target coordinates
    for tweet in data:
      doc = nlp(' '.join(tweet.content)) # feed the preprocessed tweet to the spacy model
      X.append(doc.vector) # doc.vector is the mean of all word vectors of its content
      y.append([tweet.latitude, tweet.longitude])
    return X, y # the features for training/validation and the corresponding target coords
  if set_type == 'test':
    for tweet in data:
      doc = nlp(' '.join(tweet.content))
      X.append(doc.vector)
    return X # the features for the testing


# get the features and target coordinates for the sets
X_train, y_train = get_features(training_data, 'training')
X_valid, y_valid = get_features(validation_data, 'validation')
X_test = get_features(test_data, 'test')



# ---------------------------------------------------------
# the SVR model

# define a pipeline to firstly scale the features, 
# then feed them to a SVR model wrapped in a MultiOutputRegressor
regr = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR())) # rbf, c=1

regr.fit(X_train, y_train) # fit the model to the training data

pred_valid = regr.predict(X_valid) # obtain the predictions for the validation data


# RESULTS ON THE VALIDATION SET
print(f'Mean Absolute Error:\t{mean_absolute_error(y_valid, pred_valid)}')
print(f'Mean Squared Error:\t{mean_squared_error(y_valid, pred_valid)}')
# Mean Absolute Error:	0.6651754423442366
# Mean Squared Error:   0.7930176131184464


pred_test = regr.predict(X_test) # obtain the predictions for the test data

# writing the submission file
with open('/content/drive/MyDrive/PML/Project1/submission_svr.txt', 'w') as submission_file:
  submission_file.write('id,lat,long\n')
  for i in range(len(test_data)):
    submission_file.write(test_data[i].id + ',' + str(pred_test[i][0]) + ',' + str(pred_test[i][1]) + '\n')
