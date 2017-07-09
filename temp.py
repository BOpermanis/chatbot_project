import os
import pandas as pd
import numpy as np
from string import ascii_lowercase as alphabet
import pickle

import tensorflow as tf

home_dir = "/media/bruno/data/chatbot_project/"

os.chdir("/media/bruno/data/chatbot_project/data_sets/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus")

dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )

