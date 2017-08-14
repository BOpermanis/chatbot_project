import numpy as np
import tensorflow as tf
import os, pickle, math
import random

home_dir = "/media/bruno/data/chatbot_project/"
os.chdir("/media/bruno/data/chatbot_project/embeddings")


first_word, second_word = pickle.load( open(home_dir + "embed_dataset.pickle", "rb" ) )
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )

vocabulary_size = len(dictionary)
data_set_size = len(first_word)
