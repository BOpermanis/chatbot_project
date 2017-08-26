import os, sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sent2sent.kmeans3_weighted as kmeans

home_dir = "/media/bruno/data/chatbot_project/"
os.chdir("/media/bruno/data/chatbot_project/data_sets/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus")
sys.path.append(home_dir)

## importing previous results
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )
tab = pickle.load( open(home_dir + "tab.pickle", "rb" ) )
convs = pickle.load( open(home_dir + "dataset.pickle", "rb" ) )

convs = [con for con in convs if len(con) <= 10]

ns = [len(i) for i in convs]
plt.hist(ns)
plt.show()