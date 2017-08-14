import numpy as np
import tensorflow as tf
import os, sys, pickle, math
import random

home_dir = "/media/bruno/data/chatbot_project/"
embedding_dir = home_dir + "embeddings/"
sent2sent_dir = home_dir + "sent2sent/"

print(embedding_dir)
sys.path.append(embedding_dir)

from embeddings_class2 import embedding_model

first_word, second_word = pickle.load( open(home_dir + "embed_dataset.pickle", "rb" ) )
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )

vocabulary_size = len(dictionary)
data_set_size = len(first_word)


vocabulary_size = len(dictionary)
data_set_size = len(first_word)

model = embedding_model(vocabulary_size=vocabulary_size)

model.train(first_word, second_word,num_steps=2001)

# #checking result
# model.import_session()
# ind = [random.randint(0, vocabulary_size - 1) for _ in range(30)]
# r = model.predict_embedding(ind)
# w = model.predict_inverse_embedding(r)
# for i,o in zip(ind, w):
#     print(dictionary[i], dictionary[o])