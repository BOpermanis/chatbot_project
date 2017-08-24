# import tensorflow as tf
import numpy as np

import sys, os, pickle, math
import random
import datetime

home_dir = "/media/bruno/data/chatbot_project/"
embedding_dir = home_dir + "embeddings/"
sent2sent_dir = home_dir + "sent2sent/"

sys.path.append(embedding_dir)

first_word, second_word = pickle.load( open(home_dir + "embed_dataset.pickle", "rb" ) )
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )
# batches = pickle.load( open(home_dir + "batches.pickle", "rb" ) )

vocabulary_size = len(dictionary)
data_set_size = len(first_word)


vocabulary_size = len(dictionary)
data_set_size = len(first_word)


def get_rand_seq(alt_num):
    ns = random.randint(6,7)
    a = []
    for i in range(ns):
        a.extend([k for k in range(4, 4+alt_num)])
    return a


def make_longer(l1,maxl):
    l = list(l1)
    while len(l)<maxl:
        l.append(pad)
    return l


def make_array(alt_len):
    dat = [[start] + get_rand_seq(alt_len) + [eos] for _ in range(n)]
    m = max([len(i) for i in dat])
    for i in range(len(dat)):
        dat[i] = make_longer(dat[i], m)
    return np.asarray(dat).astype(np.int32)


start = revdictionary["_START"]
eos = revdictionary["_EOS"]
pad = revdictionary["_PAD"]

n = 200

batches = []
for k in range(10):
    ar = make_array(3)
    batches.append( (ar,ar) )

for k in range(10):
    ar = make_array(4)
    batches.append( (ar, ar) )


pickle.dump(batches,open(home_dir + "batches_test.pickle", "wb"))