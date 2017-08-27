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
dictionary = pickle.load(open(home_dir + "dictionary.pickle", "rb"))
revdictionary = pickle.load(open(home_dir + "revdictionary.pickle", "rb"))
tab = pickle.load(open(home_dir + "tab.pickle", "rb"))
convs = pickle.load(open(home_dir + "dataset.pickle", "rb"))

convs = [con for con in convs if len(con) <= 10]

# ns = [len(i) for i in convs]
# plt.hist(ns)
# plt.show()

for i in range(len(convs)):
    con = convs[i]
    for k in range(10-len(con)):
        convs[i].append([revdictionary["_START"], revdictionary["_END"]])


def get_sizes(con):
    return ",".join([str(len(sent)) for sent in con])

cons = pd.DataFrame.from_dict({"sizes":[get_sizes(con) for con in convs],
                               "conv":convs})
counts = cons.groupby(["sizes"]).size()

data = np.asarray([[int(c) for c in v.split(",")] for v in counts.keys()])
weights = counts.values

k = int(round(sum(weights)/400))

# clustering
clustering_results = kmeans.kmeans_parallel(data, weights, k, B=6)

# storing results
pickle.dump(clustering_results, open("clustering_results_conv.pickle","wb"))

# clustering_results = pickle.load(open("clustering_results_conv.pickle", "rb"))


### batching
ids, closest_cluster = clustering_results

lookup = dict()
for k0 in ids:
    indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
    cut = np.take(data, indices, axis=0)
    pairs = [(','.join([str(el) for el in cut[i, :]]), k0) for i in range(cut.shape[0])]
    lookup.update(dict(pairs))

cons["batch_id"] = cons['sizes'].map(lambda x: lookup[x])




batches = []
pad = revdictionary["_PAD"]

def make_longer(l1,maxl):
    l = list(l1)
    while len(l)<maxl:
        l.append(pad)
    return l



for k0 in ids:
    cons_batch = cons[cons.batch_id == k0]

    round_lens = tuple(zip(*list(cons_batch['sizes'].map(lambda x: [int(i) for i in x.split(",")]))))

    maxinround = tuple(max(v) for v in round_lens)

    conv = []
    for i_sent in range(10):
        sents = list(cons_batch['conv'].map(lambda x: x[i_sent]))
        mlen = maxinround[i_sent]
        conv.append(np.asarray([make_longer(x, mlen) for x in sents]).astype(np.int32))
    batches.append(tuple(conv))




pickle.dump(batches, open(home_dir + "batches_conv.pickle", "wb"))

# batches = pickle.load(open(home_dir + "batches_conv.pickle", "rb"))


