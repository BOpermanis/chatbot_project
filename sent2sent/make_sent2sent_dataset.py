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


qna = []
aqlens = []
inds_conv_with_too_long = []

for i in range(len(convs)):
    conv = convs[i]
    aqlens1 = []
    qna1 = []
    if len(conv) > 1:
        for k in range(len(conv)-1):
            n = len(conv[k]) + len(conv[k+1])
            if n > 100:
                inds_conv_with_too_long.append(i)
                aqlens1, qna1 = [], []
                break
            else:
                aqlens1.append(n)
                qna1.append((conv[k], conv[k + 1]))
        aqlens.extend(aqlens1)
        qna.extend(qna1)



sents = pd.DataFrame.from_dict({"id":[i for i in range(len(qna))],
                                "n": aqlens,
                                "qna": qna})

sents = sents.sort_values("n")
sents["batch_id"] = sents["n"]
sents["qlen"] = sents["qna"].map(lambda x: len(x[0]))
sents["alen"] = sents["qna"].map(lambda x: len(x[1]))

counts = sents[['qlen','alen']].groupby(['qlen', 'alen']).size()
data = np.asarray([list(v) for v in counts.keys()])
weights = counts.values

k = int(round(sum(weights)/2000))

# visualization before clustering
x = data[:, 0]
y = data[:, 1]
plt.scatter(x, y)
# plt.show()

# clustering
#clustering_results = kmeans.kmeans(data,weights,k)

# storing results
#pickle.dump(clustering_results, open("clustering_results.pickle","wb"))

clustering_results = pickle.load(open("clustering_results.pickle", "rb"))

# visualization after clutering
ids, closest_cluster = clustering_results
for k0 in ids:
    indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
    cut = np.take(data, indices, axis=0)
    x, y = cut[:, 0], cut[:, 1]
    v = np.random.rand(3, 1)
    plt.scatter(x, y, c=tuple(v[:, 0]))
    # print("cluster " + str(cl) + " size = " + str(len(clusters[cl])))

plt.show()


####### batching ###########

lookup = dict()
for k0 in ids:
    indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
    cut = np.take(data, indices, axis=0)
    pairs = [( tuple(cut[i,:]), k0 )  for i in range(cut.shape[0])]
    lookup.update(dict(pairs))

sents["batch_id"] = sents[['qlen', 'alen']].apply(lambda row: lookup[tuple(row)], axis=1)



batches = []
pad = revdictionary["_PAD"]

def make_longer(l1,maxl):
    l = list(l1)
    while len(l)<maxl:
        l.append(pad)
    return l


# adding padding
for k0 in ids:
    sents_batch = sents[sents.batch_id == k0]
    mqlen = sents_batch['qlen'].max()
    malen = sents_batch['alen'].max()

    def add_padding(qa):
        q, a = qa
        q.extend([pad] * (mqlen - len(q)))
        a.extend([pad] * (malen - len(a)))
        return q, a

    qna = list(sents_batch["qna"])
    q, a = [], []
    for q1, a1 in qna:
        a.append(make_longer(a1, malen))
        q.append(make_longer(q1, mqlen))

    q = np.asarray(q)
    a = np.asarray(a)
    batches.append((q, a))


pickle.dump(batches,open(home_dir + "batches.pickle", "wb"))

#batches = pickle.load(open(home_dir + "batches.pickle", "rb"))