import os
import pickle
from collections import Counter
import numpy as np
import collections
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

home_dir = "/media/bruno/data/chatbot_project/"
os.chdir("/media/bruno/data/chatbot_project/data_sets/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus")


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
                                "n":aqlens,
                                "qna":qna})

sents = sents.sort_values("n")
sents["batch_id"] = sents["n"]
sents["alen"] = sents["qna"].map(lambda x: len(x[0]))
sents["qlen"] = sents["qna"].map(lambda x: len(x[1]))


#visualizing distribution of the sentence lengths
# n, bins, patches = plt.hist(aqlens, 50, normed=1, facecolor='green', alpha=0.75)
# plt.grid(True)
# plt.show()


ns = Counter(sents["n"])


## apskataas _PAD relatiivo skaitu batch_indeksu kopai
# pienjem, ka visas indeksu reprezenteejoshaas grupas eksistee
def get_rel_pad_count(k_list):

    if not isinstance(k_list, list):
        k_list = [k_list]

    max_alen = []
    max_qlen = []
    count = []
    word_count = []

    for k in k_list:
        sents_k = sents[sents.batch_id==k]
        count.append( len(sents_k) )
        max_alen.append( max(sents_k["alen"]) )
        max_qlen.append( max(sents_k["qlen"]) )
        word_count.append( sum(sents_k["n"]) )

    abs_max_alen = max(max_alen)
    abs_max_qlen = max(max_qlen)

    all_word_count = []
    for u in range(len(k_list)):
        sa = (max_alen[u] + (abs_max_alen - max_alen[u])) * count[u]
        sq = (max_qlen[u] + (abs_max_qlen - max_qlen[u])) * count[u]
        all_word_count.append(sa + sq)

    return sum(word_count)/sum(all_word_count)



tresh = 0.6

batch_inds = list(set(sents["batch_id"]))
something_to_do = True
while something_to_do:
    print(len(batch_inds))
    something_to_do = False
    result_list = []
    for i in range(len(batch_inds) - 1):
        i1 = batch_inds[i]
        i2 = batch_inds[i + 1]
        result_list.append(get_rel_pad_count([i1, i2]))
        if result_list[-1] < tresh:
            print("sup")
            sents["batch_id"] = sents["batch_id"].map(lambda x: i1 if x == i2 else x)
            something_to_do = True
    print(min(result_list), max(result_list))
    batch_inds = list(set(sents["batch_id"]))

# n, bins, patches = plt.hist(sents["batch_id"], 50, normed=1, facecolor='green', alpha=0.75)
# plt.grid(True)
# plt.show()
