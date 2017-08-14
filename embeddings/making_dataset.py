import os
import pickle
from collections import Counter
import numpy as np
import collections

home_dir = "/media/bruno/data/chatbot_project/"
os.chdir("/media/bruno/data/chatbot_project/data_sets/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus")


## importing previous results
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )
tab = pickle.load( open(home_dir + "tab.pickle", "rb" ) )

n_dic = len(dictionary)

# dropping unnesecary symbols
tab["for_word_embed"] = tab["V6"].map(lambda x: "|".join([w for w in x if w not in {"_START","_END","_PAD"} ]))



## reorganizing data
def conc(x):
    return '||'.join(x)
phrases = tab[["for_word_embed"]].apply(conc,axis=0)[0]

marks1 = [".", ",", "?", "!"]
for m in marks1:
    m1 = "|" + m + "|"
    phrases = phrases.replace(m1,"||")
phrases = phrases.replace("|||","||")

phrases = phrases.split("||")

keys = revdictionary.keys()
phrases = list(map(lambda x: [revdictionary[x0] for x0 in x.split("|") if x0 in keys],phrases))


first_word = []
second_word = []
for f in phrases:
    for i in range(len(f)-1):
        first_word.append(f[i])
        second_word.append(f[i])

# first_word = np.asarray(first_word)
# second_word = np.asarray(second_word)
## saving result
pickle.dump((first_word, second_word),open(home_dir + "embed_dataset.pickle", "wb"))