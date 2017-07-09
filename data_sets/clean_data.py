import os
import pandas as pd
import numpy as np
from functools import reduce
from string import ascii_lowercase as alphabet
import pickle
from collections import Counter

home_dir = "/media/bruno/data/chatbot_project/"

os.chdir("/media/bruno/data/chatbot_project/data_sets/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus")

tab = pd.read_csv("movie_lines_comma.csv",sep="|",names = ["V1","V2","V3","V4","V5"])

marks = [".", ",", "?", "!", "-","'"]
alphabet = alphabet + "".join(marks)


############################## Cleaning Corpus ##########################


#b = re.compile(r"<.{1}>")

special_words = ["_PAD", "_START", "_END"]

def startsEndsSpec(x):
    if len(x) > 1:
        if x[0] not in alphabet:
            x = x[1:]
        if x[-1] not in alphabet:
            x = x[:-1]
    return x

def punct(x):

    ## some special substitutions
    x = x.replace("'s"," is")
    x = x.replace("...", ",")
    x = x.replace("'d", " had")
    x = x.replace("n't", " not")
    x = x.replace("'m", " am")
    x = x.replace("in'", "ing")
    x = x.replace("-", " - ")
    x = x.replace("\"", " \" ")
    x = x.replace("\t", " ")
    x = x.replace("*", " \" ")
    x = x.replace("/", " ")

    for p in marks:
        x = x.replace(p," " + p + " ")

    x = startsEndsSpec(x)

    return x


tab["V5"] = tab["V5"].map(punct)


tab["V6"] = tab["V5"].map(lambda x: [special_words[1]] + [i.replace(" ","").lower() for i in x.split(" ") if i not in [""," "]] + [special_words[2]]  )

ns = tab["V6"].map(lambda x: len(x)).tolist()



lens = Counter(ns)


sum(ns)

Counter(lens)

length_groups = []


############################## making a dictionary ##########################


#tab["V8"] = tab["V6"].map(lambda x: set(x))
tab["V8"] = tab["V6"].map(lambda x: '|'.join(x))

def conc(x):
    return '|'.join(x)

v = tab[["V8"]].apply(conc,axis=0)[0]
v = list(set(v.split("|")))


# v = reduce(lambda x,y: x.union(y),tab["V8"].tolist())
# v = list(set(v))

v_dict = [(i,v[i]) for i in range(len(v))]
v_revdict = [(v[i],i) for i in range(len(v))]

dictionary = dict(v_dict)
revdictionary = dict(v_revdict)



pickle.dump(dictionary,open(home_dir + "dictionary.pickle", "wb"))
pickle.dump(revdictionary,open(home_dir + "revdictionary.pickle", "wb"))

# dictionary = pd.DataFrame({"words":v})
# dictionary = dictionary.select(lambda x: x!="")
# dictionary["key"] = [i for i in range(dictionary.shape[0])]
# dictionary.to_csv("/media/bruno/data/chatbot_project/dictionary.csv")
# def oneWordCleaning(w):
#     #if the first or last symbol is not a letter
#     #print(w)
#     if len(w)>1:
#         if w[0] not in alphabet:
#             w = w[1:]
#         if w[-1] not in alphabet:
#             w = w[:-1]
#     return w
# dictionary["words"] = dictionary["words"].map(oneWordCleaning)



############################# Replacing words #######################

keys = revdictionary.keys()

tab["V7"] = tab["V6"].map(lambda l: [revdictionary[w] for w in l if w in keys])

#pickle.dump(tab,open(home_dir + "tab.pickle", "wb"))


############################# Replacing words #######################

tab["V1"] = tab["V1"].map(lambda x: x.replace(" ",""))
con_dict = dict(zip(tab["V1"].tolist(),tab["V7"].tolist()))

conv = pd.read_csv("movie_conversations_comma.csv",sep="|",names = ["V1","V2","V3","V4"])

conv["V4"] = conv["V4"].map(lambda x: eval(x))

keys = con_dict.keys()

dataset = conv["V4"].map(lambda x: [con_dict[w] for w in x if w in keys])

#conv["V4"].select(lambda x: len([w for w in x if w==" "])>0)

pickle.dump(dataset,open(home_dir + "dataset.pickle", "wb"))