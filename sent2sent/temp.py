import tensorflow as tf
import numpy as np

import sys, os, pickle, math
import random

home_dir = "/media/bruno/data/chatbot_project/"
embedding_dir = home_dir + "embeddings/"
sent2sent_dir = home_dir + "sent2sent/"

sys.path.append(embedding_dir)
from embeddings_class2 import embedding_model

first_word, second_word = pickle.load( open(home_dir + "embed_dataset.pickle", "rb" ) )
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )
batches = pickle.load( open(home_dir + "batches.pickle", "rb" ) )

vocabulary_size = len(dictionary)
data_set_size = len(first_word)


vocabulary_size = len(dictionary)
data_set_size = len(first_word)


embedmod = embedding_model(vocabulary_size=vocabulary_size)
sess = embedmod.import_session(beeing_integrated=True)

# placeholders
input_seq = tf.placeholder(tf.int32, shape=[None, None], name="input_seq")
thought_vector_place = tf.placeholder(tf.float32, shape=[1, embedmod.embedding_size], name="thought_vector_place")
one_word = tf.placeholder(tf.int32, shape=[1, 1], name="one_word")


# embedding
def embed_fun(input):
    return tf.nn.embedding_lookup(embedmod.embeddings, input)

# inverse embedding (returns logits not softmax)
def inv_embed_fun(rep):
    return tf.matmul(rep, embedmod.nce_weights)

embed_seq = tf.map_fn(embed_fun, input_seq, dtype=tf.float32, name="embed_seq")

encoder_cell = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)
decoder_cell = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)

## ENCODING
with tf.variable_scope("rnn/encoding"):
    _ , thought_vector = tf.nn.dynamic_rnn(cell=encoder_cell,
                                          inputs=embed_seq,
                                          # initial_state=zero_state,
                                          dtype=tf.float32,
                                          time_major=True)


one_word_enc = embed_fun(one_word)

with tf.variable_scope("rnn/decoding"):
    _ , logit = tf.nn.dynamic_rnn(cell=decoder_cell,
                                          inputs=one_word_enc,
                                          initial_state=thought_vector_place,
                                          dtype=tf.float32,
                                          time_major=True)


logit_dec = inv_embed_fun(logit)

w_dist = tf.nn.softmax(logit_dec)
w_ind = tf.argmax(w_dist, axis=1)

saver = tf.train.Saver()
saver.restore(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model.ckpt")

input_sent = ["What", "is", "Your", "name", "?"]

input_sent_coded = [revdictionary[w.lower()] for w in input_sent]

input_sent_coded = np.asarray(input_sent_coded).reshape([len(input_sent_coded),1]).astype(np.int32)



def get_next_word(current_word,state):
    next_word_ind = sess.run(w_ind, feed_dict={
        thought_vector_place: state,
        one_word: np.asarray([[current_word]])
    })
    next_word = dictionary[next_word_ind[0]]
    return next_word, next_word_ind[0]



### actual ENCODING
thought_vector_val = sess.run(thought_vector, feed_dict={
    input_seq: input_sent_coded,
})
state = thought_vector_val

current_word_ind = revdictionary["_START"]


### actual DECODING
output_sent = []
for i in range(100):
    current_word, current_word_ind = get_next_word(current_word_ind, state)
    output_sent.append(current_word)
    if current_word==revdictionary["_END"]:
        break


print(output_sent)


sess.close()