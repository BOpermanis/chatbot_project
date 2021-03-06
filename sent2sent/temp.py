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
# sess = embedmod.import_session(beeing_integrated=True)

# placeholders
input_seq = tf.placeholder(tf.int32, shape=[None, None], name="input_seq")
thought_vector_place = tf.placeholder(tf.float32, shape=[1, embedmod.embedding_size], name="thought_vector_place")

#  placeholders for decoder
dec_input_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")

# embedding
def embed_fun(input):
    return tf.nn.embedding_lookup(embedmod.embeddings, input)

# inverse embedding (returns logits not softmax)
def inv_embed_fun(rep):
    return tf.matmul(rep, embedmod.nce_weights)

embed_seq = tf.map_fn(embed_fun, input_seq, dtype=tf.float32, name="embed_seq")
dec_input_seq = tf.map_fn(embed_fun, dec_input_seq_raw, dtype=tf.float32, name="dec_input_seq")

encoder_cell = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)
decoder_cell = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)

## ENCODING
with tf.variable_scope("rnn/encoding"):
    _ , thought_vector = tf.nn.dynamic_rnn(cell=encoder_cell,
                                          inputs=embed_seq,
                                          # initial_state=zero_state,
                                          dtype=tf.float32,
                                          time_major=True)
print(thought_vector)
print(thought_vector_place)

## DECODING
with tf.variable_scope("rnn/decoding"):
    decoder_output, _ = tf.nn.dynamic_rnn(cell=decoder_cell,
                                          inputs=dec_input_seq,
                                          initial_state=thought_vector_place,
                                          dtype=tf.float32,
                                          time_major=True)


logits = tf.map_fn(inv_embed_fun, decoder_output, dtype=tf.float32, name="logits")


w_dist = tf.nn.softmax(logits[0,:,:])
w_ind = tf.argmax(w_dist, axis=1)


sess = tf.Session()
rnn_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn")
sent2sent_saver = tf.train.Saver(rnn_scope)
sent2sent_saver.restore(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model2.ckpt")

embedmod.import_model(sess)

# input_sent = ["What", "is", "Your", "name", "?"]
input_sent = "You are my friend".split(" ")


input_sent_coded = [revdictionary[w] for w in ["_START"] + [w1.lower() for w1 in input_sent] + ["_EOS"]]
input_sent_coded = np.asarray(input_sent_coded).reshape([len(input_sent_coded), 1]).astype(np.int32)



def get_next_word(current_word,state):
    next_word_ind, state = sess.run([w_ind, decoder_output], feed_dict={
        thought_vector_place: state,
        dec_input_seq_raw: np.asarray([[current_word]])
    })
    next_word = dictionary[next_word_ind[0]]
    return next_word, next_word_ind[0], state[0, :, :]



### actual ENCODING
state = sess.run(thought_vector, feed_dict={
    input_seq: input_sent_coded,
})

current_word_ind = revdictionary["_START"]


### actual DECODING
# output_sent = []
# for i in range(20):
#     current_word, current_word_ind, state1 = get_next_word(current_word_ind, state)
#     print(np.sum(state))
#     output_sent.append(current_word)
#     state = np.copy(state1)
#     if current_word==revdictionary["_END"]:
#         break


ws = []
next_word_ind = revdictionary["_START"]

for w in range(20):
    # current_word, current_word_ind, state1 = get_next_word(i, state)
    if len(state.shape)>2:
        state = state[0,:,:]

    if isinstance(next_word_ind,np.ndarray):
        next_word_ind = next_word_ind[0]

    next_word_ind, state = sess.run([w_ind, decoder_output], feed_dict={
        thought_vector_place: state, #np.random.random_sample((1,512)).astype(np.float32)
        dec_input_seq_raw: np.asarray([[next_word_ind]])
    })
    print(next_word_ind.shape)
    ws.append(dictionary[next_word_ind[0]])


print(ws)
sess.close()