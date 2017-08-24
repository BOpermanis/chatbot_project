import tensorflow as tf
import numpy as np

import sys, os, pickle, math
import random
import datetime

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

# placeholder for batch  :  batch_size X seq_len
input_seq = tf.placeholder(tf.int32, shape=[None, None], name="input_seq")

#  placeholders for decoder
dec_input_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")
# thought_vector_place = tf.placeholder(tf.float32, shape=[1, embedmod.embedding_size], name="thought_vector_place")
thought_vector_place = (tf.placeholder(tf.float32, shape=[1, embedmod.embedding_size]),
                        tf.placeholder(tf.float32, shape=[1, embedmod.embedding_size]),
                        tf.placeholder(tf.float32, shape=[1, embedmod.embedding_size]))

dec_output_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")

# embedding
def embed_fun(input):
    return tf.nn.embedding_lookup(embedmod.embeddings, input)

# inverse embedding (returns logits not softmax)
def inv_embed_fun(rep):
    return tf.matmul(rep, embedmod.nce_weights)

embed_seq = tf.map_fn(embed_fun, input_seq, dtype=tf.float32, name="embed_seq")
dec_input_seq = tf.map_fn(embed_fun, dec_input_seq_raw, dtype=tf.float32, name="dec_input_seq")

encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)]*3)

decoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)]*3)

## ENCODING
with tf.variable_scope("rnn/encoding"):
    _ , thought_vector = tf.nn.dynamic_rnn(cell=encoder_cell,
                                          inputs=embed_seq,
                                          # initial_state=zero_state,
                                          dtype=tf.float32,
                                          time_major=True)

## DECODING


with tf.variable_scope("rnn/decoding"):


    # initial_tuple = (thought_vector,
    #                  tf.one_hot(dec_input_seq_raw[0, :], vocabulary_size))
    initial_tuple = (thought_vector_place,
                     tf.one_hot(dec_input_seq_raw[0, :], vocabulary_size))

    def fn(prev, input0):
        state, logits0 = prev
        # live_feedback = tf.matmul(tf.nn.softmax(logits0), embedmod.embeddings)

        # half live feedback, half previous word
        # input = tf.divide(live_feedback + input0, 2)
        state1, state_tuple = decoder_cell(inputs=input0, state=state)
        lin_output = inv_embed_fun(state1)

        return state_tuple, lin_output


    # running RNN for sequance lengths
    decoder_output = tf.scan(fn,
                          elems=dec_input_seq,
                          initializer=initial_tuple)

    # formating linear outputs
    logits = decoder_output[1]


w_dist = tf.nn.softmax(logits[0, :, :])
w_ind = tf.argmax(w_dist, axis=1)


rnn_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn")

sess = tf.Session()

# importing all models
# embedmod.import_model(sess)
# saver = tf.train.Saver(rnn_scope)
# saver.restore(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model4.ckpt")
saver = tf.train.Saver()
saver.restore(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model1.ckpt")




# input_sent = "What is up ?".split(" ")
# input_sent = "Tell me a story !".split(" ")
# input_sent = "It is a big city !".split(" ")
# input_sent = "Best wins !".split(" ")
# input_sent = "What is Your name ?".split(" ")
# input_sent = "Are You going to school ?".split(" ")
# input_sent = "Are You a hero ?".split(" ")
input_sent = "How old are You ?".split(" ")

input_sent_coded = [revdictionary[w] for w in ["_START"] + [w1.lower() for w1 in input_sent] + ["_EOS"]]
input_sent_coded = np.asarray(input_sent_coded).reshape([len(input_sent_coded), 1]).astype(np.int32)

# l = [4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7]
# input_sent_coded = np.asarray([l]).astype(np.int32).reshape([len(l),1])


### ENCODING
state = sess.run(thought_vector, feed_dict={
    input_seq: input_sent_coded,
})



ws = []
next_word_ind = revdictionary['_START']
end = revdictionary['_EOS']


### DECODING
for i in range(30):
    # print(i)
    # current_word, current_word_ind, state1 = get_next_word(i, state)

    if i > 0:
        state = state[0]
        state = tuple(s[0, :, :] for s in state)

    # if len(state.shape) > 2:
    #     state = state[0, :, :]

    if isinstance(next_word_ind, np.ndarray):
        next_word_ind = next_word_ind[0]



    next_word_ind, state = sess.run([w_ind,decoder_output], feed_dict={
        thought_vector_place: state, #np.random.random_sample((1,512)).astype(np.float32)
        dec_input_seq_raw: np.asarray([[next_word_ind]])
    })

    if next_word_ind==end:
        break
    ws.append(dictionary[next_word_ind[0]])


print(ws)

sess.close()