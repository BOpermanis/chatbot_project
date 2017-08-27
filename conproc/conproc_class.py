import tensorflow as tf
import numpy as np

import sys, os, pickle, math
import random
import datetime

home_dir = "/media/bruno/data/chatbot_project/"
embedding_dir = home_dir + "embeddings/"
sent2sent_dir = home_dir + "sent2sent/"

sent2sent_logdir = home_dir + "sent2sent/logdir/"


sys.path.append(embedding_dir)
sys.path.append(sent2sent_dir)
from embeddings_class2 import embedding_model
from sent2sent_class5 import sent2sent_model



# first_word, second_word = pickle.load( open(home_dir + "embed_dataset.pickle", "rb" ) )
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )
batches = pickle.load( open(home_dir + "batches_conv.pickle", "rb" ) )
# batches = pickle.load( open(home_dir + "batches_test.pickle", "rb" ) )

vocabulary_size = len(dictionary)


sent2sent = sent2sent_model()
sent2sent.build_rnn()

#placeholders for training
input_seq = []
input_seq.append(tf.placeholder(tf.int32, shape=[None, None], name="input_seq1"))
input_seq.append(tf.placeholder(tf.int32, shape=[None, None], name="input_seq2"))
input_seq.append(tf.placeholder(tf.int32, shape=[None, None], name="input_seq3"))
input_seq.append(tf.placeholder(tf.int32, shape=[None, None], name="input_seq4"))
input_seq.append(tf.placeholder(tf.int32, shape=[None, None], name="input_seq5"))

dec_input_seq_raw = []
dec_input_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw1"))
dec_input_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw2"))
dec_input_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw3"))
dec_input_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw4"))
dec_input_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw5"))

dec_output_seq_raw = []
dec_output_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw1"))
dec_output_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw2"))
dec_output_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw3"))
dec_output_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw4"))
dec_output_seq_raw.append(tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw5"))


thought_vectors = []
for i in range(5):
    thought_vectors.append(sent2sent.encoding_fun(input_seq[i]))

## input sequence for the context RNN (time X batch_size X vec_dim)
ht = tf.stack([tf.concat(thought_vector, axis=1) for thought_vector in thought_vectors])

con_cell = tf.contrib.rnn.GRUCell(num_units=sent2sent.embedmod.embedding_size*sent2sent.rnn_layers)

with tf.variable_scope("conproc"):
    con_out, _ = tf.nn.dynamic_rnn(cell=con_cell,
                                          inputs=ht,
                                          # initial_state=zero_state,
                                          dtype=tf.float32,
                                          time_major=True)

# unstacking for replying
s0 = tf.unstack(con_out, axis=0)

# tf.split(s0, num_split=, split_dim=1)
# tf.split(split_dim=1, num_split=3, value=s0)


i = 0
states = []
logits = []
last_states = []
loss_elems = []
losses = []

for i in range(5):
    state = tuple(tf.split(value=s0[i], num_or_size_splits=3, axis=1))

    logit, last_state = sent2sent.decoding_fun(dec_input_seq_raw[i], state)

    loss_elem = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(
        logit, [-1, vocabulary_size]),
        labels=tf.reshape(dec_output_seq_raw[i], [-1]))
    loss = tf.reduce_mean(loss_elem)

    states.append(state)
    logits.append(logit)
    last_states.append(last_states)
    loss_elems.append(loss_elem)
    losses.append(loss)

abs_loss = sum(losses)
train_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conproc")
train_step = tf.train.AdamOptimizer(0.0001).minimize(abs_loss, var_list=train_col)


batch_ind = 0

batch = batches[batch_ind]

train_dict = dict()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i_round in range(5):
    i_q = i_round * 2
    i_a = i_round * 2 + 1

    max_seq_len = max([batch[i].shape[1] for i in range(len(batch))])

    # batch_size = 200 if max_seq_len < 30 else 50
    batch_size = 1

    n = batches[batch_ind][0].shape[0]
    indices = [random.randint(0, n - 1) for _ in range(batch_size)]
    enc_batch = np.transpose(np.take(batch[i_q], indices, axis=0)).astype(np.int32)
    dec_batch = np.transpose(np.take(batch[i_a], indices, axis=0)).astype(np.int32)

    train_dict[input_seq[i_round]] = enc_batch
    print(np.delete(dec_batch, -1, 0).shape)
    train_dict[dec_input_seq_raw[i_round]] = np.delete(dec_batch, -1, 0)
    train_dict[dec_output_seq_raw[i_round]] = np.delete(dec_batch, 0, 0)



# batch_loss, _ = sess.run([abs_loss, train_step], feed_dict=train_dict)
# loss_epoch.append(batch_loss)

v = sess.run(thought_vectors, feed_dict=train_dict)
type(v)

sess.close()