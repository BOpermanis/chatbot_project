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

# placeholder for batch  :  batch_size X seq_len
input_seq = tf.placeholder(tf.int32, shape=[None, None], name="input_seq")

#  placeholders for decoder
dec_input_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")
dec_output_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")

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

## DECODING
with tf.variable_scope("rnn/decoding"):
    decoder_output, _ = tf.nn.dynamic_rnn(cell=decoder_cell,
                                          inputs=dec_input_seq,
                                          initial_state=thought_vector,
                                          dtype=tf.float32,
                                          time_major=True)

logits = tf.map_fn(inv_embed_fun, decoder_output, dtype=tf.float32, name="logits")

loss_elem = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits, [-1, vocabulary_size]),
                                                           labels=tf.reshape(dec_output_seq_raw, [-1]))
loss = tf.reduce_mean(loss_elem)

with tf.variable_scope("train"):
    rnn_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn")
    rnn_train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, var_list=rnn_scope)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


# importing all models
embedmod.import_model(sess)
# saver = tf.train.Saver(rnn_scope)
# saver.restore(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model2.ckpt")

for epoch in range(50):
    loss_epoch = 0.0
    # loop through batches
    for batch_ind in range(len(batches)):
        qlen = batches[batch_ind][0].shape[0]
        alen = batches[batch_ind][1].shape[0]
        batch_size = 200 if qlen<20 and alen<20 else 100

        n = batches[batch_ind][0].shape[0]
        indices = [random.randint(0, n - 1) for _ in range(batch_size)]
        enc_batch = np.transpose(np.take(batches[batch_ind][0], indices, axis=0)).astype(np.int32)
        dec_batch = np.transpose(np.take(batches[batch_ind][1], indices, axis=0)).astype(np.int32)

        train_dict = {
            input_seq: enc_batch,
            dec_input_seq_raw: np.delete(dec_batch, 0, 0),
            dec_output_seq_raw: np.delete(dec_batch, -1, 0)
        }

        batch_loss, _ = sess.run([loss, rnn_train_step], feed_dict=train_dict)
        loss_epoch += batch_loss

    if epoch % 1 == 0:
        print(str(epoch) + ") loss = " + str(loss_epoch))


saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn'))
if True:
    # saver = tf.train.Saver()
    save_path = saver.save(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model3.ckpt")
    print("Model saved in file: %s" % save_path)

sess.close()