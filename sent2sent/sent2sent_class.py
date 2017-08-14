import tensorflow as tf
import numpy as np

import sys, os, pickle, math


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
with tf.variable_scope("encoding"):
    _ , thought_vector = tf.nn.dynamic_rnn(cell=encoder_cell,
                                          inputs=embed_seq,
                                          # initial_state=zero_state,
                                          dtype=tf.float32,
                                          time_major=True)

## DECODING
with tf.variable_scope("decoding"):
    decoder_output, _ = tf.nn.dynamic_rnn(cell=decoder_cell,
                                          inputs=dec_input_seq,
                                          initial_state=thought_vector,
                                          dtype=tf.float32,
                                          time_major=True)


# logits = tf.transpose(tf.map_fn(inv_embed_fun, decoder_output, dtype=tf.float32, name="logits"), perm=[1, 0, 2])
logits = tf.map_fn(inv_embed_fun, decoder_output, dtype=tf.float32, name="logits")

log1 = tf.reshape(logits, [-1, vocabulary_size])
log2 = tf.reshape(dec_output_seq_raw, [-1])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log1, labels=log2)
# loss_total = tf.reduce_mean(self.loss)





enc_batch = np.transpose(batches[0][0]).astype(np.int32)
dec_batch = np.transpose(batches[0][1]).astype(np.int32)

feed_dict = {
    input_seq: enc_batch,
    dec_input_seq_raw: np.delete(dec_batch, 0, 0),
    dec_output_seq_raw: np.delete(dec_batch, -1, 0)
}


init = tf.global_variables_initializer()
sess.run(init)


r = sess.run(log1, feed_dict=feed_dict)
r1 = sess.run(log2, feed_dict=feed_dict)

print(r)

sess.close()