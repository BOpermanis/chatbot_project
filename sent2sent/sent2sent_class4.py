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
from embeddings_class2 import embedding_model


first_word, second_word = pickle.load( open(home_dir + "embed_dataset.pickle", "rb" ) )
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )
batches = pickle.load( open(home_dir + "batches.pickle", "rb" ) )
# batches = pickle.load( open(home_dir + "batches_test.pickle", "rb" ) )


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

encoder_cell1 = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)
encoder_cell2 = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)
encoder_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell1,encoder_cell2])

decoder_cell1 = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)
decoder_cell2 = tf.contrib.rnn.GRUCell(num_units=embedmod.embedding_size)
decoder_cell = tf.nn.rnn_cell.MultiRNNCell([decoder_cell1, decoder_cell2])


## ENCODING
with tf.variable_scope("rnn/encoding"):
    _ , thought_vector = tf.nn.dynamic_rnn(cell=encoder_cell,
                                          inputs=embed_seq,
                                          # initial_state=zero_state,
                                          dtype=tf.float32,
                                          time_major=True)

## DECODING


with tf.variable_scope("rnn/decoding"):


    initial_tuple = (thought_vector,
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


loss_elem = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits, [-1, vocabulary_size]),
                                                           labels=tf.reshape(dec_output_seq_raw, [-1]))
loss = tf.reduce_mean(loss_elem)

with tf.variable_scope("train"):
    # rnn_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn")
    # rnn_train_step = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=rnn_scope)
    rnn_train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)




init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


# importing all models
# saver = tf.train.Saver(rnn_scope)
# saver.restore(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model.ckpt")
# embedmod.import_model(sess)


saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn')+
                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embeddings'))
saver.restore(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model.ckpt")


# saves graph
# writer = tf.summary.FileWriter(sent2sent_logdir)
# writer.add_graph(sess.graph)

# batch_ind = 0
# batch_size = 1
# n = batches[batch_ind][0].shape[0]
# indices = [random.randint(0, n - 1) for _ in range(batch_size)]
# enc_batch = np.transpose(np.take(batches[batch_ind][0], indices, axis=0)).astype(np.int32)
# dec_batch = np.transpose(np.take(batches[batch_ind][1], indices, axis=0)).astype(np.int32)
# train_dict = {
#     input_seq: enc_batch,
#     dec_input_seq_raw: np.delete(dec_batch, 0, 0),
#     dec_output_seq_raw: np.delete(dec_batch, -1, 0)
# }
# sess.run(dec_output_seq_raw, feed_dict=train_dict)
# sess.run(dec_input_seq_raw, feed_dict=train_dict)
# sess.run(input_seq, feed_dict=train_dict)
# sess.close()


try:
    for epoch in range(20):
        loss_epoch = []
        # loop through batches
        a = datetime.datetime.now()
        for batch_ind in range(len(batches)):
            qlen = batches[batch_ind][0].shape[0]
            alen = batches[batch_ind][1].shape[0]
            batch_size = 200 if qlen < 20 and alen < 20 else 100

            n = batches[batch_ind][0].shape[0]
            indices = [random.randint(0, n - 1) for _ in range(batch_size)]
            enc_batch = np.transpose(np.take(batches[batch_ind][0], indices, axis=0)).astype(np.int32)
            dec_batch = np.transpose(np.take(batches[batch_ind][1], indices, axis=0)).astype(np.int32)
            train_dict = {
                input_seq: enc_batch,
                dec_input_seq_raw: np.delete(dec_batch, -1, 0),
                dec_output_seq_raw: np.delete(dec_batch, 0, 0)
            }

            batch_loss, _ = sess.run([loss, rnn_train_step], feed_dict=train_dict)
            loss_epoch.append(batch_loss)

        b = datetime.datetime.now()

        if epoch % 1 == 0:
            print(str(epoch) + ") loss = " + str(np.mean(loss_epoch)) + ", time spent = " + str(b - a))
except KeyboardInterrupt:
    pass



# saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn'))
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn')+
                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embeddings'))
if True:
    # saver = tf.train.Saver()
    save_path = saver.save(sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model.ckpt")
    print("Model saved in file: %s" % save_path)

sess.close()