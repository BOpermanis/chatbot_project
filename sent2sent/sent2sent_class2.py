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

class sent2sent_model:
    def __init__(self):
        pass

    def initalize_embeddings(self):
        self.embedmod = embedding_model(vocabulary_size=vocabulary_size)
        self.sess = self.embedmod.import_session(beeing_integrated=True)


    def build_sent2sent_graph(self):
        # placeholder for batch  :  batch_size X seq_len
        self.input_seq = tf.placeholder(tf.int32, shape=[None, None], name="input_seq")

        #  placeholders for decoder
        self.dec_input_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")
        self.dec_output_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")

        # embedding
        def embed_fun(input):
            return tf.nn.embedding_lookup(self.embedmod.embeddings, input)

        # inverse embedding (returns logits not softmax)
        def inv_embed_fun(rep):
            return tf.matmul(rep, self.embedmod.nce_weights)

        self.embed_seq = tf.map_fn(embed_fun, self.input_seq, dtype=tf.float32, name="embed_seq")
        self.dec_input_seq = tf.map_fn(embed_fun, self.dec_input_seq_raw, dtype=tf.float32, name="dec_input_seq")

        self.encoder_cell = tf.contrib.rnn.GRUCell(num_units=self.embedmod.embedding_size)
        self.decoder_cell = tf.contrib.rnn.GRUCell(num_units=self.embedmod.embedding_size)

        ## ENCODING
        with tf.variable_scope("rnn/encoding"):
            _, self.thought_vector = tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                                  inputs=self.embed_seq,
                                                  # initial_state=zero_state,
                                                  dtype=tf.float32,
                                                  time_major=True)

        ## DECODING
        with tf.variable_scope("rnn/decoding"):
            self.decoder_output, _ = tf.nn.dynamic_rnn(cell=self.decoder_cell,
                                                  inputs=self.dec_input_seq,
                                                  initial_state=self.thought_vector,
                                                  dtype=tf.float32,
                                                  time_major=True)

        self.logits = tf.map_fn(inv_embed_fun, self.decoder_output, dtype=tf.float32, name="logits")

        # for inferrence
        self.w_dist = tf.nn.softmax(self.logits[0, :, :])
        self.w_ind = tf.argmax(self.w_dist, axis=1)

    def train(self, batches, num_epoch=1000, restore=True, save=True):

        saver = tf.train.Saver()
        if restore and os.path.isfile(sent2sent_dir + "sent2sent_checkpoint/sent2sent_model.ckpt"):
            saver.restore(self.sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model.ckpt")

        self.loss_elem = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(self.logits, [-1, vocabulary_size]),
                                                                   labels=tf.reshape(self.dec_output_seq_raw, [-1]))
        self.loss = tf.reduce_mean(self.loss_elem)

        self.rnn_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnn")
        # rnn_train_step = tf.train.AdamOptimizer(0.01).minimize(loss, var_list=rnn_scope)
        self.rnn_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss, var_list=self.rnn_scope)

        init = tf.global_variables_initializer()

        self.sess.run(init)

        for epoch in range(num_epoch):
            loss_epoch = []
            # loop through batches
            for batch_ind in range(len(batches)):
                qlen = batches[batch_ind][0].shape[0]
                alen = batches[batch_ind][1].shape[0]
                batch_size = 200 if qlen < 20 and alen < 20 else 100

                n = batches[batch_ind][0].shape[0]
                indices = [random.randint(0, n - 1) for _ in range(batch_size)]
                enc_batch = np.transpose(np.take(batches[batch_ind][0], indices, axis=0)).astype(np.int32)
                dec_batch = np.transpose(np.take(batches[batch_ind][1], indices, axis=0)).astype(np.int32)

                train_dict = {
                    self.input_seq: enc_batch,
                    self.dec_input_seq_raw: np.delete(dec_batch, 0, 0),
                    self.dec_output_seq_raw: np.delete(dec_batch, -1, 0)
                }

                batch_loss, _ = self.sess.run([self.loss, self.rnn_train_step], feed_dict=train_dict)
                loss_epoch.append(batch_loss)

            loss_epoch = np.mean(loss_epoch)
            if epoch % 1 == 0:
                print(str(epoch) + ") loss = " + str(loss_epoch))

        if save:
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, sent2sent_dir + "sent2sent_checkpoint/sent2sent_model.ckpt")
            print("Model saved in file: %s" % save_path)


    def talk(self,input_sent):
        if not isinstance(input_sent):
            input_sent = input_sent.split(" ")

        input_sent_coded = [revdictionary[w] for w in ["_START"] + [w1.lower() for w1 in input_sent] + ["_EOS"]]
        input_sent_coded = np.asarray(input_sent_coded).reshape([len(input_sent_coded), 1]).astype(np.int32)

        def get_next_word(current_word, state):
            next_word_ind, state = self.sess.run([self.w_ind, self.decoder_output], feed_dict={
                self.thought_vector_place: state,
                self.dec_input_seq_raw: np.asarray([[current_word]])
            })
            next_word = dictionary[next_word_ind[0]]
            return next_word, next_word_ind[0], state[0, :, :]


