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

dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )
batches = pickle.load( open(home_dir + "batches.pickle", "rb" ) )
# batches = pickle.load( open(home_dir + "batches_test.pickle", "rb" ) )


vocabulary_size = len(dictionary)


class sent2sent_model:

    def __init__(self, rnn_layers=3):
        """
        Initializes model
        """
        self.embedmod = embedding_model(vocabulary_size=vocabulary_size)
        self.rnn_layers = rnn_layers
        self.session_initialized = False
        self.rnn_graph_built = False

    # embedding mapping
    def embed_fun(self, input):
        return tf.nn.embedding_lookup(self.embedmod.embeddings, input)

    # inverse embedding (returns logits not softmax)
    def inv_embed_fun(self, rep):
        return tf.matmul(rep, self.embedmod.nce_weights)


    def encoding_fun(self,input_seq):

        embed_seq = tf.map_fn(self.embed_fun, input_seq, dtype=tf.float32)

        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.GRUCell(num_units=self.embedmod.embedding_size)] * self.rnn_layers)

        with tf.variable_scope("sent2sent"):
            with tf.variable_scope("rnn/encoding") as scope:
                # tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
                _, thought_vector = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                           inputs=embed_seq,
                                                           # initial_state=zero_state,
                                                           dtype=tf.float32,
                                                           time_major=True)
        return thought_vector

    def decoding_fun(self, input_seq, thought_vector):

        dec_input_seq = tf.map_fn(self.embed_fun, input_seq, dtype=tf.float32)

        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.GRUCell(num_units=self.embedmod.embedding_size)] * self.rnn_layers)

        with tf.variable_scope("sent2sent"):
            with tf.variable_scope("rnn/decoding") as scope:
                scope.reuse_variables()
                decoder_output, last_state = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                           inputs=dec_input_seq,
                                                           initial_state=thought_vector,
                                                           dtype=tf.float32,
                                                           time_major=True)
        logits = tf.map_fn(self.inv_embed_fun, decoder_output, dtype=tf.float32, name="logits")

        return logits, last_state


    def build_rnn(self, train=True):
        """
        Builds RNN graphs
        """
        self.rnn_graph_built = True

        with tf.variable_scope("sent2sent"):
            # placeholder for batch  :  seq_len X batch_size
            self.input_seq = tf.placeholder(tf.int32, shape=[None, None], name="input_seq")
            self.dec_input_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")
            self.dec_output_seq_raw = tf.placeholder(tf.int32, shape=[None, None], name="dec_output_seq_raw")

            self.embed_seq = tf.map_fn(self.embed_fun, self.input_seq, dtype=tf.float32, name="embed_seq")
            self.dec_input_seq = tf.map_fn(self.embed_fun, self.dec_input_seq_raw, dtype=tf.float32, name="dec_input_seq")

            ## RNN cells
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.embedmod.embedding_size)] * self.rnn_layers)

            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.embedmod.embedding_size)] * self.rnn_layers)

            ## ENCODING
            with tf.variable_scope("rnn/encoding"):
                _, self.thought_vector = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                           inputs=self.embed_seq,
                                                           # initial_state=zero_state,
                                                           dtype=tf.float32,
                                                           time_major=True)

            if train:
                thought_vector = self.thought_vector
            else:
                self.thought_vector_place = tuple([tf.placeholder(tf.float32, shape=[1, self.embedmod.embedding_size]) for _ in range(self.rnn_layers)])
                thought_vector = self.thought_vector_place

            ## DECODING
            with tf.variable_scope("rnn/decoding"):
                self.decoder_output , self.last_state = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                           inputs=self.dec_input_seq,
                                                           initial_state=thought_vector,
                                                           dtype=tf.float32,
                                                           time_major=True)

            self.logits = tf.map_fn(self.inv_embed_fun, self.decoder_output, dtype=tf.float32, name="logits")

            if train:
                loss_elem = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(self.logits, [-1, vocabulary_size]),
                    labels=tf.reshape(self.dec_output_seq_raw, [-1]))
                self.loss = tf.reduce_mean(loss_elem)
            else:
                w_dist = tf.nn.softmax(self.logits[0, :, :])
                self.w_ind = tf.argmax(w_dist, axis=1)


    def train(self, learning_rate, max_epochs=50, train_scope=["embeddings","sent2sent"],
                    restore_scope=["embeddings", "sent2sent"],
                    save_scope=["embeddings", "sent2sent"],
                    ckpt_path=sent2sent_dir+"sent2sent_checkpoint/sent2sent_model.ckpt"):
        """
        runs training
        """

        if not self.rnn_graph_built:
            self.rnn_graph_built = True
            self.build_rnn()

        # making variables scopes:
        train_col = []
        for scope in train_scope:
            train_col = train_col + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        restore_col = []
        for scope in restore_scope:
            restore_col = restore_col + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        save_col = []
        for scope in save_scope:
            save_col = save_col + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # training variable sscope
        with tf.variable_scope("train"):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=train_col)


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # saves graph
        # writer = tf.summary.FileWriter(sent2sent_logdir)
        # writer.add_graph(sess.graph)

        if len(restore_scope) > 0:
            saver = tf.train.Saver(var_list=restore_col)
            saver.restore(sess, ckpt_path)
        else:
            self.embedmod.import_model(sess)

        try:
            for epoch in range(max_epochs):
                loss_epoch = []
                # loop through batches
                a = datetime.datetime.now()
                for batch_ind in range(len(batches)):
                    qlen = batches[batch_ind][0].shape[0]
                    alen = batches[batch_ind][1].shape[0]
                    batch_size = 300 if qlen < 30 and alen < 30 else 50

                    n = batches[batch_ind][0].shape[0]
                    indices = [random.randint(0, n - 1) for _ in range(batch_size)]
                    enc_batch = np.transpose(np.take(batches[batch_ind][0], indices, axis=0)).astype(np.int32)
                    dec_batch = np.transpose(np.take(batches[batch_ind][1], indices, axis=0)).astype(np.int32)
                    train_dict = {
                        self.input_seq: enc_batch,
                        self.dec_input_seq_raw: np.delete(dec_batch, -1, 0),
                        self.dec_output_seq_raw: np.delete(dec_batch, 0, 0)
                    }

                    batch_loss, _ = sess.run([self.loss, train_step], feed_dict=train_dict)
                    loss_epoch.append(batch_loss)

                b = datetime.datetime.now()

                if epoch % 1 == 0:
                    print(str(epoch) + ") loss = " + str(np.mean(loss_epoch)) + ", time spent = " + str(b - a))
        except KeyboardInterrupt:
            pass

        # Saving results
        if len(save_scope) > 0:
            saver = tf.train.Saver(var_list=save_col)
            save_path = saver.save(sess, ckpt_path)
            print("Model saved in file: %s" % save_path)

        sess.close()


    def initialize_chatbot(self,ckpt_path=sent2sent_dir+"sent2sent_checkpoint/sent2sent_model.ckpt"):
        if not self.rnn_graph_built:
            self.rnn_graph_built = True
            self.build_rnn(train=False)

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_path)


    def answer_something(self,input_sent):

        if not self.rnn_graph_built:
            model.initialize_chatbot()

        if isinstance(input_sent, str):
            input_sent = input_sent.split(" ")

        input_sent_coded = [revdictionary[w] for w in ["_START"] + [w1.lower() for w1 in input_sent] + ["_EOS"]]
        input_sent_coded = np.asarray(input_sent_coded).reshape([len(input_sent_coded), 1]).astype(np.int32)

        ### ENCODING
        state = self.sess.run(self.thought_vector, feed_dict={
            self.input_seq: input_sent_coded,
        })

        ws = []
        next_word_ind = revdictionary['_START']
        end = revdictionary['_EOS']

        ### DECODING
        for i in range(50):

            if isinstance(next_word_ind, np.ndarray):
                next_word_ind = next_word_ind[0]

            next_word_ind, state = self.sess.run([self.w_ind, self.last_state], feed_dict={
                self.thought_vector_place: state,
                self.dec_input_seq_raw: np.asarray([[next_word_ind]])
            })
            if next_word_ind == end:
                break
            ws.append(dictionary[next_word_ind[0]])

        return ws



if __name__ == "__main__":

    model = sent2sent_model()

    # model.train(learning_rate=0.000001, max_epochs=15) #, save_scope=[])


    qs = ["Are You smart ?",
          "What is up ?",
          "Tell me a story !",
          "It is a big city !",
          "Best wins !",
          "What is Your name ?",
          "Are You going to school ?",
          "Are You smart ?",
          "How old are You ?",
          "How are You ?",
          "Do You have a girlfriend ?",
          "God is everywhere !",
          "Ninja style !",
          "shut up !",
          "Any advice ?"]

    for q in qs:
        a = model.answer_something(q)
        print(q + "  -  " + " ".join(a))