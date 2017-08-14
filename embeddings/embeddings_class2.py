import numpy as np
import tensorflow as tf
import math
import random

home_dir = "/media/bruno/data/chatbot_project"
embedding_dir = home_dir + "/embeddings/embeddings_checkpoint/"

class embedding_model:

    def __init__(self, vocabulary_size, embedding_size=512):
        """Initializes model, builds models graph"""
        ## some hyperparameters
        self.num_sampled = 64
        self.is_session_started = False
        self.is_inverse_embeddings_initialize = False
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embedding_map")

        self.nce_weights = tf.Variable(
            tf.truncated_normal([embedding_size,vocabulary_size],
                                stddev=1.0 / math.sqrt(embedding_size)), name="nce_weights")

        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

        ## placeholders for data
        self.train_inputs = tf.placeholder(tf.int32, shape=[None], name="inputs")
        self.train_labels = tf.placeholder(tf.int32, shape=[None], name="outputs")
        self.reps = tf.placeholder(tf.float32, shape=[None, embedding_size], name="representation")

        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs, name="embeddings")

        # mapping representation->word
        self.logits = tf.matmul(self.embed, self.nce_weights) + self.nce_biases

        # Softmax cross-entropy
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.train_labels)
        self.loss_total = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.GradientDescentOptimizer(1.0, name="optimizer").minimize(self.loss_total)


    def train(self, first_word, second_word, num_steps=50001, batch_size=10000, test_size=2000, period_show=100, save=True):
        """ trains the model, generates session in the process"""
        data_set_size = len(first_word)
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        print('Initialized')

        average_train_loss = 0
        average_test_loss = 0
        for step in range(num_steps):

            batch_inds = [random.randint(0, data_set_size - 1) for _ in range(batch_size)]
            batch_inputs = np.asarray([first_word[j] for j in batch_inds])  # .reshape([batch_size, 1])
            batch_labels = np.asarray([second_word[j] for j in batch_inds]).reshape([batch_size])

            #training
            _, train_loss = session.run([self.optimizer,self.loss_total], feed_dict={self.train_inputs: batch_inputs,
                                                  self.train_labels: batch_labels}
                        )

            average_train_loss += train_loss
            if step % period_show == 0:
                print(str(step) + ") Test_error = " + str(float(average_test_loss)/period_show) + ", Training_error = " + str(float(average_train_loss)/period_show))
                average_test_loss = 0
                average_train_loss = 0

        if save:
            saver = tf.train.Saver()
            save_path = saver.save(session, embedding_dir + "embeddings_model.ckpt")
            print("Model saved in file: %s" % save_path)
        session.close()

    def import_session(self, beeing_integrated=False):
        """Restores model"""
        self.is_session_started = True
        saver = tf.train.Saver()
        session = tf.Session()
        saver.restore(session, embedding_dir + "embeddings_model.ckpt")
        if beeing_integrated:
            return session
        else:
            self.session = session

    def predict_embedding(self, x):
        """Computes representation of the word"""
        if not self.is_session_started:
            self.import_session()
            self.is_session_started = True
        if not isinstance(x, np.ndarray):
            if not isinstance(x, list):
                x = [x]
            x = np.asarray(x)

        pred = self.session.run(self.embed,feed_dict={self.train_inputs: x})
        return pred


    def initialize_inverse_embeddings(self, beeing_integrated=False):
        if not self.is_session_started:
            self.import_session(beeing_integrated=beeing_integrated)
            self.is_session_started = True

        # adding tensors for
        self.softmax = tf.nn.softmax(tf.matmul(self.reps, self.nce_weights))
        self.inverse_representation = tf.arg_max(self.softmax, dimension=1)
        self.is_inverse_embeddings_initialize = True


    def predict_inverse_embedding(self, r):
        """Computes word distribution from given representation"""

        if not self.is_inverse_embeddings_initialize:
            self.initialize_inverse_embeddings()

        invrep = self.session.run(self.inverse_representation, feed_dict={self.reps: r})
        return invrep

