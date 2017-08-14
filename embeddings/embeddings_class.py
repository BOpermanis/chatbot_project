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
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embedding_map")

        self.nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)), name="nce_weights")

        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

        ## placeholders for data
        self.train_inputs = tf.placeholder(tf.int32, shape=[None], name="inputs")
        self.train_labels = tf.placeholder(tf.int32, shape=[None, 1], name="outputs")
        self.train_reps = tf.placeholder(tf.float32, shape=[None, embedding_size], name="representation")


        # mapping representation->word
        self.logits = tf.matmul(self.nce_weights,tf.transpose(self.train_reps)) + tf.reshape(self.nce_biases,[vocabulary_size,1])
        self.inverse_representation = tf.arg_max(self.logits,dimension=0)

        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs, name="embeddings")

        # Compute the NCE loss, using a sample of the negative labels each time.
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                           biases=self.nce_biases,
                           labels=self.train_labels,
                           inputs=self.embed,
                           num_sampled=self.num_sampled,
                           num_classes=vocabulary_size), name="loss")

        self.optimizer = tf.train.GradientDescentOptimizer(1.0, name="optimizer").minimize(self.loss)

        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = tf.div(self.embeddings, self.norm, name="normalized_embeddings")

    def train(self, first_word, second_word, num_steps=50001, batch_size=100000, test_size=2000, period_show=1000, save=True):
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
            batch_labels = np.asarray([second_word[j] for j in batch_inds]).reshape([batch_size, 1])
            r = np.asarray([0 for _ in range(self.embedding_size)]).reshape([1, self.embedding_size])

            #training
            _, train_loss = session.run([self.optimizer,self.loss], feed_dict={self.train_inputs: batch_inputs,
                                                  self.train_labels: batch_labels,
                                                  self.train_reps: r}
                        )

            average_train_loss += train_loss

            if step % period_show == 0:
                print(str(step) + ") Test_error = " + str(float(average_test_loss)/period_show) + ", Training_error = " + str(float(average_train_loss)/period_show))
                average_test_loss = 0
                average_train_loss = 0

        #final_embeddings = normalized_embeddings.eval()

        if save:
            saver = tf.train.Saver()
            save_path = saver.save(session, embedding_dir + "embeddings_model.ckpt")
            print("Model saved in file: %s" % save_path)

        self.session = session

    def import_session(self, beeing_integrated=False):
        """Restores model"""
        saver = tf.train.Saver()
        session = tf.Session()
        saver.restore(session, embedding_dir + "embeddings_model.ckpt")
        if beeing_integrated:
            return session
        else:
            self.session = session

    def predict_embedding(self, x):
        """Computes representation of the word"""
        if not isinstance(x, np.ndarray):
            if not isinstance(x, list):
                x = [x]
            x = np.asarray(x)

        nx = x.shape[0]
        y = np.asarray([0 for _ in range(nx)]).reshape([nx, 1])
        r = np.asarray([0 for _ in range(self.embedding_size)]).reshape([1,self.embedding_size])
        pred = self.session.run(self.embed,
                                  feed_dict={self.train_inputs: x,
                                             self.train_labels: y,
                                             self.train_reps: r}
                                  )
        return pred

    def predict_inverse_embedding(self, r):
        """Computes word distribution from given representation"""
        x = np.asarray([0 for _ in range(self.vocabulary_size)])
        y = np.asarray([0 for _ in range(self.vocabulary_size)]).reshape([self.vocabulary_size, 1])

        invrep = self.session.run(self.inverse_representation,
                                feed_dict={self.train_inputs: x,
                                           self.train_labels: y,
                                           self.train_reps: r}
                                )
        return invrep

