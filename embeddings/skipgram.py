import numpy as np
import tensorflow as tf
import os, pickle, math
import random

home_dir = "/media/bruno/data/chatbot_project/"
os.chdir("/media/bruno/data/chatbot_project/embeddings")


first_word, second_word = pickle.load( open(home_dir + "embed_dataset.pickle", "rb" ) )
dictionary = pickle.load( open(home_dir + "dictionary.pickle", "rb" ) )
revdictionary = pickle.load( open(home_dir + "revdictionary.pickle", "rb" ) )

vocabulary_size = len(dictionary)
data_set_size = len(first_word)


## some hyperparameters
embedding_size = 512
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    ## Skipgram model parameters

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embedding_map")

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)), name="nce_weights")

    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

    ## placeholders for data
    train_inputs = tf.placeholder(tf.int32, shape=[None],name="inputs")
    train_labels = tf.placeholder(tf.int32, shape=[None, 1],name="outputs")

    embed = tf.nn.embedding_lookup(embeddings, train_inputs,name="embeddings")

    # Compute the NCE loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size),name="loss")

    optimizer = tf.train.GradientDescentOptimizer(1.0,name="optimizer").minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = tf.div(embeddings,norm)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 50001
batch_size = 100000


with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    batch_inds = [ random.randint(0,data_set_size-1) for _ in range(batch_size)]



    batch_inputs = np.asarray([first_word[j] for j in batch_inds])#.reshape([batch_size, 1])
    batch_labels = np.asarray([second_word[j] for j in batch_inds]).reshape([batch_size, 1])
    print(batch_labels.shape)

    average_loss = 0
    for step in range(num_steps):
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        # if step % 10000 == 0:
        #     sim = similarity.eval()
        #     for i in range(valid_size):
        #         valid_word = revdictionary[valid_examples[i]]
        #         top_k = 8  # number of nearest neighbors
        #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        #         log_str = 'Nearest to %s:' % valid_word
        #         for k in range(top_k):
        #             close_word = reverse_dictionary[nearest[k]]
        #             log_str = '%s %s,' % (log_str, close_word)
        #             print(log_str)
    final_embeddings = normalized_embeddings.eval()
