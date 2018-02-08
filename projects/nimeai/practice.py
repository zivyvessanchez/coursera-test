import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import math
import collections
import random

from tempfile import gettempdir
from util import clean_str
from scraper import scrape
from sklearn.manifold import TSNE


# Globals
data = ''
data_index = 0
vocab_size = 0
reverse_dictionary = ''
itr = 0

def build_dataset(words, n_words):
    global data
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
        
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def generate_batch(batch_size, num_skips, skip_window):
    """Generate a training batch for the skip-gram model."""
    global data_index, data
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]

    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)

        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
            
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def train(initial_embeddings=None):
    global vocab_size, reverse_dictionary, itr
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    num_sampled = 64      # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            if initial_embeddings is not None:
                embeddings = initial_embeddings
            else:
                embeddings = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the Noise Contrastive Estimation loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocab_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocab_size))

        # Construct Stochastic Gradient Descent optimizer with learning rate 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

    # Begin training.
    num_steps = 10001

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
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
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary.get(nearest[k], reverse_dictionary[0])
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

    try:
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels, 'tsne{}.png'.format(itr))

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)

    return final_embeddings

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def main(num_articles=10):
    global vocab_size, reverse_dictionary, itr
    rand_word = ''
    embedding_cache = None
    vocab_size = 0
    vocab_cache = None
    
    if (len(sys.argv) < 2):
        print("Usage: {0} <wikipedia article title> [<output filename>]".format(
            os.path.basename(__file__)))
        return

    for i in range(num_articles):
        itr = i
        print(itr)
        while vocab_size < 10:
            # Scrape wiki page
            rand_word = vocab[random.randint(0,len(vocab)-1)] if i > 0 else sys.argv[1]
            print("Scraping " + rand_word)
            if(len(sys.argv) >= 3):
                text = scrape(rand_word, sys.argv[2])
            else:
                text = scrape(rand_word)

            # Preprocess text before lemmatization
            text = clean_str(text)

            # Get vocab of page
            vocab = text.split()
            vocab_size = len(vocab)

            # Early out if starting article is invalid, else combine cache and current
            if vocab_size < 10 and i == 0:
                print("Initial article is invalid! Exiting...")
                return
            elif i > 0:
                for word in vocab:
                    if word not in vocab_cache:
                        vocab_cache.append(word)
                vocab = vocab_cache
                vocab_size = len(vocab)

        # Build dataset using vocab
        data, count, dictionary, reverse_dictionary = \
              build_dataset(vocab,vocab_size)

        # Generate batches for training
        batch, labels = generate_batch(batch_size=8,
                                       num_skips=2,
                                       skip_window=1)
        for j in range(8):
            print(batch[i], reverse_dictionary[batch[j]],
                  '->', labels[j, 0], reverse_dictionary[labels[j, 0]])

        if embedding_cache is not None:
            embeddings = train(embedding_cache)
        else:
            embeddings = train()

        embedding_cache = embeddings
        if i == 0:
            vocab_cache = vocab
        else:
            for word in vocab:
                if word not in vocab_cache:
                    vocab_cache.append(word)
        vocab_size = 0  # Reset

if __name__ == "__main__":
    main(10)
