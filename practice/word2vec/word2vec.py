import os
import numpy as np
import tensorflow as tf
import pickle as pkl
from scipy.sparse import csc    # Used for one-hot encoding

WINDOW_SIZE = 2         # Skipgram window size for target-context pairs
EMBEDDING_DIM = 300     # Word2vec embedding size
BATCH_SIZE = 10
i_max = 1               # Hardcoded for unit testing
FILE_DATAPAIR = 'data_pair' # Temp file to store data pairs in

def unpickle(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass

def to_one_hot(index, vocab_size):
    temp = np.zeros(vocab_size, dtype='int32')
    temp[index] = 1
    return temp

def main(filename=''):
    corpus = ''
    
    if filename == '':
        filename = 'wiki.en.text'

    with open(filename, 'r', encoding='utf-8') as f:
        i = 0

        # Concatenate lines into a single string
        for line in f:
            if(i_max >= 10 and i % int(i_max/10) == 0):
                print("Line {0}\ttotal len: {1},\tadded {2},\tnew total: {3}"\
                      .format(i, len(corpus), len(line), len(corpus) + len(line)))
            corpus += line
            i += 1
            if i >= i_max:
                break
        
        # Transform to lower case
        corpus = corpus.lower()

        # Create word dictionary and indices
        words = []
        word2int = {}
        int2word = {}

        for word in corpus.split():
            words.append(word)
        words = set(words)  # Retain only unique words
        vocab_size = len(words) # Tells the amount of unique words
        print("Vocab size is {0}".format(vocab_size))

        # Create indices
        for i, word in enumerate(words):
            word2int[word] = i
            int2word[i] = word

        # Create skipgram pairs ([target, contexts])
        print("Creating skip gram pairs")
        data = []
        corpus_array = corpus.split()
        for i, target in enumerate(corpus_array):
            for context in corpus_array[max(i - WINDOW_SIZE, 0) :
                                             min(i + WINDOW_SIZE, len(corpus_array))+1]:
                if context != target:
                    data.append([target, context])
        print("Number of skipgram target-context pairs is {0}".format(len(data)))
        # Save data into binary file to re-access later
        with open(FILE_DATAPAIR,'wb') as g:
            for i in range(len(data)):
                pkl.dump(data[i],g)

        # Delete original data pairs to save space
        del data
                
        # Create one-hot representations using saved binary file
        print("Creating one-hot pairs")
        x_train = []    # input word
        y_train = []    # output word
        with open(FILE_DATAPAIR,'rb') as g:
            for data in unpickle(g):
                x_train.append(to_one_hot(word2int[data[0]], vocab_size))
                y_train.append(to_one_hot(word2int[data[1]], vocab_size))
        os.remove(FILE_DATAPAIR)
            
        # Convert to numpy arrays
        x_train = np.asarray(x_train, dtype='int32')
        y_train = np.asarray(y_train, dtype='int32')
        print("x_train: {0}, y_train: {1}".format(
            x_train.shape, y_train.shape))
        
        # Create Tensorflow model
        
        '''
        x = tf.placeholder(tf.float32, shape=(None, vocab_size))        # input layer / words
        y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))  # output labels / words

        # Neural network 1
        W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM])) # weights
        b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))             # biases
        h1 = tf.add(tf.matmul(x, W1), b1)                               # hidden layer

        # Neural network 2
        W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size])) # weights
        b2 = tf.Variable(tf.random_normal([vocab_size]))                # biases
        h2 = tf.add(tf.matmul(h1, W2), b2)                              # hidden layer

        # Softmax
        out = tf.nn.softmax(h2)                                         # softmax layer
        
        # Train; set session to use CPU
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Loss function
        loss = tf.reduce_mean(
            -tf.reduce_sum(
                y_label * tf.log(out + 1e-30),
                reduction_indices=[1]
            )
        )
        # Train step
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        n_iters = 1000
        # Train in iterations
        x_feed = []
        y_feed = []
        for i in range(len(x_train)):
            if (i+BATCH_SIZE) >= len(x_train):
                x_feed = x_train[i:(len(x_train)-1)]
                y_feed = y_train[i:(len(y_train)-1)]
            else:
                x_feed = x_train[i:i+BATCH_SIZE]
                y_feed = y_train[i:i+BATCH_SIZE]
            
            for itr in range(n_iters):
                feed_dict = {x:x_feed, y_label: y_feed}
                sess.run(train, feed_dict=feed_dict)
                if n_iters >= 100 and itr % int(n_iters/100) == 0:
                    print("iter {0}, loss is: {1}".format(
                        itr,
                        sess.run(loss, feed_dict={x:x_train, y_label:y_train})
                    ))
        '''
    
if __name__ == '__main__':
    main()
