########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
import nose
import nltk

nltk.download('punkt')
from nltk import word_tokenize, PorterStemmer


def tokenize_and_normalize(file_name):
    """
    this function takes in a path to a song, reads the song file,
    tokenizes it into words, then stems and lowercases these words.

    INPUT:
    file_name - a path to a file as a string
    OUTPUT:
    normalized_song - a song represented as a list of stems.
    """
    ps = PorterStemmer()

    # YOUR CODE HERE
    song = open(file_name, 'r').read()
    song = word_tokenize(song.lower())
    normalized_song = [None] * len(song)
    for i, word in enumerate(song):
        normalized_song[i] = ps.stem(word)

    return normalized_song


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
dummy_song_path = "ex3/dummy_song.txt"
tokenize_and_normalize(dummy_song_path)

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

import numpy as np
from collections import Counter


def create_term_doc_matrix(songs_normalized):
    """
    this function takes in a list of songs and returns a term-document matrix as a numpy array.
    the rows are word stems, the columns are songs.
    the rows should be sorted alphabetically.
    INPUT:
    songs_normalized - a list of songs represented as a list of stems (a list of lists)
    OUTPUT:
    matrix - a matrix where columns are songs and rows are stems,
        the cells of the matrix contain stem frequencies in a song,
        the words for rows are sorted alphabetically.

    sorted_vocab - a list of all the words used in all songs (the rows of our matrix).
        the words should be strings sorted alphabetically
    """

    # YOUR CODE HERE
    vocab = []

    for song in songs_normalized:
        for word in song:
            if word not in vocab:
                vocab.append(word)
    sorted_vocab = sorted(vocab)
    matrix = np.zeros((len(sorted_vocab), len(songs_normalized)))

    for i, song in enumerate(songs_normalized):
        for word in song:
            matrix[sorted_vocab.index(word), i] += 1

    return matrix, sorted_vocab


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

dummy_songs = [['la', 'la', 'la', 'oh', ',', 'woo', "uuuuuh"],
               ['oh', 'la', 'la', 'la', "tarara", 'tadada', 'blaaa', 'blaaa', '!', '!', '!']]

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(create_term_doc_matrix(dummy_songs)[0].shape, (9, 2))
# check that the vocabulary is a list
assert_equal(type(create_term_doc_matrix(dummy_songs)[1]), list)
# check that the vocabulary is a list of strings
assert_equal(type(create_term_doc_matrix(dummy_songs)[1][0]), str)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the vocabulary is sorted properly
assert_equal(create_term_doc_matrix(dummy_songs)[1],
             ['!', ',', 'blaaa', 'la', 'oh', 'tadada', 'tarara', 'uuuuuh', 'woo'])
# check that the matrix has the right values in the right places
correct_td_dummy_matrix = np.array([[0., 3.],
                                    [1., 0.],
                                    [0., 2.],
                                    [3., 3.],
                                    [1., 1.],
                                    [0., 1.],
                                    [0., 1.],
                                    [1., 0.],
                                    [1., 0.]])
assert_array_equal(create_term_doc_matrix(dummy_songs)[0], correct_td_dummy_matrix)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def tf_idf(td_matrix):
    """
    this function takes in a term-document matrix as a numpy array,
    and weights the scores with the tf-idf algorithm described above

    INPUT:
    td_matrix - a numpy array where columns are songs and
        rows are word frequencies in a song
    OUTPUT:
    tf_idf_matrix - a numpy array where columns are songs and
        rows are word tf-idf values in a song
    idf_vector - a vector of idf values for words in the collection. the shape is (vocabulary-size, 1).
        this vector will be used to weight new query documents
    """
    # YOUR CODE HERE
    df = np.copy(td_matrix)
    df[df > 0] = 1
    df = np.sum(df, axis=1)

    idf_vector = np.expand_dims(np.log10(len(td_matrix.T) / df), axis=1)
    tf_idf_matrix = td_matrix * idf_vector
    return tf_idf_matrix, idf_vector


from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import assert_equal

dummy_songs = [['la', 'la', 'la', 'la', 'oh', ',', 'woo', "uuuuuh"],
               ['oh', 'la', 'la', 'la', "tarara", 'tadada', 'bla', 'bla', 'bla', 'bla', 'bla', '!']]

sorted_dummy_vocab = ['!', ',', 'bla', 'la', 'oh', 'tadada', 'tarara', 'uuuuuh', 'woo']
correct_td_dummy_matrix = np.array([[0., 1.],
                                    [1., 0.],
                                    [0., 5.],
                                    [4., 3.],
                                    [1., 1.],
                                    [0., 1.],
                                    [0., 1.],
                                    [1., 0.],
                                    [1., 0.]])

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(tf_idf(correct_td_dummy_matrix)[0].shape, (9, 2))
# check the shape of the idf vector
assert_equal(tf_idf(correct_td_dummy_matrix)[1].shape, (9, 1))

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the matrix has the right values in the right places
# pay attention to the "la" row: its values are zeros now. make sure you understand why.

correct_tf_idf_dummy_matrix = np.array([[0., 0.30103],
                                        [0.30103, 0.],
                                        [0., 1.50514998],
                                        [0., 0.],
                                        [0., 0.],
                                        [0., 0.30103],
                                        [0., 0.30103],
                                        [0.30103, 0.],
                                        [0.30103, 0.]])

correct_idf_dummy_vector = np.array([[0.30103],
                                     [0.30103],
                                     [0.30103],
                                     [0.],
                                     [0.],
                                     [0.30103],
                                     [0.30103],
                                     [0.30103],
                                     [0.30103]])

assert_allclose(tf_idf(correct_td_dummy_matrix)[0], correct_tf_idf_dummy_matrix)
assert_allclose(tf_idf(correct_td_dummy_matrix)[1], correct_idf_dummy_vector)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def ppmi(td_matrix, query_matrix=None):
    """
    this function takes in a term-document matrix as a numpy array,
    and weights the scores with the PPMI scheme described above

    INPUT:
    td_matrix - a numpy array where columns are collection songs and
        rows are word frequencies in a song
    query_matrix - a numpy array where columns are query songs and
        rows are word frequencies in a song
    OUTPUT:
    ppmi_matrix - a numpy array where columns are songs and
        rows are ppmi word values in a song
    """
    if query_matrix is None:
        # YOUR CODE HERE
        ctd = td_matrix
        cd = np.expand_dims(np.sum(td_matrix, axis=0), axis=0)
    else:
        # YOUR CODE HERE
        ctd = query_matrix
        cd = np.expand_dims(np.sum(query_matrix, axis=0), axis=0)
    n = np.sum(td_matrix)
    ct = np.expand_dims(np.sum(td_matrix, axis=1), axis=1)
    ppmi_matrix = np.log2((ctd * n) / (ct @ cd))
    ppmi_matrix[ppmi_matrix < 0] = 0
    return ppmi_matrix


from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import assert_equal

dummy_songs = [['la', 'la', 'la', 'la', 'oh', ',', 'woo', "uuuuuh"],
               ['oh', 'la', 'la', 'la', "tarara", 'tadada', 'bla', 'bla', 'bla', 'bla', 'bla', '!']]

sorted_dummy_vocab = ['!', ',', 'bla', 'la', 'oh', 'tadada', 'tarara', 'uuuuuh', 'woo']
correct_td_dummy_matrix = np.array([[0., 1.],
                                    [1., 0.],
                                    [0., 5.],
                                    [4., 3.],
                                    [1., 1.],
                                    [0., 1.],
                                    [0., 1.],
                                    [1., 0.],
                                    [1., 0.]])

td_dummy_query = np.array([[0.],
                           [1.],
                           [8.],
                           [1.],
                           [0.],
                           [0.],
                           [1.],
                           [2.],
                           [4.]])

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(ppmi(correct_td_dummy_matrix).shape, (9, 2))
assert_equal(ppmi(correct_td_dummy_matrix, td_dummy_query).shape, (9, 1))

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the matrix has the right values in the right places
# pay attention to the row of "woo" and the row of "tarara" why are the ppmi values different for diiferent documents?

correct_ppmi_dummy_matrix = np.array([[0., 0.73696559],
                                      [1.32192809, 0.],
                                      [0., 0.73696559],
                                      [0.51457317, 0.],
                                      [0.32192809, 0.],
                                      [0., 0.73696559],
                                      [0., 0.73696559],
                                      [1.32192809, 0.],
                                      [1.32192809, 0.]])

assert_allclose(ppmi(correct_td_dummy_matrix), correct_ppmi_dummy_matrix)

correct_ppmi_dummy_query = np.array([[0.],
                                     [0.23446525],
                                     [0.91253716],
                                     [0.],
                                     [0.],
                                     [0.],
                                     [0.23446525],
                                     [1.23446525],
                                     [2.23446525]])

assert_allclose(ppmi(correct_td_dummy_matrix, td_dummy_query), correct_ppmi_dummy_query)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def lsi(matrix, d):
    """
    this function takes in a term-document matrix as a numpy array,
    the values can be both row frequencies and weighted values (tf_idf, ppmi)
    performs and returns the VT matrix.

    INPUT:
    matrix - a numpy array where columns are songs and
        rows are words
    d - a number of features we will be reducing our matrix to.
    OUTPUT:
    U_d - an [m x d] matrix, where m is the number of word dimensions in the original matrix,
        and d is a number of features we want to keep.
    """
    # Singular-value decomposition
    U, s, VT = np.linalg.svd(matrix)

    # YOUR CODE HERE
    U_d = U[:, :d]
    return U_d


from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import assert_equal

dummy_songs = [['la', 'la', 'la', 'la', 'oh', ',', 'woo', "uuuuuh"],
               ['oh', 'la', 'la', 'la', "tarara", 'tadada', 'bla', 'bla', 'bla', 'bla', 'bla', '!']]

sorted_dummy_vocab = ['!', ',', 'bla', 'la', 'oh', 'tadada', 'tarara', 'uuuuuh', 'woo']
correct_td_dummy_matrix = np.array([[0., 1.],
                                    [1., 0.],
                                    [0., 5.],
                                    [4., 3.],
                                    [1., 1.],
                                    [0., 1.],
                                    [0., 1.],
                                    [1., 0.],
                                    [1., 0.]])

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(lsi(correct_td_dummy_matrix, 1).shape, (9, 1))

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
dummy_query_vector = np.arange(9).reshape(9, 1)

assert_allclose(lsi(correct_td_dummy_matrix, 1), np.array([[0.13232178],
                                                           [0.06933039],
                                                           [0.6616089],
                                                           [0.67428688],
                                                           [0.20165217],
                                                           [0.13232178],
                                                           [0.13232178],
                                                           [0.06933039],
                                                           [0.06933039]]))

assert_allclose(lsi(correct_td_dummy_matrix, 1).T.dot(correct_td_dummy_matrix), np.array([[3.10679086, 5.92952265]]))
assert_allclose(lsi(correct_td_dummy_matrix, 1).T.dot(dummy_query_vector), np.array([[6.71751287]]))


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def create_term_doc_matrix_queries(queries, vocab):
    """
    this function takes in a list of query songs and the sorted vocabulary of our collection,
    and returns a term-document matrix as a numpy array.
    the rows are word stems, the columns are songs.
    the rows should be sorted according to the vocabulary.
    INPUT:
    queries - a list of songs represented as a list of stems (a list of lists)
    vocab - an alphabetically sorted list of all the words used in all songs of the collection (the rows of our matrix).
    OUTPUT:
    matrix - a matrix where columns are songs and rows are stems,
        the cells of the matrix contain stem frequencies in a song,
        the words for rows are sorted alphabetically.
    """

    # YOUR CODE HERE
    matrix = np.zeros((len(vocab), len(queries)))
    for i, song in enumerate(queries):
        for word in song:
            if word in vocab:
                matrix[vocab.index(word), i] += 1
            else:
                continue
    return matrix


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

sorted_dummy_vocab = ['!', ',', 'bla', 'la', 'oh', 'tadada', 'tarara', 'uuuuuh', 'woo']
new_dummy_songs = [['la', 'oh', ',', 'pada', "uuuuuh"],
                   ['toot', 'toot']]

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(create_term_doc_matrix_queries(new_dummy_songs, sorted_dummy_vocab).shape, (9, 2))

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the matrix has the right values in the right places
correct_td_new_dummy_matrix = np.array([[0., 3.],
                                        [1., 0.],
                                        [0., 2.],
                                        [3., 3.],
                                        [1., 1.],
                                        [0., 1.],
                                        [0., 1.],
                                        [1., 0.],
                                        [1., 0.]])

assert_array_equal(create_term_doc_matrix_queries(new_dummy_songs, sorted_dummy_vocab), np.array([[0., 0.],
                                                                                                  [1., 0.],
                                                                                                  [0., 0.],
                                                                                                  [1., 0.],
                                                                                                  [1., 0.],
                                                                                                  [0., 0.],
                                                                                                  [0., 0.],
                                                                                                  [1., 0.],
                                                                                                  [0., 0.]]))


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def cosine(doc1, doc2):
    """
    this function takes in two document vectors and computes cosine similarity between them
        when any of the vectors contains only zeros, their similarity is unknown (no common words)
        your function should output -inf

    INPUT:
    doc1 - the first document vector
    doc2 - the second document vector
    OUTPUT:
    score - cosine similarity
    """

    # YOUR CODE HERE
    score = np.dot(doc1, doc2) / np.sqrt(np.sum(doc1 ** 2) * np.sum(doc2 ** 2))
    return score


from numpy.testing import assert_almost_equal, assert_array_equal, assert_allclose
from nose.tools import assert_equal

dummy_x = np.arange(3)
dummy_y = np.arange(3, 6)

assert_almost_equal(cosine(dummy_x, dummy_y), 0.8854377448471461, 3)


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def closest_n_documents(matrix_collection, matrix_queries, n):
    """
    this function takes in original document collection, new document collection,
    computes cosine similarity between documents in old and new collection,
    and outputs the list of n-closest documents to each new song.
    INPUT:
    matrix_collection - a term-document matrix of collection songs
    matrix_queries - a term-document matrix of query songs
    n - number of best documents
    OUTPUT:
    closest_docs - a list of lists of length matrix_queries.shape[1]
        each element is a list of n idices of documents in matrix_collection that were closest to the query
    """
    # YOUR CODE HERE
    best_cosines = []
    for doc1 in matrix_queries.T:
        res = np.array([cosine(doc1, doc2) for doc2 in matrix_collection.T])
        best_cosines.append(list(res.argsort()[-n:][::-1]))
    return best_cosines


from numpy.testing import assert_almost_equal, assert_array_equal, assert_allclose
from nose.tools import assert_equal

dummy_collection = np.arange(15).reshape((5, 3))
dummy_query = np.arange(5).reshape((5, 1))

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the length of the list
assert_equal(len(closest_n_documents(dummy_collection, dummy_query, 1)), 1)
# check the len of the first element
assert_equal(len(closest_n_documents(dummy_collection, dummy_query, 1)[0]), 1)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
closest_vector_id = closest_n_documents(dummy_collection, dummy_query, 1)[0][0]
assert_equal(closest_vector_id, 0)

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# the order of models is: raw counts, tf-idf, PPMI, LSI raw counts, LSI tf-idf, LSI PPMI

# type in the answer as a list of 6 values
average_precision = None
# type in the answer as a list of 6 values
average_recall = None
# type in the answer as a list of 6 values
average_accuracy = None
# type in the answer as a list of 6 values
average_error = None
# type in the answer as a list of 6 values
average_f_measure = None
# type in the answer as a list of 6 values

# Remember to remove the raise NotImplementedError line:
# YOUR CODE HERE

model_names = ["raw counts", "tf-idf", "PPMI", "LSI raw counts", "LSI tf-idf", "LSI PPMI"]
for i, model_name in enumerate(model_names):
    print(model_name)
    print("average previcision:", average_precision[i])
    print("average_accuracy:", average_accuracy[i])
    print("average_recall:", average_recall[i])
    print("average_error:", average_error[i])
    print("average_f_measure:", average_f_measure[i])
    print("============")
