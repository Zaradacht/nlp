#!/usr/bin/env python
# coding: utf-8

# ELEC-E5550 - Statistical Natural Language Processing
# # SET 4: 
# 
# # Released: 27.02.2020
# # Deadline: 11.03.2020 at midnight

# After completing this assignment, you'll be able to perform sequence tagging tasks such as Part-of-Speech tagging with Hiddent Markov Models. Moreover, you'll learn to analyze the performance of your model.
# 
# KEYWORDS:
# 
# * Part-of-Speech (POS) tagging 
# * Hidden Markov Models (HMMs)
# * Viterbi algorithm
# * Confusion Matrix

# ### Data
# The GUM corpus (https://corpling.uis.georgetown.edu/gum/) annotated with Universal Dependencies POS tags (https://universaldependencies.org/u/pos/).
# 
# * */home/contentis/aalto/nlp/coursedata/POS/tags_vocab.txt* - vocabulary of UD tags sorted in alphabetical order
# * */home/contentis/aalto/nlp/coursedata/POS/train.txt* - corpus for training (4219 sentences, 12181 tokens)
# * */home/contentis/aalto/nlp/coursedata/POS/words_vocab.txt* - vocabulary of the training corpus sorted in alphabetical order
# * */home/contentis/aalto/nlp/coursedata/POS/test_words.txt* - unlabelled test corpus (1055 sentences, 5262 tokens)
# * */home/contentis/aalto/nlp/coursedata/POS/test_tagss.txt* - correct tags for test corpus

# **Part-of-speech tagging** (**POS tagging**) is the process of annotating words in an input sequence with their corresponding part-of-speech labels. Word's **part-of-speech** gives us more information about the word itself and about its neighboring words (nouns are preceded by determiners and adjectives). For instance, we can use POS tags as features for Named Entitity recognition task: Proper Nouns like names are usually these entities. Moreover, word's **part-of-speech** provides us with an understanding on how to pronounce this word: cOntent if it is a noun and contEnt if it is an adjective. This helps in such tasks as speech recognition and synthesis.
# 
# In this assignment we're going to create two algorithms for assigning POS tags. They are both based on statistics collected from a corpus annotated with POS tags by humans. First, we will be assigning words the most frequent tag it has been seen with. This will works as a **baseline** we will be trying to beat. Second, we will create an **HMM** model and compare it to our baseline.
# 
# The POS tagging algorithm is judged by how **accurate** it is.

# ## TASK 1
# ## Read the data
# 
# In this assignment we are lucky to get already pre-processed text. However, be careful with pre-processing your text for POS tagging: your pre-processing steps should match the pre-processing of the corpus you're collecting statistics from. For example, *POS* tag employed in Penn Treebank Project (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) is only used for *'s*.
# 
# ### Read vocabularies
# ## 1.1
# The files for word and tag vocabularies contain each vocabulary member on its own line. Write a function to collect these vocabularies as lists.

# In[19]:


def read_vocab(file_name):
    """
    this function takes in a path to a vocabulary file, reads the file,
    and returns a vocabulary list
    
    
    INPUT: file_name - a path to a file as a string
    OUTPUT: vocab - a list of strings. the elements of the list should have the same order as in the file
    """

    # YOUR CODE HERE
    vocab = open(file_name, 'r').read().split()

    return vocab


# In[20]:

from numpy.testing import assert_array_equal
from nose.tools import assert_equal

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
dummy_vocab_path = "/home/contentis/aalto/nlp/coursedata/POS/dummy_vocab.txt"

# check that the output of the function is a list
assert_equal(type(read_vocab(dummy_vocab_path)), list)
# check that it's a list of strings
assert_equal(type(read_vocab(dummy_vocab_path)[0]), str)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
correct_dummy_vocab = ['this', 'is', 'a', 'dummy', 'voabulary', '!']
assert_equal(read_vocab(dummy_vocab_path), correct_dummy_vocab)


# ### Read the training corpus
# ## 1.2
# The training corpus contains 4219 sentences with words labelled with their correct POS tags by humans. Each sentence is located on its own line, the words are separated from each other by whitespaces. The word is separated from its tag like this: Word_/_TAG.

# In[ ]:


def read_train(file_name):
    """
    this function takes in a path to a training corpus file, reads the file,
    and returns a list of sentences. each sentence is in turn a list of tuples, 
    where the first element is a word string and the second element is its tag
    
    INPUT: file_name - a path to a file as a string
    OUTPUT: words_and_tags - a list of lists. [[('word1','tag'),('word2', 'tag')],[('word3','tag')]]
    """

    # YOUR CODE HERE
    f = open(file_name, 'r').read()
    sentences = f.split('\n')
    words_and_tags = []
    for sentence in sentences:
        tmp = []
        for el in sentence.split():
            if len(el.split('/')) != 2:
                t = el.split('/')[-1]
                w = '/_'
            else:
                w, t = el.split('/')
            tmp.append((w[:-1], t[1:]))
        words_and_tags.append(tmp)

    return words_and_tags


# In[ ]:


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
dummy_train_path = "/home/contentis/aalto/nlp/coursedata/POS/dummy_train.txt"

# check that the output of the function is a list
assert_equal(type(read_train(dummy_train_path)), list)
# check that it's a list of lists
assert_equal(type(read_train(dummy_train_path)[0]), list)
# check that it's a list of lists of tuples
assert_equal(type(read_train(dummy_train_path)[0][0]), tuple)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
correct_dummy_train = [[('word1', 'TAG1'), ('word2', 'TAG2')],
                       [('word3', 'TAG3'), ('word4', 'TAG4')]]

assert_equal(read_train(dummy_train_path), correct_dummy_train)


# ### Read the test corpus
# ## 1.3
# The test corpus is located in two separate files. One contains 1055 unlabelled test sentences. Each sentence is located on its own line. Another file contains cortresponding 1055 sequences of tags. Each tag sequence is located on its own line. The words and tags are separated from each other by whitespaces.

# In[27]:


def read_test(file_name):
    """
    this function takes in a path to a test corpus file, reads the file,
    and returns a list of sentences. each sentence is in turn a list of words or a list of tags

    INPUT: file_name - a path to a file as a string
    OUTPUT: test_sents - a list of lists. [['A','B'],['C']]
    """

    # YOUR CODE HERE
    text = open(file_name, 'r').read()
    sentences = text.split('\n')
    test_sents = []
    for sentence in sentences:
        test_sents.append(sentence.split())

    return test_sents


# In[28]:


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
dummy_test_path = "/home/contentis/aalto/nlp/coursedata/POS/dummy_test.txt"

# check that the output of the function is a list
assert_equal(type(read_test(dummy_test_path)), list)
# check that it's a list of lists
assert_equal(type(read_test(dummy_test_path)[0]), list)
# check that it's a list of lists of strings
assert_equal(type(read_test(dummy_test_path)[0][0]), str)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
correct_dummy_test = [['A', 'B', 'C'], ['D', 'E', 'F', 'G']]

assert_equal(read_test(dummy_test_path), correct_dummy_test)

# #### read our data by running the cell below

# In[ ]:


tags_vocab = read_vocab("/home/contentis/aalto/nlp/coursedata/POS/tags_vocab.txt")
words_vocab = read_vocab("/home/contentis/aalto/nlp/coursedata/POS/words_vocab.txt")

words_and_tags = read_train('/home/contentis/aalto/nlp/coursedata/POS/train.txt')

test_words = read_test("/home/contentis/aalto/nlp/coursedata/POS/test_words.txt")
test_tags = read_test("/home/contentis/aalto/nlp/coursedata/POS/test_tags.txt")

# ## TASK 2
# ## Study the data
# 
# It is always good to look at closely at your data. In this task we're going to study things like: how many words are ambiguous, what the most popular POS in English is, what POS is most likely to start a sentence. 
# 
# ### Collect word to tag statistics
# ## 2.1 
# Using the statistics from our trainin corpus, create a matrix, where rows are words and columns are tags. The cells of this matrix are the number of times a word was seen with some tag.
# 
# For example, imagine, that our training corpus looks like this:
# 
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/corpus.png">
# 
# Then, our word to tag matrix will be as follows:
# 
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/wt.png">

# In[ ]:


import numpy as np
from collections import Counter


def create_word_tag_matrix(training_corpus, vocab_words, vocab_tags):
    """
    this function takes in a training_corpus, it word and tag vocabularies,
    and creates a word-tag matrix
    
    INPUT: 
    training_corpus - a list of lists. [[('word1','tag'),('word2', 'tag')],[('word3','tag')]]
    vocab_words - a list of words
    vocab_tags - a list of UD tag labels
    
    OUTPUT: wt_matrix - an numpy array containg word to tag statistics [len(vocab_words) X len(vocab_tags)]
    """

    wt_matrix = np.zeros((len(vocab_words), len(vocab_tags)))

    for sentence in training_corpus:
        for tuple in sentence:
            w, t = tuple
            if t in vocab_tags and w in vocab_words:
                widx = vocab_words.index(w)
                tidx = vocab_tags.index(t)
                wt_matrix[widx, tidx] += 1

    # YOUR CODE HERE
    return wt_matrix


# In[ ]:


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

dummy_train = [[('word1', 'TAG1'), ('word2', 'TAG2')],
               [('word3', 'TAG2'), ('word4', 'TAG2'), ('word2', 'TAG1')]]
dummy_word_vocabulary = ['word1', 'word2', 'word3', 'word4']
dummy_tag_vocabulary = ['TAG1', 'TAG2']

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(create_word_tag_matrix(dummy_train,
                                    dummy_word_vocabulary,
                                    dummy_tag_vocabulary).shape, (4, 2))

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the matrix has the right values in the right places
correct_wt_dummy_matrix = np.array([[1., 0.],
                                    [1., 1.],
                                    [0., 1.],
                                    [0., 1.]])

assert_array_equal(create_word_tag_matrix(dummy_train,
                                          dummy_word_vocabulary,
                                          dummy_tag_vocabulary), correct_wt_dummy_matrix)

# #### create the word-to-tag matrix by running the cell below

# In[ ]:


wt_matrix = create_word_tag_matrix(words_and_tags, words_vocab, tags_vocab)

# ###  words and their possible tags / tags and their possible words
# ## 2.2
# 
# Looking at the matrix we've created, answer the following question:
# - What is the most frequent tag?
# - What tag was given to the least number of different words?
# - What is the maximum number of different tags one word in the training corpus has?
# - What is the word with the maximum number of different tags?
# - How many words are unambiguous (words having only one tag)?
# - What is the proportion of unambiguous word types in the vocabulary?

# In[ ]:


# type in the answer as a string. it should be written exactly as in the tag vocabulary.
# For example:
# most_popular_tag = "ADJ"
most_popular_tag = tags_vocab[np.argmax(np.sum(wt_matrix, axis=0))]

# type in the answer as a string. it should be written exactly as in the tag vocabulary.
# For example:
# least_open_tag = "ADJ"
least_open_tag = tags_vocab[np.argmin(np.sum(wt_matrix, axis=0))]

# type in the answer as an integer.
# For example:
# max_n_of_different_tags = 2
max_n_of_different_tags = np.max(np.sum(np.array(wt_matrix, dtype='bool'), axis=1))

# type in the answer as a string. it should be written exactly as in the word vocabulary.
# For example:
# most_ambiguos_word = "."
most_ambiguos_word = words_vocab[np.argmax(np.sum(np.array(wt_matrix, dtype='bool'), axis=1))]

# type in the answer as an integer.
# For example:
# n_of_ambiguous_words = 200
n_of_ambiguous_words = np.sum(np.sum(np.array(wt_matrix, dtype='bool'), axis=1) == 1)

# type in the answer as a float number from 0 to 1.
# For example:
# part_of_unambiguous_words = 0.2
part_of_unambiguous_words_in_vocab = n_of_ambiguous_words / len(words_vocab)

# Remember to remove the raise NotImplementedError line:
# YOUR CODE HERE

# In[ ]:


### This cell contains hidden tests for the correct answers.
from numpy.testing import assert_almost_equal
from nose.tools import assert_equal


# ### Collect tag-to-tag transition statistics
# ## 2.3
# 
# Another thing we can easily do with our data is creating a bi-gram language model for tags! We will represent it as a tag-to-tag transition statistics data. 
# 
# As you remember from our language modelling assignment, we also want information about what tag starts a sentence and what tag ends it, so we will need to modify our tag sequences by appending special start-of-sentence **&lt;s>** and end-of-sentence **&lt;/s>** symbols.
# 
# Create a tag-to-tag transition matrix to capture this information. The first row of the matrix will correspond to the start symbol, other rows are just tags in their alphabetical order. The columns of the matrix are, again, tags in their their alphabetical order, and the last column is an end of sentence tag. Each cell corresponds to the number of times a column tag was seen after a row tag in our training corpus.
# 
# For our toy corpus this matrix will look this way:
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/tt.png">

# In[ ]:


def create_tag_to_tag_transition_matrix(training_corpus, vocab_tags):
    """
    this function takes in a training corpus and its tag vocabulary,
    and return a tag_to_tag_transition_matrix of size [len(tag_vocabulary)+1 X len(tag_vocabulary)+1]
    the first row contains the number of times each tag started a sentence
    the last column contains the number of times each tag enedd a sentence
    
    INPUT: 
    training_corpus - a list of lists. [[('word1','tag'),('word2', 'tag')],[('word3','tag')]]
    vocab_tags - a list of UD tag labels
    
    OUTPUT: tag_transition_matrix - an numpy array containg tag to tag transition statistics [len(tag_vocabulary)+1 X len(tag_vocabulary)+1]
    """
    tag_transition_matrix = np.zeros((len(vocab_tags) + 1, len(vocab_tags) + 1))

    # YOUR CODE HERE
    for sentence in training_corpus:
        cur = 0
        for el in sentence:
            w, t = el
            nxt = vocab_tags.index(t)
            tag_transition_matrix[cur, nxt] += 1
            cur = nxt+1
        tag_transition_matrix[cur, -1] += 1

    return tag_transition_matrix.astype(int)


# In[ ]:


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

dummy_train2 = [[('word1', 'TAG1'), ('word2', 'TAG2')],
                [('word3', 'TAG1'), ('word4', 'TAG2'), ('word2', 'TAG1')]]

dummy_tag_vocabulary = ['TAG1', 'TAG2']

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(create_tag_to_tag_transition_matrix(dummy_train2, dummy_tag_vocabulary).shape, (3, 3))

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the matrix has the right values in the right places
correct_tt_dummy_matrix = np.array([[2, 0, 0],
                                    [0, 2, 1],
                                    [1, 0, 1]])

assert_array_equal(create_tag_to_tag_transition_matrix(dummy_train2, dummy_tag_vocabulary), correct_tt_dummy_matrix)

# #### create and display the tag-to-tag transition matrix by running the cell below

# In[ ]:


tt_matrix = create_tag_to_tag_transition_matrix(words_and_tags, tags_vocab)

import pandas as pd

df = pd.DataFrame(data=tt_matrix, index=["START"] + tags_vocab, columns=tags_vocab + ["END"])
df

# ###  tag to tag transions
# ## 2.4
# 
# Looking at the tag-to-tag transition matrix we've created, answer the following question:
# - What is the most popular tag bi-gram?
# - What tag is most likely to follow the adjective tag?
# - What tag is most likely to precede  interjection?
# - What tag is most likely to start a sentence?
# - What tag is most likely to end a sentence?
# - What tag can never start a sentence according to our training data?
# - How many tags can never end a sentence according to our training data?

# In[ ]:

idx = np.unravel_index(np.argmax(df.values), df.values.shape)
# type in the answer as a tuple containing two strings. write tags exactly as in the tag vocabulary.
# For example:
# most_popular_tag_bi_gram = ('ADJ', 'ADJ')
most_popular_tag_bi_gram = (df.index[idx[0]], df.columns[idx[1]])

# type in the answer as a string. it should be written exactly as in the tag vocabulary.
# For example:
# tag_after_adj = 'ADJ'
tag_after_adj = df.columns[np.argmax(df.loc['ADJ'].values)]

# type in the answer as a string. it should be written exactly as in the tag vocabulary.
# For example:
# tag_before_intj = 'ADJ'
tag_before_intj = df.index[np.argmax(df.INTJ.values)]

# type in the answer as a string. it should be written exactly as in the tag vocabulary.
# For example:
# start_tag = 'ADJ'
start_tag = df.columns[np.argmax(df.loc['START'].values)]

# type in the answer as a string. it should be written exactly as in the tag vocabulary.
# For example:
# end_tag = 'ADJ'
end_tag = df.index[np.argmax(df.END.values)]

# type in the answer as an integer number.
# For example:
# n_of_non_final_tags = 15
n_of_non_final_tags = np.sum(np.array(df.END.values, dtype='bool')==0)

# Remember to remove the raise NotImplementedError line:
# YOUR CODE HERE

# In[ ]:


### This cell contains hidden tests for the correct answers.
from numpy.testing import assert_almost_equal
from nose.tools import assert_equal


# ## TASK 3
# ## BASELINE
# ### Create a baseline
# ## 3.1
# 
# To understand if our tagger is any good we will need to compare it to some baseline model. One popular approach is to assign each word a tag that it has been labelled the most with in the trainin data.
# 
# Create a function, that labels test word sequences with the most frequent tags. If some word has several tags with the same frequency, just select the one that comes first alphabetically. Assign the word unseen in the training corpus with the 'X' tag.

# In[ ]:


def baseline(word_tag_matrix, test_words, vocab_words, vocab_tags):
    """
    this function takes in word to tag matrix, test sentences to label, word and tag vocabularies, 
    and assigns every word in test sentences the most frequent tag it was seen with
    
    INPUT: 
    word_tag_matrix - an numpy array containg word to tag statistics [len(vocab_words) X len(vocab_tags)]
    test_sents - a list of lists. [['word1','word2'],['word3']]
    vocab_words - a list of words in the training corpus
    vocab_tags - a list of UD tag labels
    
    
    OUTPUT: test_tags_predicted - predicted tags. a list of lists. [['tag1','tag2'],['tag3']]
    """
    most_freq_tag = {}
    for ind, word in enumerate(vocab_words):
        tag_ind = np.argmax(word_tag_matrix[ind])
        most_freq_tag[word] = vocab_tags[tag_ind]

    test_tags_predicted = []
    # YOUR CODE HERE
    for sentence in test_words:
        tmp = []
        for word in sentence:
            if word in most_freq_tag:
                tmp.append(most_freq_tag[word])
            else:
                tmp.append('X')
        test_tags_predicted.append(tmp)
    return test_tags_predicted


# In[ ]:


from numpy.testing import assert_array_equal
from nose.tools import assert_equal

wt_dummy_train = np.array([[1., 0.],
                           [1., 1.],
                           [1., 0.],
                           [0., 1.]])

dummy_word_vocabulary = ['word1', 'word2', 'word3', 'word4']
dummy_tag_vocabulary = ['TAG1', 'TAG2']
dummy_test = [['word1', 'word2', 'word5']]

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check that the output is a list
assert_equal(type(baseline(wt_dummy_train, dummy_test, dummy_word_vocabulary, dummy_tag_vocabulary)), list)
# check that the output is a list of lists
assert_equal(type(baseline(wt_dummy_train, dummy_test, dummy_word_vocabulary, dummy_tag_vocabulary)[0]), list)
# check that the output is a list of lists of strings
assert_equal(type(baseline(wt_dummy_train, dummy_test, dummy_word_vocabulary, dummy_tag_vocabulary)[0][0]), str)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the matrix has the right values in the right places
correct_dummy_tags = [['TAG1', 'TAG1', 'X']]

assert_array_equal(baseline(wt_dummy_train, dummy_test, dummy_word_vocabulary, dummy_tag_vocabulary),
                   correct_dummy_tags)


# ### Evaluate accuracy
# ## 3.2
# Create a function to estimate how accurate our POS tagging algorithm is. It should produce the percentage of tags that were assigned correctly.

# In[ ]:


def accuracy(y_true, y_predicted):
    """
    this function takes in true labels for the test data and labels that were output by the algorith,
    and then returns the percent of labels that were right.
    
    INPUT: 
    y_true - a list of lists with right tags for each sentence in test corpus. [['TAG1','TAG2'],['TAG1']]
    y_predicted - a list of lists with predicted tags for each sentence in test corpus. [['TAG1','TAG2'],['TAG1']]
    
    OUTPUT: 
    accuracy - percentage of correctly predicted tags
    
    """
    # YOUR CODE HERE
    accuracy = []
    for stgt, spred in zip(y_true, y_predicted):
        for tgt, pred in zip(stgt, spred):
            accuracy.append(tgt==pred)
    accuracy = float(np.mean(accuracy))*100
    return accuracy


# In[ ]:


from numpy.testing import assert_almost_equal
from nose.tools import assert_equal

dummy_true_y = [['TAG1', 'TAG1', 'TAG2']]
dummy_predicted_y = [['TAG1', 'TAG1', 'X']]

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check that the output is a float number
assert_equal(type(accuracy(dummy_true_y, dummy_predicted_y)), float)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check the function is working as expected
assert_almost_equal(accuracy(dummy_true_y, dummy_predicted_y), 66.66, 2)

# #### find out how accurate our baseline model by running the cell below

# In[ ]:


tags_predicted = baseline(wt_matrix, test_words, words_vocab, tags_vocab)
print(accuracy(tags_predicted, test_tags))

# ### Confusion matrix
# 
# The accuracy score of your baseline model should be around 85 percent. But it doesn't tell us much about what's going wrong.
# 
# We will create a confusion matrix. A confusion matrix tells us how many times each true tag was predicted as itself and as some other tag. The rows of the matrix are correct tabels, the columns are all tags it could have been confused with. The cell tells how many times a true tag was predicted as some column tag.
# 
# Run the cell below to calculate the confusion matrix for our baseline model. 

# In[ ]:


from sklearn.metrics import confusion_matrix


def create_confusion_matrix(y_true, y_predicted):
    y_true_as_array_of_tags = [tag for sent in y_true for tag in sent]
    y_predicted_as_array_of_tags = [tag for sent in y_predicted for tag in sent]

    cm = confusion_matrix(y_true_as_array_of_tags, y_predicted_as_array_of_tags)

    return cm


cm = create_confusion_matrix(tags_predicted, test_tags)


# ### Normalized Confusion matrix
# ## 3.3
# 
# You can already inspect the confusion matrix above, but let's be honest, raw counts are hard to compare and plot. 
# We will need to normalize our matrix: to make a number of predictions of every tag sum to 1. This will help us to compare mistakes made for both frequent and infrequent tags. The true labels in our confusion matrix are marked as rows, the cells in these rows should correspond to the fraction of times this tag was predicted as a tag that marks the column.
# 
# Create a function, that takes in a matrix and normalizes its values across a given axis. 

# In[ ]:


def normalize_matrix(matrix, axis):
    """
    this function takes in a matrix, sums its values across a given axis (columns if axis = 0, rows if axis = 1), 
    and normalizes its cell values according to this sum.
    
    INPUT: 
    matrix - a 2d numpy array
    
    OUTPUT: 
    normalized_matrix - a 2d numpy array
    """
    # YOUR CODE HERE
    a = np.sum(matrix, axis=axis)
    if axis:
        normalized_matrix = (matrix.T/a).T
    else:
        normalized_matrix = matrix/a
    return normalized_matrix


# In[ ]:


from numpy.testing import assert_array_almost_equal
from nose.tools import assert_equal

dummy_matrix = np.array([[1, 2],
                         [3, 3]])

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check the shape of the matrix
assert_equal(normalize_matrix(dummy_matrix, 1).shape, (2, 2))

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check that the matrix has the right values in the right places
correct_normalised_matrix_axis_zero = np.array([[0.25, 0.4],
                                                [0.75, 0.6]])
correct_normalised_matrix_axis_one = np.array([[0.33, 0.67],
                                               [0.5, 0.5]])

assert_array_almost_equal(normalize_matrix(dummy_matrix, 0), correct_normalised_matrix_axis_zero, 2)
assert_array_almost_equal(normalize_matrix(dummy_matrix, 1), correct_normalised_matrix_axis_one, 2)

# #### visualize normalized confusion matrix by running the cell below

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, vocab_tags):
    plt.figure(figsize=(11, 8))
    colors = sns.light_palette((220, 50, 20), input="husl", n_colors=80)
    ax = sns.heatmap(np.around(cm, 2),
                     annot=True,
                     linewidths=.8,
                     cmap=colors)
    ax.set_ylim(bottom=17, top=0)
    ax.set(xticklabels=vocab_tags)
    ax.set(yticklabels=vocab_tags)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.ylabel("True Tags")
    plt.xlabel("Predicted Tags")
    plt.show()


# normalize cm by rows
cm_normalized = normalize_matrix(cm, 1)
plot_confusion_matrix(cm_normalized, tags_vocab)

# ### Study the baseline results
# ## 3.4
# By looking at the visualization of normalized confusion matrix, briefly answer the following questions in the cell below:
# - What tags were predicted best? How would you explain it?
# - Why some tags (nouns, proper nouns, verbs...) were predicted as 'X' that often? Why 'X' was never mistaked with some other classes?
# - How many unseen words were in the test corpus? Does it affect performance of the baseline algorithm?

# YOUR ANSWER HERE

# ## TASK 4
# ## HMM POS-tagger
# An **HMM** is a probabilistic sequence model. In our case, given a sequence of words, it computes a probability distribution over possible sequences of POS tags and chooses the best tag sequence.
# 
# The hidden part of our HMM are tags, because they are some abstract classes that are not directly observed from text sequences. The observed part of out HMM are words these hidden tags produce. 
# The components of our HMM will be:
# 1. T - a set of $N$ POS tags
# 
# 2. $A$ - a transition probability matrix. Each cell $a_{i,j}$ represents a probability of moving from $tag_i$ to $tag_j$: $P(t_j|t_i)$. We also add the start and end probabilities to this matrix, so it has $N+1$x$N+1$ dimensions. The row $a_1$ represents the probabilities of each tag to start a sentence. The last column of transition probability matrix A contains the probability of every tag to end a sentence.
# 
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/A.png">
# 3. $B$ - an observation likelihood matrix. Each cell $b_{i,j}$ represents a probability of a $word_i$ being generated out of some $tag_j$: $P(w_i|t_j)$
# 
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/B.png">
# 
# ### Collect probability matrices A and B
# ## 4.1
# 
# Turns out, we already have everything for our HMM, we just need to turn the frequencies that we collected previously into maximum likelihood probabilities.
# 
# You can do it by normalizing out word-tag and tag-tag matrices across the appropriate axis.

# In[ ]:


# type in the right axis instead of None value to normalize word to tag matrix
# For example:
# axis_to_normalize_wt_matrix_by = 0
axis_to_normalize_wt_matrix_by = None
# type in the right axis instead of None value to normalize tag to tag matrix
# For example:
# axis_to_normalize_tt_matrix_by = 0
axis_to_normalize_tt_matrix_by = None

# Remember to remove the raise NotImplementedError line:
# YOUR CODE HERE
raise NotImplementedError()

# In[ ]:


### This cell contains hidden tests for the correct answers.
from numpy.testing import assert_almost_equal
from nose.tools import assert_equal

# #### create HMM probability matrices by running the cell below

# In[ ]:


A = normalize_matrix(tt_matrix, axis_to_normalize_tt_matrix_by)
B = normalize_matrix(wt_matrix, axis_to_normalize_wt_matrix_by)

# #### display tag transition matrix A by running the cell below
# Examine that everything looks as you would expect it to.

# In[ ]:


import pandas as pd

df = pd.DataFrame(data=A, index=["START"] + tags_vocab, columns=tags_vocab + ["END"])
df


# ### HMM decoding
# 
# The aim of an HMM decoding is to choose the tag sequence $t^n_1$ that will be the most probable given the observation sequence of $n$ words $w^n_1$: $\underset{t^n_1}{\arg\max}P(t^n_1|w^n_1)$
# 
# Using Bayes' rule, we can very conveniently flip this into: $\underset{t^n_1}{\arg\max}\frac{P(w^n_1|t^n_1)P(t^n_1)}{P(w^n_1)}$. You can also notice, that we don't need denominator for maximizing the tag sequence probability. Thus:
# 
# *best tag sequence* $= \underset{t^n_1}{\arg\max}P(w^n_1|t^n_1)P(t^n_1)$
# 
# Now we can simplify it even further by assuming:
# 
# 1. the probability of a particular tag depends only on the previous tag
# 2. the probability of an observed word depends only on the tag that produced this word
# 
# *best tag sequence* $= \underset{t^n_1}{\arg\max}\displaystyle\prod_{i=1}^{n}P(w_i|t_i)P(t_i|t_{i-1})$
# 
# Lucky us, we've already collected probabilities $P(w_i|t_i)$ in the matrix A, and $P(t_i|t_{i-1})$ in the matrix B.

# ### Viterbi algorithm
# 
# The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states. In our case, its output is the most probable sequence of POS tags and its probability for some word sequence.
# 
# * STEP 1: get a sequence you want to tag:
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/seq.png">
# * STEP 2: create a path probability matrix $V$ with the shape (number of tags, number of words to tag). 
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/table1.png">
# Each cell $V_{j,i}$ represents the probability that the HMM has tag $j$ after seeing $i$ words and passing through the most probable tag sequence $t_1,t_2...t_{i-1}$.
# This most probable path of tags so far is represented as maximum over all previous tag sequences. The probabilities of $V_{i,j}$ are computed by starting from the most probable of the extensions of the paths that lead to the current cell.
# 
# * STEP 3: creat a backpointer table P of shape (number of tags, number of words to tag). You can think of it as a table where to write the most probable tag that generated the word before the word we want to tag now. You can skip the first word, since the tag before it was just the beggining of the sentence. 
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/table2.png">
# * STEP 4: start filling the first column of $V$. Each cell $V_{j,1}$ contains a probability of a tag $j$ being the starting tag of the sentence $P(t_j|start)$ multiplied by the probability of the first word in the sequence being generated by this tag $P(word_1|t_j)$
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/table1_1.png">
# * STEP 5: move on to fill in the second column of $V$. Each cell $V_{j,2}$ contains a product of two probabilities:
#     1. the maximum from the products of every value in the previous column of $V$ and every transition probability to the tag $j$. This value keeps balance between what is likely according to the tag language model and what we have seen before.
#     <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/max.png">
#     2. the probability of the second word in the sequence being generated by the tag $j$ $P(word_2|t_j)$
#     <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/max_and_wt.png">
# * STEP 6: save the backpointer to the index of the maximum from the products of every value in the previous column of $V$ and every transition probability to the tag $j$ (to the most probable tag of the previous word).
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/backpointer.png">
# * STEP 7: compute the rest of the table $V$ as in STEP 5, don't forget to save backpointers as in STEP 6.
# * STEP 8: compute the final output probabilities of your tag paths by multiplying the last column of V by probability of each tag to end a sentence. 
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/final.png">
# * STEP 9: add the last backpointer by chosing the most probable last tag from STEP 8.
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/last_pointer.png">
# * STEP 10: trace back the indices in your backpointer table P, starting with one that was output by STEP 9
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/path.png">
# * STEP 11: return the most probable tag sequence from the backtraced path.
# <img src= "../../../home/contentis/aalto/nlp/coursedata/notebook_illustrations/tags_predicted.png">

# ### Create Viterbi algorithm
# ## 4.2
# Write a function for the viterbi algorithm. 
# 
# * Note 1: to avoid numerical underflow, use log probabilities.
# * Note 2: when you encounter an unseen word, cheat and don't include the observation probabilities. 

# In[ ]:


def viterbi(A, B, word_sequence, tags_vocab, words_vocab):
    """
    this function takes in HMM matrices A and B, test sequence to tag, word and tag vocabularies,
    and returns the tags for the test sequence.
    
    INPUT: 
    A - transition probability matrix for POS tags
    B - an observation likelihood matrix
    word_sequence - a list of word for a test sequence
    tags_vocab - a list of UD tag labels
    words_vocab - a list of words seen in during training
      
    OUTPUT: 
    best_path - a list of tags for the words in a test sequence
    """

    path_probability_matrix = np.zeros((len(tags_vocab), len(word_sequence)))
    backpointer_table = np.full((len(tags_vocab), len(word_sequence)), -1)

    # FILL IN THE FIRST WORD'S COLUMN
    # YOUR CODE HERE
    raise NotImplementedError()

    # FILL IN THE REST OF THE path_probability_matrix
    # DON'T FORGET TO KEEP BACKPOINTERS
    # YOUR CODE HERE
    raise NotImplementedError()

    # FIND THE LAST BACKPOINTER
    # YOUR CODE HERE
    raise NotImplementedError()

    # BACKTRACE THE BEST PATH
    best_path = []
    # YOUR CODE HERE
    raise NotImplementedError()

    return best_path


# In[ ]:


from nose.tools import assert_equal

dummy_A = np.array([[3 / 4, 1 / 4, 0], [1 / 4, 3 / 4, 0], [0, 0, 1.]])
dummy_B = np.array([[0, 1], [3 / 4, 0], [1 / 4, 0]])
dummy_test = ['red', 'right', 'hand']
dummy_tags_vocab = ["ADJ", "NOUN"]
dummy_words_vocab = ['hand', 'red', 'right']

# CHECKING THE GENERAL PROPERTIES OF THE OUTPUT
# check that the output is a list
assert_equal(type(viterbi(dummy_A, dummy_B, dummy_test, dummy_tags_vocab, dummy_words_vocab)), list)
# check that the output is a list of strings
assert_equal(type(viterbi(dummy_A, dummy_B, dummy_test, dummy_tags_vocab, dummy_words_vocab)[0]), str)

# CHECKING THAT THE FUNCTION IS WORKING AS IT SHOULD
# check the function is giving out the right tag sequence
correct_dummy_tags = ['ADJ', 'ADJ', 'NOUN']
assert_equal((viterbi(dummy_A, dummy_B, dummy_test, dummy_tags_vocab, dummy_words_vocab)), correct_dummy_tags)

# CHECKING THAT THE FUNCTION IS WORKING WITH UNKNOWN WORDS
dummy_test2 = ['red', 'right', 'leg']
correct_dummy_tags2 = ['ADJ', 'ADJ', 'NOUN']
assert_equal((viterbi(dummy_A, dummy_B, dummy_test2, dummy_tags_vocab, dummy_words_vocab)), correct_dummy_tags2)

# ### Evaluate HMM
# #### run the cell below to get an accuracy score for your HMM tagger and to plot the confusion matrix
# 

# In[ ]:


hmm_tags_predicted = []
for i in range(len(test_words)):
    hmm_tags_predicted.append(viterbi(A, B, test_words[i], tags_vocab, words_vocab))

print(accuracy(hmm_tags_predicted, test_tags))

cm_hmm = create_confusion_matrix(hmm_tags_predicted, test_tags)
cm_hmm_normalized = normalize_matrix(cm_hmm, 1)
plot_confusion_matrix(cm_hmm_normalized, tags_vocab)

# ### Compare HMM and Baseline
# ## 4.2
# Briefly decribe in the cell below the differences in the performance of our HMM and the baseline model. What can be done to further improve the HMM model?

# YOUR ANSWER HERE
