from numpy.testing import assert_almost_equal
import math
import itertools
import random
import nltk.tokenize
import nltk.util
from collections import Counter
from nose.tools import assert_equal


def pad(token_list, n):
    """
    this function takes in a list of tokens and pads them with special symbols

    INPUT:
    token_list - a list of tokens to be padded
    n - the length of a token sequence
    OUTPUT:
    padded_list - a padded list of tokens
    """
    start = "<s>"
    end = "</s>"
    # YOUR CODE HERE
    padded_list = [start] * (n - 1) + token_list + [end] * (n - 1)

    return padded_list


def make_n_grams(token_list, n):
    """
    this function takes in a list of tokens and forms a list of n-grams (tuples)

    INPUT:
    token_list - a list of tokens to be converted into n-grams
    n - the length of a token sequence in an n-gram
    OUTPUT:
    n_grams - a list of n-gram tuples
    """
    if n > len(token_list):
        print("The N is too large.")
    # YOUR CODE HERE
    n_grams = [tuple(token_list[i: i + n]) for i in range(len(token_list) - n + 1)]

    return n_grams


def get_counts(sentence_list, n):
    """
    this function takes in a list of tokenized and padded sentences,
    forms a list of n-grams and gives out a dictionary
    with counts for every seen n-gram

    INPUT:
    sentence_list - a list of tokenized and padded sentences to be converted into n-grams
    n - the length of a token sequence
    OUTPUT:
    n_gram_dict - a dictionary of n_gram history parts as keys,
    where their values are a dictionary of all continuations and their counts
    {('a',): {'b': 3 'c': 4}

    """
    # YOUR CODE HERE
    n_gram_dict = {}
    ngrams_list = [make_n_grams(sentence, n) for sentence in sentence_list]
    for ngrams in ngrams_list:
        for ngram in ngrams:
            if ngram[:-1] not in n_gram_dict.keys():
                n_gram_dict[ngram[:-1]] = {ngram[-1]: 1}
            else:
                if ngram[-1] not in n_gram_dict[ngram[:-1]].keys():
                    n_gram_dict[ngram[:-1]][ngram[-1]] = 1
                else:
                    n_gram_dict[ngram[:-1]][ngram[-1]] += 1
    return n_gram_dict


def score_mle(model_counts, n_gram, **scoring_parameters):
    """
    this function takes in a dictionary of ngram counts and some n-gram,
    and gives out an MLE estimate for this n-gram

    INPUT:
    model_counts - a dictionary of n_gram history parts as keys,
    where their values are a dictionary of all continuations and their counts
        {('a',): {'b': 3 'c': 4}
    n_gram - an ngram as tuple
    scoring_parameters - additional, optional scoring parameters, making the function interface generic,
        however not used here.

    OUTPUT:
    mle_score - MLE score for the n_gram
    """

    n = len(list(model_counts.keys())[0]) + 1
    # YOUR CODE HERE
    if n_gram[:-1] not in model_counts.keys() or n_gram[-1] not in model_counts[n_gram[:-1]].keys():
        return 0
    else:
        n_gram_score = model_counts[tuple(n_gram[:-1])][n_gram[-1]] / sum(model_counts[tuple(n_gram[:-1])].values())
    return n_gram_score


def score_smoothed(model_counts, n_gram, **scoring_parameters):
    """
    this function takes in a dictionary of ngram counts, some ngram and delta to be added to this ngram's score,
    and gives out a smoothed estimate for this ngram.
    if some word in an n-gram is unseen during training, the estimate is zero.

    INPUT:
    model_counts - a dictionary of n_gram history parts as keys,
        where their values are a dictionary of all continuations and their counts
        {('a',): {'b': 3 'c': 4}
    n_gram - an ngram as a tuple
    scoring_parameters - additional, optional scoring parameters, which make the function interface generic
        here we will look for scoring_parameters["delta"] - the delta value to be added to the counts
                        and for scoring_parameters["vocab"] -  the vocabulary that can be used as
                        a continuation of an (n-1)-gram.

    OUTPUT:
    smoothed_score - a smoothed score for the n_gram
    """
    delta = scoring_parameters["delta"]
    vocab = scoring_parameters["vocab"]
    context_length = len(next(iter(model_counts)))
    n = context_length + 1  # the ngram length n
    context = n_gram[:context_length]
    word_to_predict = n_gram[-1]
    if any(token not in vocab | {'<s>'} for token in n_gram):
        raise ValueError("Distribution not defined on this n_gram:" + repr(n_gram))

    # YOUR CODE HERE
    if n_gram[:-1] not in model_counts.keys():
        denominator = delta * len(vocab)
        nominator = delta
    else:
        if n_gram[-1] not in model_counts[n_gram[:-1]].keys():
            nominator = delta
        else:
            nominator = delta + model_counts[tuple(n_gram[:-1])][n_gram[-1]]
        denominator = delta * len(vocab) + sum(model_counts[tuple(n_gram[:-1])].values())

    smoothed_score = nominator / denominator
    return smoothed_score


def sentence_logprob(sentence, model_counts, score_function, **scoring_parameters):
    """
    this function takes in a tokenized sentence, language model counts, and a score function.
    it pads the sentence with special symbols,
    and gives out its probability according to the n-gram model and the scoring method

    INPUT:
    sentence - a tokenized sentence
    model_counts - a dictionary of n_gram history parts as keys,
    where their values are a dictionary of all continuations and their counts
    ('a',): {'b': 3 'c': 4}
    score_function - a function which takes a dictionary of counts, an n-gram, and possible additional parameters,
        and produces a score for the last token of the n-gram, given the rest as context
    scoring_parameters - additional, optional scoring parameters, passed to score_function
        like this: score_function(model_counts, ngram, **scoring_parameters)
    OUTPUT:
    logprob - a log probability score of a sentence
    """
    context_length = len(next(iter(model_counts)))
    n = context_length + 1  # the ngram length n
    sentence_grams = make_n_grams(pad(sentence, n), n)
    logprob = 0

    # YOUR CODE HERE
    for ngram in sentence_grams:
        mle = score_function(model_counts, ngram, **scoring_parameters)
        if not mle:
            return -float('inf')
        logprob += math.log(mle)
    return logprob


def perplexity(text, model_counts, score_function, **scoring_parameters):
    """
    this function takes in test text and the n-gram model counts and gives out the perplexity of this n-gram model

    INPUT:
    text - a list of lists of tokenized sentences
    model_counts - a dictionary of n_gram history parts as keys,
    where their values are a dictionary of all continuations and their counts
    {('a',): {'b': 3 'c': 4}
    score_function - a function which takes a dictionary of counts, an n-gram, and possible additional parameters,
        and produces a score for the last token of the n-gram, given the rest as context
    scoring_parameters - additional, optional scoring parameters, passed to score_function
        like this: score_function(model_counts, ngram, **scoring_parameters)

    OUTPUT:
    ppl - a perplexity score of a sentence
    """

    context_length = len(next(iter(model_counts)))
    n = context_length + 1  # the ngram length n
    logprob = 0
    num_predictions = 0  # NOTE: The number of predictions per sentence is len(sentence) + context_length
    # YOUR CODE HERE
    corpus_padded = [pad(sentence, n) for sentence in text]
    n_grams = [make_n_grams(sentence, n) for sentence in corpus_padded][0]
    prob = 1
    num_predictions = len(n_grams)
    for n_gram in n_grams:
        if not score_function(model_counts, n_gram, **scoring_parameters):
            return float('inf')
        else:
            prob *= 1 / score_function(model_counts, n_gram, **scoring_parameters)
    ppl = prob ** (1 / num_predictions)
    return ppl


def generate_text_mle(model_counts):
    """
    this function takes in the n-gram model and produces text until the end symbol is generated.

    INPUT:
    model_counts - a dictionary of n_gram history parts as keys,
    where their values are a dictionary of all continuations and their counts
    ('a',): {'b': 3 'c': 4}
    OUTPUT:
    sentence - a sentence generated by model as a list of tokens.
    """
    context_length = len(next(iter(model_counts)))
    n = context_length + 1  # the ngram length n
    start = tuple(['<s>'] * (n - 1))
    end = '</s>'
    sentence = list(start)

    while sentence[-1] != end:
        # YOUR CODE HERE
        n_grams = []
        n_word = list(model_counts[tuple(sentence[-2:])].keys())
        for key in n_word:
            n_grams.append((sentence[-2], sentence[-1], key))
        p = [score_mle(model_counts, n_gram) for n_gram in n_grams]
        sentence.append(*random.choices(n_word, weights=p))

    # Strip the padding:
    sentence = sentence[n - 1:-1]
    return sentence


def generate_text_smoothed(model_counts, delta, vocab):
    """
    this function takes in the n-gram model and produces text until the end symbol is generated.

    INPUT:
    model_counts - a dictionary of n_gram history parts as keys,
    where their values are a dictionary of all continuations and their counts
    ('a',): {'b': 3 'c': 4}
    delta - the delta value to be added to the counts
    vocab - a set of words in the vocabulary
    OUTPUT:
    sentence - a sentence generated by model as a list of tokens.
    """
    context_length = len(next(iter(model_counts)))
    n = context_length + 1  # the ngram length n
    start = tuple(['<s>'] * (context_length))
    end = '</s>'
    sentence = list(start)

    while sentence[-1] != end:
        curr_context_counts = {}
        try:
            curr_context_counts.update(model_counts[tuple(sentence[-context_length:])])
        except KeyError:  # This context was not seen once
            pass
        # NEXT: Add smoothing delta, then generate the next token.
        # YOUR CODE HERE
        vocab_list = list(vocab)
        if end not in vocab_list:
            vocab_list.append(end)
        n_grams = []
        for word in vocab:
            n_grams.append((sentence[-2], sentence[-1], word))
        p = [score_smoothed(model_counts, n_gram, delta=delta, vocab=vocab) for n_gram in n_grams]
        sentence.append(*random.choices(vocab_list, weights=p))

    # Strip the padding:
    sentence = sentence[context_length:-1]
    return sentence


def replace_oovs(vocab, data, unk="<unk>"):
    """
    vocab: set of tokens
    data: list of lists, i.e. list of sentences, which are lists of tokens
    unk: token to replace tokens which are not in the vocabulary

    This function replaces all tokens not in the vocabulary with the unknown token
    """
    # YOUR CODE HERE
    data_oovs_replaced = list(data)
    for i, sentence in enumerate(data_oovs_replaced):
        for j, token in enumerate(sentence):
            if token not in vocab:
                data_oovs_replaced[i][j] = unk

    return data_oovs_replaced