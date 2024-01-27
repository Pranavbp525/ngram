import re
import math
from random import choices
from collections import Counter

import nltk

nltk.download('gutenberg')
nltk.download('punkt')

from nltk.corpus import gutenberg
from nltk.util import ngrams


def clean_and_preprocess(raw_text):
    """
    Cleans and preprocess the raw text by removing non-alphanumeric characters/words, removing html tags and, lowering the
    case of all the text
    :param raw_text: The raw unfiltered unprocessed text.
    :return: clean text that can be used for language modeling
    """
    clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', raw_text)
    clean_text = re.sub('<[^>]+>', '', clean_text)
    clean_text = clean_text.replace('\u202f', ' ')
    clean_text = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', clean_text)
    clean_text = clean_text.lower()

    return clean_text


raw_text = gutenberg.raw("austen-emma.txt")
clean_text = clean_and_preprocess(raw_text)
words = nltk.word_tokenize(clean_text)

test = gutenberg.raw("austen-persuasion.txt")
clean_test = clean_and_preprocess(test)
test_words = nltk.word_tokenize(clean_test)


class NGRAM:

    def __init__(self, n, words):
        """
        Represents a NGRAM class that consists of all the methods for creating a ngram next word prediction model,
        that calculates the probabilites of all the words given n-1 grams, and then samples the next word according to
        the probabilites and generates the sampled word. The class also contains methods for generating a sentence by,
        repeatedly predicting the next word, as well as for calculating the perplexity.
        :param n: n determines, on how many of the previous words in the sentence does the probability of the next word
        depend on.
        :param words: a list of tokens, that were result of tokenizing a dataset of text.
        """
        self.n = n
        self.words = words
        self.n_grams = self.gen_n_grams(self.words, self.n)
        self.n_minus_1_grams = self.gen_n_grams(self.words, self.n - 1)
        self.n_gram_counter = Counter(self.n_grams)
        self.n_minus_1_gram_counter = Counter(self.n_minus_1_grams)

    def gen_n_grams(self, tokens, n=2):
        """
        Converts the list of tokens into a list of tuples where each tuple represents an n-gram
        :param tokens: a list of tokens that were result of tokenizing a text
        :param n: the n value of the n-gram (2 or bi-gram, 3 for tri-gram etc.)
        :return:  a list of n-grams
        """
        return ngrams(tokens, n)

    def calculate_probability(self, word, n_minus_1_gram):
        """
        Calculates the probabilities of a word given a prefix of n-1 words in a sentence preeceding it
        :param word: The word whose probability of occurance given the n-1 gram you want to compute
        :param n_minus_1_gram: The n-1 gram that acts as a prefix for computing the probability
        :return: The probability of the word occuring given the n-1 gram has occured in a sentence
        """
        n_gram_frequency = self.n_gram_counter[n_minus_1_gram + (word,)]
        n_minus_1_gram_frequency = self.n_minus_1_gram_counter[n_minus_1_gram]
        probability = n_gram_frequency / n_minus_1_gram_frequency
        return probability

    def predict_next_word(self, sequence_of_words):
        """
        Predicts the next word in the sequence given a sequence of words, by extracting the n-1 gram from the sequence,
        and calculating probabilities based on the n-1 gram for all the words in the corpus and using the probabilites
        to sample the next word
        :param sequence_of_words: a sequence of words that serve as a prefix to generate the next word.
        :return: the sampled next word, that was predicted from probabilites
        """
        n_minus_1_gram = tuple(nltk.word_tokenize(sequence_of_words)
                               [-self.n + 1:])
        probs = {}

        for word in set(self.words):
            probs[word] = self.calculate_probability(word, n_minus_1_gram)

        words = list(probs.keys())
        probs = list(probs.values())
        next_word = choices(words, probs)

        return next_word[0]

    def generate_sentence(self, length, n_minus_1_words):
        """
        Generates a sentence of fixed length by repeatedly generating next word based on the previous n-1 words.
        :param length: The length of the sentence to be generated
        :param n_minus_1_words: the initial prefix or n-1 words to get the model started on the repetitive generation
        of next word
        :return: a sentence of length specified by user.
        """
        for i in range(length - (self.n - 1)):
            next_word = self.predict_next_word(n_minus_1_words)
            n_minus_1_words += " " + next_word

        return n_minus_1_words

    def perplexity(self, test_words):
        """
        Evaluates the ngram model using the perplexity formulae.
        :param test_words: The test set on which we want to evaluate the n-gram model
        :return: the perplexity score of the model (lower is better)
        """
        N = 0
        log_prob_sum = 0
        n_minus_1_grams = list(self.gen_n_grams(test_words, self.n - 1))
        for i in range(self.n - 1, len(test_words)):
            n_minus_1_gram = n_minus_1_grams[i - (self.n - 1)]
            word = test_words[i]
            N += 1
            log_prob_sum -= math.log(self.calculate_probability(word,
                                                                n_minus_1_gram))
        perplexity = 2 ** (log_prob_sum / N)
        return perplexity


class SmoothNGRAM(NGRAM):

    def __init__(self, n, words):
        """
        SmoothNGRAM inherits NGRAM, and has all the functionality of NGRAM, except while calculating the probabilites,
        this model applies laplace smoothing inorder to avoid the division by zero exception that may occur while making
        the ngram model geenerate next words, based on never before seen n-1 grams as prefixes.
        :param n: n determines, on how many of the previous words in the sentence does the probability of the next word
        depend on.
        :param words: a list of tokens, that were result of tokenizing a dataset of text.
        """
        super().__init__(n, words)
        self.vocab_size = len(list(set(self.words)))

    def calculate_probability(self, word, n_minus_1_gram):
        """
        Calculates the laplace smoothed probabilities of a word given a prefix of n-1 words in a sentence preeceding it
        :param word: The word whose smoothed probability of occurance given the n-1 gram you want to compute
        :param n_minus_1_gram: The n-1 gram that acts as a prefix for computing the smoothed probability
        :return: The laplace smoothed probability of the word occuring given the n-1 gram has occured in a sentence
        """
        n_gram_frequency = self.n_gram_counter[n_minus_1_gram + (word,)]
        n_minus_1_gram_frequency = self.n_minus_1_gram_counter[n_minus_1_gram]

        probability = (n_gram_frequency + 1) / (n_minus_1_gram_frequency +
                                                self.vocab_size)

        return probability
