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
        self.n = n
        self.words = words
        self.n_grams = self.gen_n_grams(self.words, self.n)
        self.n_minus_1_grams = self.gen_n_grams(self.words, self.n - 1)
        self.n_gram_counter = Counter(self.n_grams)
        self.n_minus_1_gram_counter = Counter(self.n_minus_1_grams)

    def gen_n_grams(self, tokens, n=2):
        return ngrams(tokens, n)

    def calculate_probability(self, word, n_minus_1_gram):
        n_gram_frequency = self.n_gram_counter[n_minus_1_gram + (word,)]
        n_minus_1_gram_frequency = self.n_minus_1_gram_counter[n_minus_1_gram]
        probability = n_gram_frequency / n_minus_1_gram_frequency
        return probability

    def predict_next_word(self, sequence_of_words):
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
        for i in range(length - (self.n - 1)):
            next_word = self.predict_next_word(n_minus_1_words)
            n_minus_1_words += " " + next_word
            print(next_word, end=" ")

        return n_minus_1_words


    def perplexity(self, test_words):
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
        super().__init__(n, words)
        self.vocab_size = len(list(set(self.words)))

    def calculate_probability(self, word, n_minus_1_gram):
        n_gram_frequency = self.n_gram_counter[n_minus_1_gram + (word,)]
        n_minus_1_gram_frequency = self.n_minus_1_gram_counter[n_minus_1_gram]

        probability = (n_gram_frequency + 1) / (n_minus_1_gram_frequency +
                                                self.vocab_size)

        return probability


ngram = NGRAM(n=3, words=words)
print("The generated sentence without laplace smoothing is:")
print("EMMA BY", end=" ")

ngram.generate_sentence(length=10, n_minus_1_words="emma by")

print()
print("The generated sentence with laplace smoothing is:")
print("EMMA BY", end=" ")

smooth_ngram = SmoothNGRAM(n=3, words=words)

smooth_ngram.generate_sentence(length=10, n_minus_1_words="emma by")

print("Comparing perplexities of different ngram models with smoothing:\n")
n = []
perplexities = []

for i in range(2, 10):
    ngram_model = SmoothNGRAM(n=i, words=words)
    perplexity = ngram_model.perplexity(test_words)
    print(f"For n = {i}, Perplexity is {perplexity}")
    n.append(i)
    perplexities.append(perplexity)
