import sys
import collections
from collections import defaultdict
import math
import string
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall B 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba

Code completed by Manxueying Li
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, 
    where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    n_grams = []
    starts = ['START'] * (n - 1)
    stop = ['STOP']
    sequence = starts + sequence + stop
    for i in range(len(sequence) - n + 1):
        n_grams.append(tuple(sequence[i:i + n]))
    return n_grams


def cnt_unigram(corpusfile, lexicon=None):
        generator = corpus_reader(corpusfile, lexicon=lexicon)
        unicnttotal = 0
        unigram = []
        for sentence in generator:
            unigram += get_ngrams(sentence, 1)
        unigramcounts = dict(collections.Counter(unigram))
        # print(unigramcounts)
        for key, val in unigramcounts.items():
            unicnttotal += val
        # print("unicnttotal", unicnttotal)
        return unicnttotal


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = {}  # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        # Your code here
        unigramcounts = []
        bigramcounts = []
        trigramcounts = []

        for sentence in corpus:
            unigramcounts += get_ngrams(sentence, 1)

            bigramcounts += get_ngrams(sentence, 2)

            trigramcounts += get_ngrams(sentence, 3)

        self.unigramcounts = dict(collections.Counter(unigramcounts))
        self.bigramcounts = dict(collections.Counter(bigramcounts))
        self.trigramcounts = dict(collections.Counter(trigramcounts))

        self.unicnttotal = 0
        for key, val in self.unigramcounts.items():
            self.unicnttotal += val

        self.unicnt = self.unicnttotal
        # return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram in self.trigramcounts and trigram[:-1] in self.bigramcounts:
            return self.trigramcounts[trigram] / self.bigramcounts[trigram[:-1]]
        else:
            return 0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram in self.bigramcounts and bigram[0] in self.unigramcounts:
            return self.bigramcounts[bigram] / self.unigramcounts[bigram[0]]
        else:
            return 0

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        if unigram in self.unigramcounts:
            return self.unigramcounts[unigram] / self.unicnttotal
        else:
            return 0

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        # print(trigram, trigram[1:], trigram[-1])
        return lambda1 * self.raw_trigram_probability(trigram)\
            + lambda2 * self.raw_bigram_probability(trigram[1:])\
            + lambda3 * self.raw_unigram_probability(trigram[-1])

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        logprob = 0
        for i in trigrams:
            if self.smoothed_trigram_probability(i):
                # print(self.smoothed_trigram_probability(i))
                logprob += math.log2(self.smoothed_trigram_probability(i))
        # print(logprob)
        return logprob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        summ = 0
        for si in corpus:
            summ += self.sentence_logprob(si)
        # print("2 power:", summ, self.unicnt)
        return 2 ** (-summ / self.unicnt)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)#high
    model2 = TrigramModel(training_file2)#low

    total = 0
    correct = 0
    
    #high
    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(
            os.path.join(testdir1, f), model1.lexicon))
        # ..
        pp2 = model2.perplexity(corpus_reader(
            os.path.join(testdir1, f), model2.lexicon))
        if pp1 < pp2:
            correct += 1
        total += 1
    #low
    for f in os.listdir(testdir2):
        pp2 = model2.perplexity(corpus_reader(
            os.path.join(testdir2, f), model2.lexicon))
        # ..
        pp1 = model1.perplexity(corpus_reader(
            os.path.join(testdir2, f), model1.lexicon))
        if pp1 > pp2:
            correct += 1
        total += 1

    return correct / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    
    dev_corpus_train = corpus_reader(sys.argv[1], model.lexicon)
    pp_train = model.perplexity(dev_corpus_train)
    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    # print(model.unigramcounts[('the',)])
    # print("train cnt:", model.unicnttotal)
    print("pp train:", pp_train)

    
    # print(dev_corpus[:3])
    model.unicnt = cnt_unigram(sys.argv[2])
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # print("test cnt:", model.unicnt)
    pp = model.perplexity(dev_corpus)
    print("pp test:", pp)
    

    # Essay scoring experiment:
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 
        'hw1_data/ets_toefl_data/train_low.txt', 'hw1_data/ets_toefl_data/test_high', 
        'hw1_data/ets_toefl_data/test_low')
    print("Accuracy: ", acc)
