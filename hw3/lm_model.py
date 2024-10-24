from collections import Counter
import numpy as np
import math
import random

"""
CS 4120, Fall 2024
Homework 3 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS


def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
    Args:
      tokens (list): a list of tokens as strings
      n (int): the length of n-grams to create

    Returns:
      list: list of tuples of strings, each tuple being one of the individual n-grams
    """
    tupleList = []
    for i in range(len(tokens) - n + 1):
        tupleList.append(tuple((tokens[i : i + n])))
    return tupleList


def read_file(path: str) -> list:
    """
    Reads the contents of a file in line by line.
    Args:
      path (str): the location of the file to read

    Returns:
      list: list of strings, the contents of the file
    """
    # PROVIDED
    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents


def tokenize_line(
    line: str,
    ngram: int,
    by_char: bool = True,
    sentence_begin: str = SENTENCE_BEGIN,
    sentence_end: str = SENTENCE_END,
):
    """
    Tokenize a single string. Glue on the appropriate number of
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
      line (str): text to tokenize
      ngram (int): ngram preparation number
      by_char (bool): default value True, if True, tokenize by character, if
        False, tokenize by whitespace
      sentence_begin (str): sentence begin token value
      sentence_end (str): sentence end token value

    Returns:
      list of strings - a single line tokenized
    """
    # PROVIDED
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
        # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = (
            ([sentence_begin] * (ngram - 1))
            + inner_pieces
            + ([sentence_end] * (ngram - 1))
        )
    # always count the unigrams
    return tokens


def tokenize(
    data: list,
    ngram: int,
    by_char: bool = True,
    sentence_begin: str = SENTENCE_BEGIN,
    sentence_end: str = SENTENCE_END,
):
    """
    Tokenize each line in a list of strings. Glue on the appropriate number of
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
      data (list): list of strings to tokenize
      ngram (int): ngram preparation number
      by_char (bool): default value True, if True, tokenize by character, if
        False, tokenize by whitespace
      sentence_begin (str): sentence begin token value
      sentence_end (str): sentence end token value

    Returns:
      list of strings - all lines tokenized as one large list
    """
    # PROVIDED
    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram):
        """Initializes an untrained LanguageModel
        Args:
          n_gram (int): the n-gram order of the language model to create
        """
        self.n_gram_num = n_gram
        self.ngram_dict = Counter()
        self.context_dict = Counter()
        self.vocab = set()

    def train(self, tokens: list, verbose: bool = True) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          tokens (list): tokenized data to be trained on as a single list
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """
        for token in tokens:
            self.vocab.add(token)
        ngrams = create_ngrams(tokens, self.n_gram_num)
        for ngram in ngrams:
            self.ngram_dict[ngram] += 1
            context = ngram[:-1]
            self.context_dict[context] += 1
        allTokensCount = Counter(tokens)
        single_occurence_tokens = []
        for token in allTokensCount:
            if allTokensCount[token] == 1:
                single_occurence_tokens.append(token)
        if len(single_occurence_tokens) > 0:
            self.vocab.add(UNK)
        for token in single_occurence_tokens:
            self.vocab.remove(token)
        updated_ngram_dict = Counter()
        updated_context_dict = Counter()
        for ngram, count in self.ngram_dict.items():
            updated_ngram = tuple(
                UNK if token in single_occurence_tokens else token for token in ngram
            )
            updated_ngram_dict[updated_ngram] += count
            context_replaced = updated_ngram[:-1]
            updated_context_dict[context_replaced] += count

        self.ngram_dict = updated_ngram_dict
        self.context_dict = updated_context_dict

        # print(f"Vocab: {self.vocab}")
        # print(f"Ngram_dict: {self.ngram_dict}")
        # print(f"Context_dict: {self.context_dict}")

    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
        Args:
          sentence_tokens (list): a tokenized sequence to be scored by this model

        Returns:
          float: the probability value of the given tokens for this model
        """
        updated_sentence_tokens = []
        for token in sentence_tokens:
            if token in self.vocab:
                updated_sentence_tokens.append(token)
            else:
                updated_sentence_tokens.append(UNK)

        ngrams = create_ngrams(updated_sentence_tokens, self.n_gram_num)
        vocab_size = len(self.vocab)
        sentence_probabilty = 1
        for ngram in ngrams:
            ngram_count = self.ngram_dict[ngram]
            context = ngram[:-1]
            context_count = self.context_dict[context]
            probability = (ngram_count + 1) / (context_count + vocab_size)
            sentence_probabilty *= probability
        return sentence_probabilty

    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          list: the generated sentence as a list of tokens
        """
        sentence = ["<s>"] * (self.n_gram_num - 1)

        while True:
            context = tuple(sentence[-(self.n_gram_num - 1) :])

            possibleChoices = []
            for ngram, count in self.ngram_dict.items():
                if ngram[:-1] == context:
                    possibleChoices.append((ngram[-1], count))

            total_count = 0
            for choice in possibleChoices:
                total_count += choice[1]
            if total_count == 0:
                break

            possibleTokens = [token for token, _ in possibleChoices]
            tokenWeights = [count / total_count for _, count in possibleChoices]
            next_word = random.choices(possibleTokens, weights=tokenWeights)[0]
            sentence.append(next_word)
            if next_word == "</s>":
                break
        return sentence

    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
        Args:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing lists of strings, one per generated sentence
        """
        # PROVIDED
        return [self.generate_sentence() for i in range(n)]

    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
        Args:
          sequence (list): a tokenized sequence to be evaluated for perplexity by this model

        Returns:
          float: the perplexity value of the given sequence for this model
        """
        # STUDENTS IMPLEMENT
        sentence_probability = self.score(sequence)
        N = len(sequence)
        if sentence_probability == 0:
            return float("inf")
        return (1 / sentence_probability) ** (1 / N)


# not required
if __name__ == "__main__":
    testString = "if having a main is helpful to you, do whatever you want here, but please don't produce too much output"
    print(tokenize_line(testString, 4, False, "if", "output"))
    print(
        "if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)"
    )
