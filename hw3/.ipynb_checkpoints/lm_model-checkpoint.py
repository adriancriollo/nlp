from collections import Counter
import numpy as np
import math

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
    for i in range(len(tokens) - n):
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
        self.ngram_dict = {}
        self.ngram_probabilities = {}
        self.ngram_num = n_gram

    def train(self, tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          tokens (list): tokenized data to be trained on as a single list
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """
        ngrams = create_ngrams(tokens, self.ngram_num)
        self.ngram_dict = Counter(ngrams)
        single_occurence = []
        for ngram in self.ngram_dict:
            if self.ngram_dict[ngram] == 1:
                single_occurence.append(ngram)
        for occurence in single_occurence:
            del self.ngram_dict[occurence]
            self.ngram_dict["<UNK>"] += 1 if "<UNK>" in self.ngram_dict else 1

        total_ngrams = sum(self.ngram_dict.values()) + len(self.ngram_dict)
        # Probability with laplace smoothing
        for ngram in self.ngram_dict:
            self.ngram_probabilities[ngram] = (
                self.ngram_dict[ngram] + 1
            ) / total_ngrams
        pass

    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
        Args:
          sentence_tokens (list): a tokenized sequence to be scored by this model

        Returns:
          float: the probability value of the given tokens for this model
        """
        sentence_probability = 1
        if self.ngram_num == 1:
            sentence_ngrams = sentence_tokens
        else:
            sentence_ngrams = create_ngrams(sentence_tokens, self.ngram_num)
        total_ngrams = sum(self.ngram_dict.values()) + len(self.ngram_dict)
        for token in sentence_ngrams:
            if token in self.ngram_probabilities:
                sentence_probability *= self.ngram_probabilities[token]
            else:
                sentence_probability *= 1 / total_ngrams
        return sentence_probability

    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          list: the generated sentence as a list of tokens
        """
        sentence = ["<s>"]
        while True:
            context = tuple(sentence[-(self.ngram_num - 1) :])
            next_token = self.sample_next_token(context)
            sentence.append(next_token)
            if next_token == "</s>":
                break
        return sentence[1:-1]

    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
        Args:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing lists of strings, one per generated sentence
        """
        # PROVIDED
        return [self.generate_sentence() for i in range(n)]

    def sample_next_token(self, context):
        possible_next_tokens = {
            ngram[-1]: prob
            for ngram, prob in self.ngram_probabilities.items()
            if ngram[:-1] == context
        }

        # Sample based on the probabilities of possible next tokens
        next_token = random.choices(
            list(possible_next_tokens.keys()),
            weights=list(possible_next_tokens.values()),
        )[0]

        return next_token

    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
        Args:
          sequence (list): a tokenized sequence to be evaluated for perplexity by this model

        Returns:
          float: the perplexity value of the given sequence for this model
        """
        # STUDENTS IMPLEMENT
        pass


# not required
if __name__ == "__main__":
    testString = "if having a main is helpful to you, do whatever you want here, but please don't produce too much output"
    print(tokenize_line(testString, 4, False, "if", "output"))
    print(
        "if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)"
    )
