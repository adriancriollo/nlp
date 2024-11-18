# for word tokenization
import nltk
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
import numpy as np

nltk.download('punkt')

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"


def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   space_char: str = ' ',
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
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
    space_char (str): if by_char is True, use this character to separate to replace spaces
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  # PROVIDED
  inner_pieces = None
  if by_char:
    line = line.replace(' ', space_char)
    inner_pieces = list(line)
  else:
    # otherwise use nltk's word tokenizer
    inner_pieces = nltk.word_tokenize(line)

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def read_file_spooky(datapath, ngram, by_character=False):
    '''Reads and Returns the "data" as list of list (as shown above)'''
    data = []
    with open(datapath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # THIS IS WHERE WE GET CHARACTERS INSTEAD OF WORDS
            # replace spaces with underscores
            data.append(tokenize_line(row['text'].lower(), ngram, by_char=by_character, space_char="_"))
    return data


# Feel free to add more functions here as you would like!

def get_vocab_and_chars_size(texts):
    char_vocab = set(''.join(texts))
    char_vocab_size = len(char_vocab)
    word_vocab = set()
    for text in texts:
        word_vocab.update(text.split())
    word_vocab_size = len(word_vocab)
    
    return char_vocab_size, word_vocab_size

def preprocess_text(line):
    punctuation = ".,!?;:\"'()[]{}<>-_"
    for char in punctuation:
        line = line.replace(char, '')
    return line.lower()

def create_model(input_shape, vocab_size):
    """
    Create a feedforward neural language model.
    
    Parameters:
    - input_shape (tuple): Shape of the input (NGRAM-1, EMBEDDINGS_SIZE).
    - vocab_size (int): Size of the output vocabulary.
    
    Returns:
    - model (Sequential): Compiled Keras model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=input_length),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, steps_per_epoch, epochs=1):
    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
    return history

def generate_seq(model, tokenizer, seed, max_length=50, end_token='</s>'):
    sentence = tokenizer.sequences_to_texts([seed])[0].split()

    for _ in range(max_length):
        X_input = np.array(seed).reshape(1, len(seed))  # Correct input shape for model with Embedding layer
        
        predicted_probabilities = model.predict(X_input, verbose=0)
        next_token_idx = np.argmax(predicted_probabilities)
        
        next_token = tokenizer.index_word[next_token_idx]
        sentence.append(next_token)
        
        if next_token == end_token:
            break
        
        seed = seed[1:] + [next_token_idx]

    return ' '.join(sentence)