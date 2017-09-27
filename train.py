
import os
import random

import numpy as np
from bpe.encoder import Encoder

from model import Seq2SeqConfig

PCT_TEST_DATA = 0.20
VOCAB_SIZE = 2**12
MAX_LEN_INPUT = 25
WORD_EMBED_SIZE = 100
CONTEXT_SIZE = 150
BATCH_SIZE = 32
MAX_EPOCHS = 100

LSTM_OUTPUT_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.2
DENSE_DROPOUT = 0.2

USE_GLOVE = False
USE_ATTENTION = False

META_BATCH_SIZE = 1024 * BATCH_SIZE


START = '<start/>'
END = '<end/>'
OOV = '<oov/>'

SEQ2SEQ_PARAMS = Seq2SeqConfig(
    message_len=MAX_LEN_INPUT,
    batch_size=32,
    context_size=100,
    embed_size=100,
    use_cuda=False,
    vocab_size=2**13,
)


def train():
    """ Train keras model and save to disk in models/latest.bin """
    print("Loading development data...")
    develop_data = load_develop_data()
    train_in, train_out, test_in, test_out = train_test_split(develop_data)
    print("Selected {} training examples and {} test examples...".format(
        len(train_in), len(test_in)))

    print("Fitting vocab from loaded data...")
    encoder = encoder_for_lines(train_in + train_out + test_in + test_out)

    print("Transforming input and output data...")    
    train_x = encode_data(encoder, train_in, is_input=True)
    train_y = encode_data(encoder, train_out, is_input=False)
    test_x = encode_data(encoder, test_in, is_input=True)
    test_y = encode_data(encoder, test_out, is_input=False)
    encoder.mute()

    print("Building pytorch seq2seq model...")
    from gru_model import build_model
    model = build_model(SEQ2SEQ_PARAMS, encoder.word_vocab[START])


    train_model(model, train_x, train_y, test_x, test_y)

    while True:
        print("Reply to what?")
        text = input('> ')
        print(get_model_response(model, encoder, text))


def train_model(model, train_x, train_y, test_x, test_y):
    """ Train provided model """
    model.train_epoch(train_x, train_y)

def get_model_response(model, encoder, text):
    """ Encode text, evaluate model, return response """


def train_test_split(develop_data):
    test_movie_ids = set(random.sample(develop_data.keys(), int(len(develop_data)*PCT_TEST_DATA)))

    train_in, train_out = [], []
    test_in, test_out = [], []
    for movie_id, (input_data, output_data) in develop_data.items():
        if movie_id in test_movie_ids:
            test_in.extend(input_data)
            test_out.extend(output_data)
        else:
            train_in.extend(input_data)
            train_out.extend(output_data)

    return train_in, train_out, test_in, test_out


def load_develop_data(path='data/develop/'):
    fnames = [fname for fname in os.listdir(path) if fname.startswith('m')]
    movie_ids = set(fname.split('_')[0] for fname in fnames)

    def read_movie(movie_id):
        with open('{}/{}_in.txt'.format(path.rstrip('/'), movie_id), 'rb') as infile:
            in_lines = [line.decode('utf-8', 'ignore').strip() for line in infile]
        with open('{}/{}_out.txt'.format(path.rstrip('/'), movie_id), 'rb') as infile:
            out_lines = [line.decode('utf-8', 'ignore').strip() for line in infile]
        return in_lines, out_lines

    return {movie_id: read_movie(movie_id) for movie_id in movie_ids}


def encode_data(encoder, text, is_input=False):
    """ Encode data using provided encoder, for use in training/testing """
    return np.array(list(encoder.transform(text, reverse=is_input, fixed_length=MAX_LEN_INPUT)),
                    dtype='int32')


def encoder_for_lines(lines):
    """ Calculate BPE encoder for provided lines of text """
    encoder = Encoder(vocab_size=VOCAB_SIZE, required_tokens=[START])
    encoder.fit(lines)
    encoder.save('latest_encoder.json')
    return encoder
