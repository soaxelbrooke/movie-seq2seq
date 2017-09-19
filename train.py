
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np

PCT_TEST_DATA = 0.20
VOCAB_SIZE = 2**13
MAXLEN_INPUT = 25
WORD_EMBED_SIZE = 100
SENTENCE_EMBED_SIZE = 200
BATCH_SIZE = 8


def train():
    """ Train keras model and save to disk in models/latest.bin """
    print("Loading development data...")
    develop_data = load_develop_data()
    train_in, train_out, test_in, test_out = train_test_split(develop_data)
    print("Selected {} training examples and {} test examples...".format(
        len(train_in), len(test_in)))

    print("Fitting vocab from loaded data...")
    vocab, tokenize = vocab_for_lines(train_in + train_out + test_in + test_out)

    print("Building keras seq2seq model...")
    model = build_model()


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


def build_model():
    """ Build keras model """
    from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, \
        concatenate
    from keras.optimizers import Adam, SGD
    from keras.models import Model
    from keras.models import Sequential
    from keras.layers import Activation, Dense
    from keras.callbacks import EarlyStopping
    from keras.preprocessing import sequence

    # Builds "question" encoder
    # First layer is word embedding
    question_input = Input(shape=(MAXLEN_INPUT, ), dtype='int32', name='question_input')

    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=MAXLEN_INPUT,
    )

    embedded_question = shared_embedding(question_input)

    # This is encoded recurrently into a sentence/context vector
    encoded_question = LSTM(
        SENTENCE_EMBED_SIZE,
        kernel_initializer='lecun_uniform',
        name='question_encoder',
    )(embedded_question)

    # Next create answer encoder
    answer_input = Input(shape=(MAXLEN_INPUT, ), dtype='int32', name='answer_input')
    
    embedded_answer = shared_embedding(answer_input)

    encoded_answer = LSTM(
        SENTENCE_EMBED_SIZE, 
        kernel_initializer='lecun_uniform', 
        name='answer_encoder'
    )(embedded_answer)

    merge_layer = concatenate([encoded_question, encoded_answer], axis=1)

    next_word_dense = Dense(int(VOCAB_SIZE / 2), activation='relu')(merge_layer)
    next_word = Dense(VOCAB_SIZE, activation='softmax')(next_word_dense)

    model = Model(inputs=[question_input, answer_input], outputs=[next_word])

    print("Built keras {} GB keras model.".format(get_model_memory_usage(model)))
    return model


def get_model_memory_usage(model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    total_memory = 4 * BATCH_SIZE * (shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = round(total_memory / (1024 ** 3), 3)
    return gbytes


def vocab_for_lines(lines):
    """ Build vocab for provided training/test data """
    vectorizer = CountVectorizer(max_features=VOCAB_SIZE - 2)
    vectorizer.fit(lines)
    vocab = vectorizer.vocabulary_
    vocab['<PAD>'] = len(vocab)
    vocab['<OOV>'] = len(vocab)
    return defaultdict(lambda: vocab['<OOV>'], vocab.items()), vectorizer.build_analyzer()
