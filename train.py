
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import label_binarize
from collections import defaultdict
import numpy as np
import toolz

PCT_TEST_DATA = 0.20
VOCAB_SIZE = 2**12
MAXLEN_INPUT = 25
WORD_EMBED_SIZE = 100
CONTEXT_SIZE = 150
BATCH_SIZE = 32
MAX_EPOCHS = 100

LSTM_OUTPUT_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.2
DENSE_DROPOUT = 0.2

USE_GLOVE = False
USE_ATTENTION = False

META_BATCH_SIZE = 4 * 1024 * BATCH_SIZE

def train():
    """ Train keras model and save to disk in models/latest.bin """
    print("Loading development data...")
    develop_data = load_develop_data()
    train_in, train_out, test_in, test_out = train_test_split(develop_data)
    print("Selected {} training examples and {} test examples...".format(
        len(train_in), len(test_in)))

    print("Fitting vocab from loaded data...")
    vocab, tokenize = vocab_for_lines(train_in + train_out + test_in + test_out)
    inverse_vocab = {v: k for k, v in vocab.items()}
    print(tokenize(train_in[0]))

    print("Transforming input and output data...")
    train_x = vectorize_data(tokenize, vocab, train_in, is_input=True)
    train_y = vectorize_data(tokenize, vocab, train_out, is_input=False)
    test_x = vectorize_data(tokenize, vocab, test_in, is_input=True)
    test_y = vectorize_data(tokenize, vocab, test_out, is_input=False)

    print("Building keras seq2seq model...")
    model = build_custom_model(vocab)
    # model = build_model(vocab)

    print()
    print(model.summary())

    print("Training model...")
    train_model(model, vocab, inverse_vocab, tokenize, train_x, train_y, test_x, test_y)

    while True:
        print("Reply to what?")
        text = input('> ')
        print(decode_input_text(model, vocab, inverse_vocab, tokenize, text))


def build_one_word_output_model():
    """ Builds a model with full encoder and one-word decoder """
    from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Reshape, Flatten, \
        RepeatVector, Dropout, GRU, Bidirectional, add, Activation
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model
    from keras.activations import softmax

    question_input = Input(shape=(MAXLEN_INPUT, ), dtype='int32', name='question_input')

    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=MAXLEN_INPUT,
        weights=[load_embedding_glove_weights(vocab)] if USE_GLOVE else None,
    )

    embedded_question = shared_embedding(question_input)

    base_encoding_layer = \
        Bidirectional(LSTM(
            CONTEXT_SIZE,
            kernel_initializer='lecun_uniform',
            name='question_encoder_0',
            dropout=LSTM_OUTPUT_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_sequences=True,
        ))(embedded_question)

    last_encoding_layer = \
        LSTM(
            CONTEXT_SIZE,
            kernel_initializer='lecun_uniform',
            name='question_encoder_1',
            dropout=LSTM_OUTPUT_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_sequences=True,
        )(base_encoding_layer)

    last_encoding_layer, *context_state = \
        LSTM(
            CONTEXT_SIZE,
            kernel_initializer='lecun_uniform',
            name='question_encoder_3',
            dropout=LSTM_OUTPUT_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_sequences=False,
            return_state=True,
        )(last_encoding_layer)

    context = last_encoding_layer

    # DECODER -----------------------------------
    # ===========================================

    answer_input = Input(shape=(1, ), dtype='int32', name='answer_input')
    
    embedded_answer = shared_embedding(answer_input)

    decoder_lstm = LSTM(
        CONTEXT_SIZE,
        kernel_initializer='lecun_uniform',
        name='decoder_lstm',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
    )(concatenate([
        RepeatVector(MAXLEN_INPUT)(context),
        embedded_answer,
        # Reshape((MAXLEN_INPUT, VOCAB_SIZE))(answer_input),
    ], axis=2), context_state)

    answer_output = Dropout(DENSE_DROPOUT)(
        Dense(
            VOCAB_SIZE,
            activation='softmax',
            name='answer_output',
        )(decoder_lstm))

    model = Model(inputs=[question_input, answer_input], outputs=[answer_output])

    return model


def build_custom_model(vocab):
    from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Reshape, Flatten, \
        RepeatVector, Dropout, GRU, Bidirectional, add, Activation, Lambda, dot, multiply
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model
    from keras.activations import softmax
    from keras import backend as K

    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=1,
        name='embedding',
        weights=[load_embedding_glove_weights(vocab)] if USE_GLOVE else None,
    )

    encoder_input = Input(
        shape=(MAXLEN_INPUT, ),
        dtype='int32',
        name='encoder_input',
        batch_shape=(BATCH_SIZE, MAXLEN_INPUT),
    )
    embedded_input = shared_embedding(encoder_input)

    encoder_lstm = LSTM(
        CONTEXT_SIZE,
        name='encoder_0',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        return_state=True,
        stateful=True,
    )
    current_encoder_output, *current_encoder_state = encoder_lstm(
        Lambda(lambda x: x[:, 0:1], output_shape=(1, WORD_EMBED_SIZE))(embedded_input)
    )

    encoder_outputs = [current_encoder_output]

    for input_idx in range(1, MAXLEN_INPUT):
        encoder_input_slice = \
            Lambda(
                lambda x: x[:, input_idx:input_idx+1], 
                output_shape=(1, WORD_EMBED_SIZE),
            )(embedded_input)

        current_encoder_output, *current_encoder_state = encoder_lstm(
            encoder_input_slice,
            current_encoder_state,
        )

        encoder_outputs.append(current_encoder_output)

    # DECODER TIME!!
    # =============================================================================

    decoder_input = Input(
        shape=(MAXLEN_INPUT, ),
        dtype='int32',
        name='decoder_input_0',
        batch_shape=(BATCH_SIZE, MAXLEN_INPUT),
    )

    embedded_decoder_input = shared_embedding(decoder_input)

    decoder_lstm = LSTM(
        CONTEXT_SIZE,
        name='decoder_0',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        return_state=True,
        stateful=True,
    )

    current_decoder_state = current_encoder_state
    decoder_outputs = []


    def attend_to(decoder_state):
        """ Creates an attention that is softmax(decoder_state dot encoder_out) * encoder_out
            for time slice before this.
        """ 
        # TODO: what is the decoder state we care about?
        sub_attentions = \
            concatenate([
                dot([decoder_state[0], K.transpose(encoder_output)], axes=(0, 1))
                for encoder_output in encoder_outputs
            ], axis=1)

        print(sub_attentions.shape)

        softmaxed = \
            Dense(
                CONTEXT_SIZE * MAXLEN_INPUT,
                activation='softmax',
            )(sub_attentions)

        print(softmaxed.shape)
        
        cated = concatenate(encoder_outputs, axis=0)
        print(cated.shape)
        scaled = \
            multiply([
                cated,
                softmaxed,
            ])

        print(scaled.shape)

        return add(scaled, axis=1)

    for decoder_idx in range(0, MAXLEN_INPUT):
        decoder_input_slice = \
            Lambda(
                lambda x: x[:, decoder_idx:decoder_idx+1], 
                output_shape=(1, WORD_EMBED_SIZE),
            )(embedded_decoder_input)

        if USE_ATTENTION:
            decoder_input_slice = concatenate([
                decoder_input_slice, 
                attend_to(current_decoder_state)
            ])

        current_decoder_output, *current_decoder_state = \
            decoder_lstm(
                decoder_input_slice,
                current_decoder_state,
            )

        decoder_outputs.append(current_decoder_output)

    decoder_outputs = Reshape(
            (MAXLEN_INPUT, CONTEXT_SIZE)
        )(concatenate(decoder_outputs, axis=1))
    print('Decoder Concat Shape: ', decoder_outputs.shape)

    word_dists = \
        TimeDistributed(
            Dense(
                VOCAB_SIZE,
                activation=lambda layer: softmax(layer, axis=1),
                name='word_dist',
            )
        )(decoder_outputs)
    # print(word_dists.shape)

    model = Model(inputs=[encoder_input, decoder_input], outputs=word_dists)
    return model



def build_model(vocab):
    """ Build keras model """
    from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Reshape, Flatten, RepeatVector, \
        Dropout, GRU, Bidirectional, add, multiply
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model

    RecurrentCell = LSTM
    num_recurrent_layers = 3

    # Builds "question" encoder
    # First layer is word embedding
    question_input = Input(shape=(MAXLEN_INPUT, ), dtype='int32', name='question_input')

    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=MAXLEN_INPUT,
        name='embedding',
        weights=[load_embedding_glove_weights(vocab)] if USE_GLOVE else None,
    )

    embedded_question = shared_embedding(question_input)

    base_encoding_layer = \
        Bidirectional(RecurrentCell(
            CONTEXT_SIZE,
            name='question_encoder_0',
            dropout=LSTM_OUTPUT_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_sequences=True,
        ))(embedded_question)

    last_encoding_layer = \
        RecurrentCell(
            CONTEXT_SIZE,
            name='question_encoder_1',
            dropout=LSTM_OUTPUT_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_sequences=True,
        )(base_encoding_layer)

    last_encoding_layer, *context_state = \
        RecurrentCell(
            CONTEXT_SIZE,
            name='question_encoder_2',
            dropout=LSTM_OUTPUT_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT,
            return_sequences=False,
            return_state=True,
        )(last_encoding_layer)

    context = last_encoding_layer

    # This is encoded recurrently into a sentence/context vector

    # Next create answer encoder
    answer_input = Input(shape=(MAXLEN_INPUT, ), dtype='int32', name='answer_input')
    
    embedded_answer = shared_embedding(answer_input)

    # attention = multiply([
    #     Dense(
    #         MAXLEN_INPUT * WORD_EMBED_SIZE,

    #     )(concatenate()),

    # ])

    decoder_lstm = RecurrentCell(
        CONTEXT_SIZE,
        name='decoder_lstm',
        return_sequences=True,
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
    )(concatenate([
        RepeatVector(MAXLEN_INPUT)(context), # Replace with attention
        embedded_answer,
    ], axis=2), context_state)

    answer_output = Dropout(DENSE_DROPOUT)(TimeDistributed(
        Dense(
            VOCAB_SIZE,
            activation='softmax',
        ),
        name='answer_output',)(decoder_lstm))

    model = Model(inputs=[question_input, answer_input], outputs=[answer_output])

    return model


def train_model(model, vocab, inverse_vocab, tokenize, train_x, train_y, test_x, test_y):
    from keras import callbacks, optimizers
    # early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', patience=0)
    tb_callback = callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=30,
        batch_size=BATCH_SIZE,
        write_graph=True,
        write_grads=True,
        write_images=False,
        embeddings_freq=30,
        embeddings_layer_names=['embedding'],
        embeddings_metadata=None,
    )
    # adam = optimizers.Adam(lr=0.5, clipnorm=5.0)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    START = vocab['<START>']
    PAD = vocab['<PAD>']

    TEST_INPUT = [
        "__romance When did you start drinking again?",
        "__thriller So you're from around here?",
    ]

    print('Model input shapes:')
    for input in model.inputs:
        print(input.shape)

    print('Model output shapes:')
    for output in model.outputs:
        print(output.shape)

    for text in TEST_INPUT:
        print(text)
        print(tokenize(text))
        print([vocab[tok] for tok in tokenize(text)])
        print([inverse_vocab[vocab[tok]] for tok in tokenize(text)])

    try:
        for epoch in range(MAX_EPOCHS):
            print("Training epoch {}".format(epoch))
            N = META_BATCH_SIZE #0000000000

            for text in TEST_INPUT:
                print('> ', text)
                print(decode_input_text(model, vocab, inverse_vocab, tokenize, text))

            answers_in_train = np.hstack([
                np.ones((len(train_x), 1), dtype='int32') * START,
                train_y[:, :-1]
            ])[:N]
            answers_out_train = (np.hstack([
                train_y[:, :-1],
                np.ones((len(train_x), 1), dtype='int32') * PAD,
            ])).reshape((len(train_x), MAXLEN_INPUT, 1))[:N]

            answers_in_test = np.hstack([
                np.ones((len(test_x), 1), dtype='int32') * START,
                test_y[:, :-1]
            ])[:N]
            answers_out_test = (np.hstack([
                test_y[:, :-1],
                np.ones((len(test_x), 1), dtype='int32') * PAD,
            ])).reshape((len(test_x), MAXLEN_INPUT, 1))[:N]

            model.fit(
                x=[train_x.reshape((len(train_x), MAXLEN_INPUT))[:N], answers_in_train],
                y=answers_out_train,
                epochs=1,
                batch_size=BATCH_SIZE,
                # callbacks=[tb_callback],
                # validation_split=0.1,
            )

            test_n = BATCH_SIZE * int(min([len(test_x), N]) / BATCH_SIZE)
            score = model.evaluate(
                x=[test_x.reshape((len(test_x), MAXLEN_INPUT))[:test_n], answers_in_test[:test_n]],
                y=(answers_out_test[:test_n]),
                batch_size=BATCH_SIZE,
            )

            print('\n')
            print('score:', score)


    except KeyboardInterrupt:
        print("Caught keyboard interrupt, halting training...")
        return model



def split_vec(vec):
    return np.split(vec, MAXLEN_INPUT, axis=1)


def decode_input_text(model, vocab, inverse_vocab, tokenize, text):
    result_tokens = [vocab['<START>']]
    for i in range(MAXLEN_INPUT):
        answer_vec = (result_tokens + [vocab['<PAD>']] * MAXLEN_INPUT)[:MAXLEN_INPUT]
        input_vec = [vectorize_data(tokenize, vocab, [text], is_input=True), 
                     np.array(answer_vec).reshape((1, MAXLEN_INPUT))]
        prediction = model.predict([np.vstack([vec]*BATCH_SIZE) for vec in input_vec])[0].reshape(MAXLEN_INPUT, VOCAB_SIZE)
        tokens = list(prediction.argmax(axis=1))
        # print([inverse_vocab[idx] for idx in tokens])
        result_tokens.append(tokens[i])
    return ' '.join([inverse_vocab[idx] for idx in result_tokens])



def binarize_labels(labels, word_idx=None):
    # return label_binarize(labels[:, word_idx], classes=list(range(VOCAB_SIZE)), sparse_output=True)
    from keras.utils import np_utils
    if word_idx:
        return np_utils.to_categorical(labels[:, word_idx], num_classes=VOCAB_SIZE)
    else:
        return np.array([np_utils.to_categorical(row, num_classes=VOCAB_SIZE) for row in labels])


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


def vectorize_data(tokenize, vocab, lines, is_input=False):
    vectors = []

    for tokens in map(tokenize, lines):
        vector = []

        for idx in range(MAXLEN_INPUT):
            if idx < len(tokens):
                vector.append(vocab[tokens[idx]])
            else:
                vector.append(vocab['<PAD>'])

        if is_input:
            vectors.append(vector[::-1])
        else:
            vectors.append(vector)

    return np.array(vectors)


def vocab_for_lines(lines):
    """ Build vocab for provided training/test data """
    vectorizer = CountVectorizer(max_features=VOCAB_SIZE - 4, stop_words=[], token_pattern='\w+')
    vectorizer.fit(lines)
    vocab = vectorizer.vocabulary_
    vocab['<PAD>'] = len(vocab)
    vocab['<START>'] = len(vocab)
    vocab['<END>'] = len(vocab)
    vocab['<OOV>'] = len(vocab)

    return defaultdict(lambda: vocab['<OOV>'], vocab.items()), vectorizer.build_analyzer()


def load_embedding_glove_weights(vocab):
    print("Loading glove weights...")
    with open('glove/glove.twitter.27B.{}d.txt'.format(WORD_EMBED_SIZE)) as infile:
        line_iter = (line.strip().split(' ') for line in infile)
        word_embeds = {word: list(map(float, rest)) for word, *rest in line_iter if word in vocab}

    inverse_vocab = {v: k for k, v in vocab.items()}
    sorted_words = [inverse_vocab[idx] for idx in range(VOCAB_SIZE)]
    def default():
        return list(np.random.rand(WORD_EMBED_SIZE))

    results = []

    hits = 0
    for word in sorted_words:
        if word in word_embeds:
            hits += 1
            results.append(word_embeds[word])
        else:
            results.append(default())

    array_result = np.array(results, dtype='float32')
    print("{} glove hits out of {} words in vocab ({}%)".format(
        hits, VOCAB_SIZE, 100 * hits / VOCAB_SIZE))
    assert array_result.shape == (VOCAB_SIZE, WORD_EMBED_SIZE)
    return array_result


