
import os
import random

import numpy as np
from bpe.encoder import Encoder
from keras import Input
from keras.engine import Model
from keras.layers import Concatenate, Embedding, LSTM, Dense, TimeDistributed, Lambda, Multiply, \
    Add, Reshape, concatenate, Dot

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

META_BATCH_SIZE = 1024 * BATCH_SIZE


START = '<start/>'
END = '<end/>'
OOV = '<oov/>'


def train():
    """ Train keras model and save to disk in models/latest.bin """
    print("Building keras seq2seq model...")
    model = build_model()

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

    print("Summary of built model:")
    print(model.summary())

    print("Training model...")
    train_model(model, encoder, train_x, train_y, test_x, test_y)

    model.save('latest_model.h5')

    while True:
        print("Reply to what?")
        text = input('> ')
        print(decode_input_text(model, encoder, text))


def build_model_2():
    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=MAXLEN_INPUT,
        name='embedding',
    )

    encoder_input = Input(
        shape=(MAXLEN_INPUT, ),
        dtype='int32',
        name='encoder_input',
        batch_shape=(BATCH_SIZE, MAXLEN_INPUT),
    )
    print(encoder_input.shape)

    embedded_input = shared_embedding(encoder_input)
    print(embedded_input.shape)

    encoder_output, encoder_state = LSTM(
        CONTEXT_SIZE,
        name='encoder_0',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        return_sequences=False,
        return_state=True,
    )(embedded_input)

    decoder_output, decoder_state = LSTM(
        CONTEXT_SIZE,
        name='decoder_0',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        return_sequences=True,
        return_state=True,
    )(encoder_input, encoder_state)

    word_dists = TimeDistributed(
        Dense(
            VOCAB_SIZE,
            activation='softmax',
            name='word_dist',
        )
    )(decoder_output)

    model = Model(inputs=[encoder_input], outputs=word_dists)
    print(model.summary())
    return model


def build_model():
    """ Shared embedding used by encoder and decoder input tokens.  GloVe weight initialization is
        optional, but it was found to not impact the results meaningfully or increase the speed of
        training.
    """
    # Shape of embeddings are (N, WORD_EMBED_SIZE)
    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=1,
        name='embedding',
    )

    # =============================================================================================
    # ====================================  ENCODER TIME!  ========================================
    # =============================================================================================

    # Encoder inputs - uses token IDs, which are turned into embeddings via the embedding layer 
    # above.  Shape of encoder inputs are (MAXLEN_INPUT, )
    encoder_input = Input(
        shape=(MAXLEN_INPUT, ),
        dtype='int32',
        name='encoder_input',
        batch_shape=(BATCH_SIZE, MAXLEN_INPUT),
    )

    # Embedded version of the encoder input.  This is all MAXLEN_INPUT tokens, and will be split
    # for each "invocation" of the first encoder LSTM layer.
    embedded_input = shared_embedding(encoder_input)

    # Defines the encoder LSTM that is hared for each input token
    encoder_lstm = LSTM(
        CONTEXT_SIZE,
        name='encoder_0',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        return_state=True,
        stateful=True,
    )

    # First encoder LSTM invocation doesn't have any initial recurrent state, so it must be
    # invoked once outside the loop.
    current_encoder_output, *current_encoder_state = encoder_lstm(
        Lambda(
            lambda x: x[:, 0:1], 
            output_shape=(1, WORD_EMBED_SIZE)
        )(embedded_input)
    )

    encoder_outputs = []

    for input_idx in range(0, MAXLEN_INPUT):
        # We have to use a Lambda layer here to slice the (MAXLEN_INPUT, WORD_EMBED_SIZE)-shaped
        # input tensor into (1, WORD_EMBED_SIZE)-shaped tensors for each LSTM invocation.  We have
        # to tell Keras the shape of the result, since it can't be inferred from the lambda inside.
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

    # At this point, our encoder is complete.  The `current_encoder_state` is two tensors that
    # represent the LSTM state after the last token has been fed in, which we need to intiialize
    # the decoder with.

    # If we were going to add attention, we would build a list of `current_encoder_out` for each
    # encoder step, and make an attention unit for each decoder LSTM input based on all encoder
    # outputs (not states) and the previous decoder state.  For this implementation, they are
    # not used, since the context vector is passed as the LSTM state.

    # =============================================================================================
    # ====================================  DECODER TIME!  ========================================
    # =============================================================================================

    # Input for the decoder, essentially the same as the encoder input above.  In training, the 
    # whole decoder token string is fed.  For test decoding of strings, zeros are fed for future 
    # tokens, and the model is re-evaluated for each token to be generated.  In real applications,
    # you'd want to re-wire the network so that each LSTM output took the result of the prior 
    # invocation of the LSTM.

    # Inputs, embedding, and decoder LSTM creation are unchanged from the encoder stage.

    decoder_inputs = Input(
        shape=(MAXLEN_INPUT, ),
        dtype='int32',
        name='decoder_inputs_0',
        batch_shape=(BATCH_SIZE, MAXLEN_INPUT),
    )

    embedded_decoder_input = shared_embedding(decoder_inputs)

    decoder_lstm = LSTM(
        CONTEXT_SIZE,
        name='decoder_0',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        return_state=True,
        stateful=True,
    )

    # Set initial decoder state from encoder, and prepare to aggregate decoder LSTM outputs.
    # These decoder outputs will be put through a dense softmax layer for sampling into tokens.

    current_decoder_state = current_encoder_state
    decoder_outputs = []

    attn_dot = Dot(axes=(2, 1), normalize=True)
    attn_cat = Concatenate(axis=1, input_shape=(1,), name='attn_cat')
    attn_softmax = Dense(MAXLEN_INPUT, input_shape=(MAXLEN_INPUT, ), activation='softmax',
                         name='attn_softmax')
    attn_multiply = Multiply(name='attn_multiply')
    attn_sum = Add(input_shape=(CONTEXT_SIZE, ), name='attn_sum')

    def attend_to(decoder_state):
        # Dot product result shape: (BATCH_SIZE, 1) per decoder input
        _dots = [
            attn_dot([
                Reshape((1, CONTEXT_SIZE))(decoder_state),
                Reshape((CONTEXT_SIZE, 1))(encoder_outputs[idx]),
            ])
            for idx in range(MAXLEN_INPUT)
        ]
        # Concatenation for the attention softmax, new shape (BATCH_SIZE, MAXLEN_INPUT)
        _cat = Reshape((MAXLEN_INPUT,))(attn_cat(_dots))
        _softmax = attn_softmax(_cat)
        # Softmax provides weights per encoder output, need to split with lambda
        _multiplies = [
            attn_multiply([
                encoder_outputs[idx],
                Lambda(
                    lambda x: x[:, idx:idx+1],
                    output_shape=(MAXLEN_INPUT, CONTEXT_SIZE),
                )(_softmax)
            ])
            for idx in range(MAXLEN_INPUT)
        ]

        _sum = attn_sum(_multiplies)
        # Lambda required here to show Keras the result layer shape
        return Lambda(
                    lambda x: x,
                    output_shape=(1, CONTEXT_SIZE),
                )(_sum)

    for decoder_idx in range(0, MAXLEN_INPUT):
        # Same slicing of inputs and passing of last decoder state into next decoder invoation that
        # we saw in the endocer above.  If attention was added, it would be concatenated with the
        # decoder input here.
        decoder_input_slice = \
            Lambda(
                lambda x: x[:, decoder_idx],
                output_shape=(1, WORD_EMBED_SIZE),
            )(embedded_decoder_input)

        if USE_ATTENTION:
            decoder_input = concatenate([
                decoder_input_slice,
                attend_to(current_decoder_state[0])
            ])
        else:
            decoder_input = decoder_input_slice

        current_decoder_output, *current_decoder_state = \
            decoder_lstm(
                Reshape((CONTEXT_SIZE + WORD_EMBED_SIZE, 1))(decoder_input),
                current_decoder_state,
            )

        # Aggregate decoder outputs for merging.
        decoder_outputs.append(current_decoder_output)

    # Reshape is required here to turn concated array from (None, MAXLEN_INPUT * CONTEXT_SIZE) into
    # shape of (None, MAXLEN_INPUT, CONTEXT_SIZE) for the time-distributed dense layer below.  This
    # allows the dense embedding-to-sampled-token layer to be shared for each output token.
    decoder_outputs = Reshape(
            (MAXLEN_INPUT, CONTEXT_SIZE)
        )(concatenate(decoder_outputs, axis=1))

    # Dense layer transforming decoder output into a softmax word-distribution.
    word_dists = \
        TimeDistributed(
            Dense(
                VOCAB_SIZE,
                activation='softmax',
                name='word_dist',
            )
        )(decoder_outputs)

    model = Model(inputs=[encoder_input, decoder_inputs], outputs=word_dists)
    return model


def train_model_epoch(model, train_x, train_y, test_x, test_y, N, START):
    # Prepend <START> to training responses
    answers_in_train = np.hstack([
        np.ones((len(train_x), 1), dtype='int32') * START,
        train_y[:, :-1]
    ])
    # We have to add an extra dimension for Keras' sparse_categorical_crossentropy
    answers_out_train = train_y.reshape((len(train_x), MAXLEN_INPUT, 1))

    answers_in_test = np.hstack([
        np.ones((len(test_x), 1), dtype='int32') * START,
        test_y[:, :-1]
    ])
    answers_out_test = test_y.reshape((len(test_x), MAXLEN_INPUT, 1))

    model.fit(
        x=[train_x[:N], answers_in_train[:N]],
        y=answers_out_train[:N],
        epochs=1,
        batch_size=BATCH_SIZE,
    )

    # Need to ensure our datasize for testing is evenly divisible by batch size.  This is 
    # done by default for training size, but test size is smaller, so it must be done
    # again.
    test_n = BATCH_SIZE * int(min([len(test_x), N]) / BATCH_SIZE)
    score = model.evaluate(
        x=[test_x[:test_n], answers_in_test[:test_n]],
        y=(answers_out_test[:test_n]),
        batch_size=BATCH_SIZE,
    )

    print('\nValidation score:', score)


def train_model(model, encoder, train_x, train_y, test_x, test_y):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    start_idx = encoder.word_vocab[START]

    TEST_INPUT = [
        "__romance When did you start drinking again?",
        "__crime So you're from around here?",
        "__thriller Oh god, what do we do?",
        "__romance Do you love me?",
    ]

    try:
        for epoch in range(MAX_EPOCHS):
            print("Training epoch {}".format(epoch))
            N = META_BATCH_SIZE

            print("Evaluating test examples...")
            for text in TEST_INPUT:
                print('> ', text)
                print(decode_input_text(model, encoder, text))

            train_model_epoch(model, train_x, train_y, test_x, test_y, N, start_idx)

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, halting training...")
        return model


def decode_input_text(model, encoder, text):
    """ Horrible hack of a function to test responses to arbitrary strings. Don't try this at home,
        kids.
    """
    result_tokens = [encoder.word_vocab[START]]
    for i in range(MAXLEN_INPUT):
        answer_vec = (result_tokens + 
                      [encoder.bpe_vocab[encoder.PAD]] * MAXLEN_INPUT)[:MAXLEN_INPUT]
        input_vec = [encode_data(encoder, [text], is_input=True), 
                     np.array(answer_vec).reshape((1, MAXLEN_INPUT))]
        prediction = model.predict([np.vstack([vec]*BATCH_SIZE) 
                                    for vec in input_vec])[0].reshape(MAXLEN_INPUT, VOCAB_SIZE)
        tokens = list(prediction.argmax(axis=1))
        result_tokens.append(tokens[i])
    return list(encoder.inverse_transform([result_tokens]))


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
    return np.array(list(encoder.transform(text, reverse=is_input, fixed_length=MAXLEN_INPUT)),
                    dtype='int32')


def encoder_for_lines(lines):
    """ Calculate BPE encoder for provided lines of text """
    encoder = Encoder(vocab_size=VOCAB_SIZE, required_tokens=[START])
    encoder.fit(lines)
    encoder.save('latest_encoder.json')
    return encoder
