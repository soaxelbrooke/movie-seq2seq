
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import label_binarize
from collections import defaultdict
import numpy as np
import toolz
from bpe.encoder import Encoder

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

META_BATCH_SIZE = 1024 * BATCH_SIZE


START = '<start/>'
PAD = '<pad/>'
END = '<end/>'
OOV = '<oov/>'


def train():
    """ Train keras model and save to disk in models/latest.bin """
    print("Loading development data...")
    develop_data = load_develop_data()
    train_in, train_out, test_in, test_out = train_test_split(develop_data)
    print("Selected {} training examples and {} test examples...".format(
        len(train_in), len(test_in)))

    print("Fitting vocab from loaded data...")
    encoder = encoder_for_lines(train_in + train_out + test_in + test_out)
    # vocab, tokenize = vocab_for_lines(train_in + train_out + test_in + test_out)
    # inverse_vocab = {v: k for k, v in vocab.items()}
    # print(tokenize(train_in[0]))
    print(train_in[0:1])
    print(list(encoder.transform(train_in[0:1])))
    print(list(encoder.inverse_transform(encoder.transform(train_in[0:1]))))

    print("Transforming input and output data...")    
    train_x = encode_data(encoder, train_in, is_input=True)
    train_y = encode_data(encoder, train_out, is_input=False)
    test_x = encode_data(encoder, test_in, is_input=True)
    test_y = encode_data(encoder, test_out, is_input=False)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)

    # train_x = vectorize_data(tokenize, vocab, train_in, is_input=True)
    # train_y = vectorize_data(tokenize, vocab, train_out, is_input=False)
    # test_x = vectorize_data(tokenize, vocab, test_in, is_input=True)
    # test_y = vectorize_data(tokenize, vocab, test_out, is_input=False)
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    print("Building keras seq2seq model...")
    # model = build_model(vocab)
    model = build_model(None)

    print("Summary of built model:")
    print(model.summary())

    print("Training model...")
    # train_model(model, vocab, inverse_vocab, tokenize, train_x, train_y, test_x, test_y)
    train_model_2(model, encoder, train_x, train_y, test_x, test_y)

    model.save('latest_model.h5')

    while True:
        print("Reply to what?")
        text = input('> ')
        # print(decode_input_text(model, vocab, inverse_vocab, tokenize, text))
        print(decode_input_text_2(model, encoder, text))


def build_model(vocab):
    from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Reshape, Flatten, \
        RepeatVector, Dropout, GRU, Bidirectional, add, Activation, Lambda, dot, multiply
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model
    from keras.activations import softmax
    from keras import backend as K

    # Shared embedding used by encoder and decoder input tokens.  GloVe weight initialization is
    # optional, but it was found to not impact the results meaningfully or increase the speed of
    # training.
    # Shape of embeddings are (N, WORD_EMBED_SIZE)
    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=1,
        name='embedding',
        weights=[load_embedding_glove_weights(vocab)] if USE_GLOVE else None,
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

    for input_idx in range(1, MAXLEN_INPUT):
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

    # Set initial decoder state from encoder, and prepare to aggregate decoder LSTM outputs.
    # These decoder outputs will be put through a dense softmax layer for sampling into tokens.

    current_decoder_state = current_encoder_state
    decoder_outputs = []

    for decoder_idx in range(0, MAXLEN_INPUT):
        # Same slicing of inputs and passing of last decoder state into next decoder invoation that
        # we saw in the endocer above.  If attention was added, it would be concatenated with the
        # decoder input here.
        decoder_input_slice = \
            Lambda(
                lambda x: x[:, decoder_idx:decoder_idx+1], 
                output_shape=(1, WORD_EMBED_SIZE),
            )(embedded_decoder_input)

        current_decoder_output, *current_decoder_state = \
            decoder_lstm(
                decoder_input_slice,
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
                # activation=lambda layer: softmax(layer, axis=1),
                activation='softmax',
                name='word_dist',
            )
        )(decoder_outputs)

    model = Model(inputs=[encoder_input, decoder_input], outputs=word_dists)
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


def train_model_2(model, encoder, train_x, train_y, test_x, test_y):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    start_idx = encoder.word_vocab[START + encoder.EOW]

    TEST_INPUT = [
        "__romance When did you start drinking again?",
        "__thriller So you're from around here?",
    ]

    try:
        for epoch in range(MAX_EPOCHS):
            print("Training epoch {}".format(epoch))
            N = META_BATCH_SIZE

            print("Evaluating test examples...")
            for text in TEST_INPUT:
                print('> ', text)
                print(decode_input_text_2(model, encoder, text))

            train_model_epoch(model, train_x, train_y, test_x, test_y, N, start_idx)

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, halting training...")
        return model


def decode_input_text(model, vocab, inverse_vocab, tokenize, text):
    """ Horrible hack of a function to test responses to arbitrary strings. Don't try this at home,
        kids.
    """
    result_tokens = [vocab[START]]
    for i in range(MAXLEN_INPUT):
        answer_vec = (result_tokens + [vocab[START]] * MAXLEN_INPUT)[:MAXLEN_INPUT]
        input_vec = [vectorize_data(tokenize, vocab, [text], is_input=True), 
                     np.array(answer_vec).reshape((1, MAXLEN_INPUT))]
        prediction = model.predict([np.vstack([vec]*BATCH_SIZE) 
                                    for vec in input_vec])[0].reshape(MAXLEN_INPUT, VOCAB_SIZE)
        tokens = list(prediction.argmax(axis=1))
        result_tokens.append(tokens[i])
    return ' '.join([inverse_vocab[idx] for idx in result_tokens])


def decode_input_text_2(model, encoder, text):
    """ Horrible hack of a function to test responses to arbitrary strings. Don't try this at home,
        kids.
    """
    result_tokens = [encoder.word_vocab[START + encoder.EOW]]
    for i in range(MAXLEN_INPUT):
        answer_vec = (result_tokens + 
                      [encoder.word_vocab[PAD + encoder.EOW]] * MAXLEN_INPUT)[:MAXLEN_INPUT]
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


def vectorize_data(tokenize, vocab, lines, is_input=False):
    vectors = []

    for tokens in map(tokenize, lines):
        vector = []

        for idx in range(MAXLEN_INPUT):
            if idx < len(tokens):
                vector.append(vocab[tokens[idx]])
            else:
                vector.append(vocab[PAD])

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
    vocab[PAD] = len(vocab)
    vocab[START] = len(vocab)
    vocab[END] = len(vocab)
    vocab[OOV] = len(vocab)

    return defaultdict(lambda: vocab[OOV], vocab.items()), vectorizer.build_analyzer()


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


def encode_data(encoder, text, is_input=False):
    """ Encode data using provided encoder, for use in training/testing """
    return np.array(list(encoder.transform(text, reversed=is_input, fixed_length=MAXLEN_INPUT,
                                           padding=PAD)),
                    dtype='int32')


def encoder_for_lines(lines):
    """ Calculate BPE encoder for provided lines of text """
    encoder = Encoder(vocab_size=VOCAB_SIZE, required_tokens=[START, PAD])
    encoder.fit(lines)
    encoder.save('latest_encoder.json')
    return encoder
