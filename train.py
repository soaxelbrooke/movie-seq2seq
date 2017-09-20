
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import label_binarize
from collections import defaultdict
import numpy as np

PCT_TEST_DATA = 0.20
VOCAB_SIZE = 2**13
MAXLEN_INPUT = 25
WORD_EMBED_SIZE = 100
CONTEXT_SIZE = 200
BATCH_SIZE = 48
MAX_EPOCHS = 100

LSTM_OUTPUT_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.2
DENSE_DROPOUT = 0.2

META_BATCH_SIZE = 64 * BATCH_SIZE

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
    model = build_model(vocab)

    print("Training model...")
    train_model(model, vocab, inverse_vocab, tokenize, train_x, train_y, test_x, test_y)

    while True:
        print("Reply to what?")
        text = input('> ')
        print(decode_input_text(model, vocab, inverse_vocab, tokenize, text))



def build_model(vocab):
    """ Build keras model """
    from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Reshape, Flatten, RepeatVector, Dropout
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Model

    # Builds "question" encoder
    # First layer is word embedding
    question_input = Input(shape=(MAXLEN_INPUT, ), dtype='int32', name='question_input')

    shared_embedding = Embedding(
        output_dim=WORD_EMBED_SIZE,
        input_dim=VOCAB_SIZE,
        input_length=MAXLEN_INPUT,
        weights=[load_embedding_glove_weights(vocab)],
    )

    embedded_question = shared_embedding(question_input)

    # This is encoded recurrently into a sentence/context vector
    context = LSTM(
        CONTEXT_SIZE,
        kernel_initializer='lecun_uniform',
        name='question_encoder',
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
    )(embedded_question)

    # Next create answer encoder
    answer_input = Input(shape=(MAXLEN_INPUT, ), dtype='int32', name='answer_input')
    
    embedded_answer = shared_embedding(answer_input)

    context_embedded_answer_merge = concatenate([
        RepeatVector(MAXLEN_INPUT)(context),
        embedded_answer, # Shape: (MAXLEN_INPUT, WORD_EMBED_SIZE)
    ], axis=2)

    encoded_answer = LSTM(
        CONTEXT_SIZE, 
        kernel_initializer='lecun_uniform', 
        name='answer_encoder',
        return_sequences=True,
        dropout=LSTM_OUTPUT_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
    )(context_embedded_answer_merge)

    next_word_dense = Dropout(DENSE_DROPOUT)(TimeDistributed(
        Dense(
            int(VOCAB_SIZE / 2),
            activation='relu',
        ),
        name='next_word_dense',)(encoded_answer))

    answer_output = Dropout(DENSE_DROPOUT)(TimeDistributed(
        Dense(
            VOCAB_SIZE,
            activation='softmax',
        ),
        name='answer_output',)(next_word_dense))

    model = Model(inputs=[question_input, answer_input], outputs=[answer_output])

    print("Built keras {} GB keras model.".format(get_model_memory_usage(model)))
    return model


def train_model(model, vocab, inverse_vocab, tokenize, train_x, train_y, test_x, test_y):
    from keras import callbacks
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', patience=0)
    tb_callback = callbacks.TensorBoard(
        log_dir='./logs', 
        histogram_freq=0, 
        batch_size=BATCH_SIZE, 
        write_graph=True, 
        write_grads=False, 
        write_images=False, 
        embeddings_freq=0, 
        embeddings_layer_names=None, 
        embeddings_metadata=None
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    START = vocab['<START>']
    PAD = vocab['<PAD>']

    TEST_INPUT = [
        'AXBcatAXBcrime Well, what do you think, dear?',
        'It\'s time we got going.',
        'It\'s great to see you again!',
    ]

    try:
        for epoch in range(MAX_EPOCHS):
            print("Training epoch {}".format(epoch))


            window_start_iter = list(range(0, len(train_x), META_BATCH_SIZE))[:-1]
            window_end_iter = list(range(0, len(train_x), META_BATCH_SIZE))[1:]
            for start, end in list(zip(window_start_iter, window_end_iter)):
                for text in TEST_INPUT:
                    print('> ', text)
                    print(decode_input_text(model, vocab, inverse_vocab, tokenize, text))
                print("Processing batch {} to {} of {}".format(start, end, len(train_x)))
                batch_train_x = train_x[start:end, :]
                answers_in_train = np.hstack([
                    np.ones((len(batch_train_x), 1), dtype='int32') * START,
                    train_y[start:end, :-1]
                ])
                batch_train_y = binarize_labels(np.hstack([
                    train_y[start:end, :-1],
                    np.ones((len(batch_train_x), 1), dtype='int32') * PAD,
                ]))

                test_idx = random.sample(list(range(len(test_x))), int(META_BATCH_SIZE / 2))
                batch_test_x = test_x[test_idx]
                answers_in_test = np.hstack([
                    np.ones((len(batch_test_x), 1), dtype='int32') * START,
                    test_y[test_idx, :-1]
                ])
                batch_test_y = binarize_labels(np.hstack([
                    test_y[test_idx, :-1],
                    np.ones((len(batch_test_x), 1), dtype='int32') * PAD,
                ]))

                model.fit(
                    x=[batch_train_x, answers_in_train],
                    y=batch_train_y,
                    epochs=1,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stop_callback, tb_callback],
                )

                print(model.evaluate(
                    x=[batch_test_x, answers_in_test],
                    y=batch_test_y,
                    batch_size=BATCH_SIZE,
                ))

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, halting training...")
        return model


def decode_input_text(model, vocab, inverse_vocab, tokenize, text):
    result_tokens = [vocab['<START>']]
    for i in range(MAXLEN_INPUT):
        answer_vec = (result_tokens + [vocab['<PAD>']] * MAXLEN_INPUT)[:MAXLEN_INPUT]
        input_vec = [vectorize_data(tokenize, vocab, [text], is_input=True), 
                     np.array(answer_vec).reshape((1, MAXLEN_INPUT))]
        prediction = model.predict(input_vec)
        tokens = list(prediction.argmax(axis=2)[0])
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


