""" Prepare data for seq2seq learning.

Each example is the input text (the prior line of dialog/utterance) preceded by its metadata, 
and the output is the response utterance and its metadata.  For example:

Input:
    __m2 __cat_comedy __u302 __g_female Hey this is my first line that's cool.

Output:
    __m2 __cat_comedy __u305 __g_? That's great, here's a reply.


Then, each reply has its own row of input and output, meaning most data is in both the input and
output.  Input reversing happens during model training.

"""

import random
from collections import namedtuple, defaultdict, Counter
import toolz
import numpy
from sklearn.model_selection import StratifiedKFold
import warnings

pct_heldout = 0.10

DELIM = b'+++$+++'

Utterance = namedtuple('Utterance', 
    ('utterance_id', 'character_id', 'movie_id', 'character_name', 'text'))
Movie = namedtuple('Movie', 
    ('movie_id', 'movie_title', 'release_year', 'imdb_rating', 'imdb_votes', 'genres'))
Character = namedtuple('Character', 
    ('character_id', 'character_name', 'movie_id', 'movie_title', 'gender', 'credits_position'))
Conversation = namedtuple('Conversation',
    ('first_character_id', 'second_character_id', 'movie_id', 'utterance_ids'))


def load_from_file(path, datatype):
    with open(path, 'rb') as infile:
        return [datatype(*map(bytes.strip, line.strip().split(DELIM))) for line in infile]


def load_movies(path='data/movie_titles_metadata.txt'):
    def row_to_movie(*row):
        new_row = row[:-1] + (list(map(str.encode, eval(row[-1]))), )
        return Movie(*new_row)
    return load_from_file(path, row_to_movie)


def load_characters(path='data/movie_characters_metadata.txt'):
    return load_from_file(path, Character)


def load_conversations(path='data/movie_conversations.txt'):
    def row_to_conversation(*row):
        new_row = row[:-1] + (list(map(str.encode, eval(row[-1]))), )
        return Conversation(*new_row)
    return load_from_file(path, row_to_conversation)


def load_utterances(path='data/movie_lines.txt'):
    return load_from_file(path, Utterance)


def in_out_from_utterances(prior, reply, movies, characters, min_genres):
    movie_tag = b'__%b' % (prior.movie_id)
    genre_tag = b'__cat__%b' % (min_genres[prior.movie_id])
    
    prior_char_tag = b'__%b' % (reply.character_id)
    prior_gender_tag = b'__%b' % (characters[reply.character_id].gender.lower())
    in_datum = b'%b %b %b %b %b' % (genre_tag, movie_tag, prior_gender_tag, prior_char_tag,
                                        prior.text)

    reply_char_tag = b'__%b' % (reply.character_id)
    reply_gender_tag = b'__%b' % (characters[reply.character_id].gender.lower())
    out_datum = b'%b %b %b %b %b' % (genre_tag, movie_tag, reply_gender_tag, reply_char_tag,
                                         reply.text)

    return in_datum, out_datum


def develop_heldout_split_movie_ids(movies):
    """ Evenly and randomly splits movies based on their most minority genre """
    genre_counts = Counter(toolz.concat(movie.genres for movie in movies.values()))

    def minority_genre(movie):
        if len(movie.genres) == 0:
            return b'no_genre'
        return min([(genre, genre_counts[genre]) for genre in movie.genres], 
                   key=lambda p: p[1])[0]

    minority_movie_genres = [(movie.movie_id, minority_genre(movie)) for movie in movies.values()]
    movie_ids, genres = map(numpy.array, zip(*minority_movie_genres))

    with warnings.catch_warnings():
        # Some classes just aren't well enough represented, so ignore the warning.
        warnings.simplefilter("ignore")
        kfold = StratifiedKFold(n_splits=int(1 / pct_heldout))
        develop_idx, heldout_idx = next(kfold.split(movie_ids, genres))

    return list(movie_ids[develop_idx]), list(movie_ids[heldout_idx]), dict(minority_movie_genres)


def prep():
    print("Loading movie, character, and conversation data...")
    characters = {char.character_id: char for char in load_characters()}
    movies = {movie.movie_id: movie for movie in load_movies()}
    conversations = load_conversations()
    utterances = {utterance.utterance_id: utterance for utterance in load_utterances()}

    print("Splitting data into develop and heldout data based on movie...")
    develop_movie_ids, heldout_movie_ids, min_genres = develop_heldout_split_movie_ids(movies)
    print("{} of {} movies chosen for heldout...".format(len(heldout_movie_ids), len(movies)))

    movie_in_data = defaultdict(list)
    movie_out_data = defaultdict(list)

    for conv in conversations:
        for prior_id, reply_id in zip(conv.utterance_ids[:-1], conv.utterance_ids[1:]):
            prior = utterances[prior_id]
            reply = utterances[reply_id]
            in_data, out_data = in_out_from_utterances(
                prior, reply, movies, characters, min_genres)
            movie_in_data[conv.movie_id].append(in_data)
            movie_out_data[conv.movie_id].append(out_data)

    develop_in_data = list(toolz.concat([movie_in_data[mid] for mid in develop_movie_ids]))
    develop_out_data = list(toolz.concat([movie_out_data[mid] for mid in develop_movie_ids]))
    heldout_in_data = list(toolz.concat([movie_in_data[mid] for mid in heldout_movie_ids]))
    heldout_out_data = list(toolz.concat([movie_out_data[mid] for mid in heldout_movie_ids]))

    print("Writing develop and heldout data to data/*_data.txt...")
    with open('data/develop_in_data.txt', 'wb') as outfile:
        outfile.write(b'\n'.join(develop_in_data))

    with open('data/develop_out_data.txt', 'wb') as outfile:
        outfile.write(b'\n'.join(develop_out_data))

    with open('data/heldout_in_data.txt', 'wb') as outfile:
        outfile.write(b'\n'.join(heldout_in_data))

    with open('data/heldout_out_data.txt', 'wb') as outfile:
        outfile.write(b'\n'.join(heldout_out_data))

    print("Done with prep!")