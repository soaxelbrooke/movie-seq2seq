
# Conversation seq2seq with Keras and Cornell Movie Dialog Dataset

## Data

[Cornell Movie Dialog Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

Size:

- 304,713 lines of dialog
- 9,035 characters
- 616 movies
- 24 categories
  - (adult, fantasy, drama, biography, action, musical, comedy, documentary, history, sport, adventure, mystery, film-noir, crime, western, sci-fi, thriller, horror, music, war, family, romance, short, animation)


Each line of `movie_lines.txt` has the line ID, character ID, movie ID, character name, and the line of dialog.

## Usage

First, download the dataset linked above, and symlink or copy it to `./data/`.

Second, generate development and heldout data from dataset:

```bash
$ # We should see the following results
$ ls data
chameleons.pdf                 movie_conversations.txt  movie_titles_metadata.txt  README.txt
movie_characters_metadata.txt  movie_lines.txt          raw_script_urls.txt
$ # Run the data prep to create develop and heldout split
$ PYTHONPATH=$(pwd) python3 prep
```
