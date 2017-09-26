
# Conversation seq2seq with Keras and Cornell Movie Dialog Dataset

## Goal

The goal of this repo is to demonstrate creating a _real_ seq2seq model in Keras, and evaluating 
it's results.  There are a lot of incorrect and incomplete seq2seq implementations out there, and
I was unable to find a reference implementation in Keras with actual results against an open dataset
discuessed anywhere.

This model does not implement attention, though thanks to the correct implementation of seq2seq 
here, it would not be difficult to add.

## Data

[Cornell Movie Dialog Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

Size:

- 304,713 lines of dialog
- 9,035 characters
- 616 movies
- 24 categories

Each line of `movie_lines.txt` has the line ID, character ID, movie ID, character name, and the line of dialog.

## Usage

First, download the dataset linked above, and symlink or copy it to `./data/`.

Second, generate development and heldout data from dataset:

```bash
$ pip3 install -r requirements.txt --user # if necessary
$ # We should see the following results
$ ls data
chameleons.pdf                 movie_conversations.txt  movie_titles_metadata.txt  README.txt
movie_characters_metadata.txt  movie_lines.txt          raw_script_urls.txt
$ # Run the data prep to create develop and heldout split
$ PYTHONPATH=$(pwd) python3 main.py prep
Loading movie, character, and conversation data...
Splitting data into develop and heldout data based on movie...
75 of 617 movies chosen for heldout...
Writing development data to data/develop/...
Writing heldout data to data/heldout/...
Done with prep!
$ ls data
chameleons.pdf  heldout                        movie_conversations.txt  movie_titles_metadata.txt  README.txt
develop         movie_characters_metadata.txt  movie_lines.txt          raw_script_urls.txt
$ PYTHONPATH=$(pwd) python3 main.py train
... lots of training stuff...
```


## Results

`TODO`
