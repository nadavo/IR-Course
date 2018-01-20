# import modules & set up logging
from gensim.models import KeyedVectors
from ModelInterface import ModelInterface
from queryparsing import QueriesParser
from difflib import SequenceMatcher
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Timer import Timer
import logging
import enchant
import pickle

# In order to download the stopwords list uncomment the next line
# download('stopwords')

# Global Constants and Parameters
SYMBOLS = {'@', '#', '\\', '/', ':', '.', ',', ';', '+', '=', '%', '!', '~', '^', '*', '(', ')', '[', ']', '{', '}', '|', '<', '>', '?', '`', '\'', '\"', '\n', '\t', '-', ' '}
STOPWORDS = set(stopwords.words('english'))
QUERYFILE = 'trainqueriesFirst50.xml'
MIN_WORD_LENGTH = 2
MIN_SIM_SCORE = 0.5
MAX_SIM_RATIO = 0.8
NUM_ADD_WORDS = 3
SYNONYMS_CACHE = "cache/synonyms_{}_{}_{}.pkl".format(MIN_WORD_LENGTH, MIN_SIM_SCORE, MAX_SIM_RATIO)
LOAD_FROM_CACHE = True


def main():
    timer = Timer("Total Runtime")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # model_interface = ModelInterface(model)
    queriesDB = QueriesParser(QUERYFILE)

    # model_interface.test_interface()

    # TODO implement similar word lookup and query expansion only for short queries
    load_model_sym_words(queriesDB, LOAD_FROM_CACHE)
    print('finished with the model, printing the synonyms')
    print('The synonyms dict is {}\n the length is {} (number of keys)'.format(queriesDB.synonyms, len(queriesDB.synonyms)))
    print('now starting add similar')
    queriesDB.add_similar_words(NUM_ADD_WORDS)
    print('writing to file')
    queriesDB.write_to_file()
    timer.stop()


def load_model_sym_words(queriesDB, cached):
    if cached:
        with open(SYNONYMS_CACHE, 'rb') as cache:
            queriesDB.synonyms = pickle.load(cache)
    else:
        timer = Timer("Word2Vec Model Calculations")
        # using gzipped/bz2 input works too, no need to unzip:
        model = KeyedVectors.load_word2vec_format('Word2Vec/google/GoogleNewsnegative300.bin.gz', binary=True)

        for word in queriesDB.get_voc_words():
            queriesDB.synonyms[word] = get_similar_words(word, model)
        with open(SYNONYMS_CACHE, 'wb') as cache:
            pickle.dump(queriesDB.synonyms, cache)
        timer.stop()


def get_similar_words(word, model):
    similar_words = []

    temp_list = []
    if model.vocab.get(word) is not None:
        temp_list += model.wv.similar_by_word(word)
    if model.vocab.get(word.title()) is not None:
        temp_list += model.wv.similar_by_word(word.title())
    if model.vocab.get(word.upper()) is not None:
        temp_list += model.wv.similar_by_word(word.upper())

    for simword, score in temp_list:
        synword = simword.lower()
        if '_' in simword:
            synword = synword.split('_')
            for sword in synword:
                if word_filter(sword, score, word, similar_words):
                    similar_words.append(sword)
        else:
            if word_filter(synword, score, word, similar_words):
                similar_words.append(synword)

    return similar_words


def word_filter(simword, score, word, similar_words):
    if len(simword) <= MIN_WORD_LENGTH or len(word) <= MIN_WORD_LENGTH:
        return False
    if not is_correct_spelling(simword):
        return False
    if simword in STOPWORDS:
        return False
    for symbol in SYMBOLS:
        if symbol in simword:
            return False
    if are_stems_equal(word, simword):
        return False
    for added_word in similar_words:
        if are_stems_equal(simword, added_word):
            return False
    if score < MIN_SIM_SCORE:
        return False
    return True


def are_stems_equal(word, simword):
    stemmer = PorterStemmer()
    simword_stem = stemmer.stem(simword)
    word_stem = stemmer.stem(word)
    if word == simword or similar(word, simword) > MAX_SIM_RATIO or simword_stem == word_stem or simword_stem == word or simword == word_stem:
        return True
    return False


def is_correct_spelling(simword):
    spellchecker = enchant.Dict("en_US")
    return spellchecker.check(simword)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


if __name__ == '__main__':
    main()
