# import modules & set up logging
import logging
from gensim.models import KeyedVectors
from queryparsing import QueriesParser
from ModelInterface import ModelInterface
from collections import defaultdict
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import download

# In order to download the stopwords list uncomment the next line
# download('stopwords')

STOPWORDS = set(stopwords.words('english'))
QUERYFILE = 'trainqueriesFirst50.xml'


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # model_interface = ModelInterface(model)
    queriesDB = QueriesParser(QUERYFILE)

    # model_interface.test_interface()

    load_model_sym_words(queriesDB)
    print('finished with the model, printing the synonyms')
    print('The synonyms dict is {}\n the length is {} (number of keys)'.format(queriesDB.synonyms, len(queriesDB.synonyms)))
    print('now starting add similar')
    queriesDB.add_similar_words()
    print('writing to file')
    queriesDB.write_to_file()


def load_model_sym_words(queriesDB):
    # using gzipped/bz2 input works too, no need to unzip:
    model = KeyedVectors.load_word2vec_format('Word2Vec/google/GoogleNewsnegative300.bin.gz', binary=True)

    for word in queriesDB.get_voc_words():
        queriesDB.synonyms[word] = get_similar_words(word, model)


def get_similar_words(word, model):
    similar_words = set()

    temp_list = []
    if model.vocab.get(word) is not None:
        temp_list += model.wv.similar_by_word(word)
    if model.vocab.get(word.title()) is not None:
        temp_list += model.wv.similar_by_word(word.title())
    if model.vocab.get(word.upper()) is not None:
        temp_list += model.wv.similar_by_word(word.upper())
    for symword, score in temp_list:
        synword = symword.lower().strip('-').strip('#').strip('\'').strip('\\').strip().split('_')
        if type(synword) is str:
            if filter(synword, score, word):
                similar_words.update(synword)
        else:
            for sword in synword:
                if filter(sword, score, word):
                    similar_words.update(sword)

    return similar_words


def filter(symword, score, word):
    if symword not in STOPWORDS:
        if similar(word, symword) < 0.5:
            if score > 0.3:
                return True
    return False


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


if __name__ == '__main__':
    main()
