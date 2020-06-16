from gensim.models import KeyedVectors

# using gzipped/bz2 input works too, no need to unzip:
word2vec_trained_model = KeyedVectors.load_word2vec_format('./google/GoogleNewsnegative300.bin.gz', binary=True)
