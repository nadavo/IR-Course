# import modules & set up logging
import gensim, logging
from gensim.models import Word2Vec, KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Loading the model
# Simple loading (general)
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
#
# using gzipped/bz2 input works too, no need to unzip:
model = KeyedVectors.load_word2vec_format('google/GoogleNewsnegative300.bin.gz', binary=True)


while True:
    input1 = input('input first word\n')
    input2 = input('input 2nd word \n')
    if input1 == 'end':
        break
    if model.vocab.get(input1) is None:
        print('{} is not in the vocab'.format(input1))
        continue
    if model.vocab.get(input2) is None:
        print('{} is not in the vocab'.format(input2))
        continue
    print('These are most similar to {}:\n'.format(input1))
    print(model.similar_by_word(input1))
    print('now print similarity {} to {} \n'.format(input1, input2))
    print(model.similarity(input1, input2))

