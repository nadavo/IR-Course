

class ModelInterface:
    def __init__(self, model):
        self.model = model

    def test_interface(self):
        while True:
            input1 = input('input first word\n')
            input2 = input('input 2nd word \n')
            if input1 == 'end':
                break
            if self.model.vocab.get(input1) is None:
                print('{} is not in the vocab'.format(input1))
                continue
            if self.model.vocab.get(input2) is None:
                print('{} is not in the vocab'.format(input2))
                continue
            print('These are most similar to {}:\n'.format(input1))
            print(self.model.similar_by_word(input1))
            print('now print similarity {} to {} \n'.format(input1, input2))
            print(self.model.similarity(input1, input2))
