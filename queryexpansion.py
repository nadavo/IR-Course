import xml.etree.ElementTree as eT
from collections import defaultdict


class QueryWriter(eT.TreeBuilder):
    def __init__(self, file, queriesDB):
        super().__init__()
        self.file = file
        self.tree = eT.parse(self.file)
        self.root = self.tree.getroot()
        self.queries = queriesDB.get_query_words_dict()
        self.original_commands = queriesDB.queries_full_commands
        self.synonyms = queriesDB.synonyms
        self.expanded = defaultdict(str)

    def add_similar_words(self, num_words=3):
        """
        Adds the similar words from word2vec model to original queries
        :parameter: num_words: number of similar words to add to query for each query word (default is 3)
        """
        for index, words in self.queries.items():
            query_length = len(words)
            syn = ['#weight(']
            # Will add more words to shorter queries
            num_sim = max(0, num_words - query_length)
            if num_sim == 0:
                continue
            for word in words:
                similar_words = self.synonyms.get(word, [])
                num_sim = min(num_sim, len(similar_words))
                for i in range(num_sim):
                    score = str(similar_words[i][0])
                    symword = similar_words[i][1]
                    syn.append(' '.join([score, symword]))
            syn.append(')')
            self.expanded[index] = ' '.join(syn)

    def write_to_file(self, orig_wieght=0.8):
        for query in self.root.iter('query'):
            index = query.find('number').text
            original_words = self.original_commands[index]
            expanded_words = self.expanded[index]
            if expanded_words == '':
                continue
            text = '#weight( {} {} {:.1f} {} )'.format(orig_wieght, original_words, 1 - orig_wieght, expanded_words)
            query.find('text').text = text
        self.tree.write(self.file)
