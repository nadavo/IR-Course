import xml.etree.ElementTree as eT
from collections import defaultdict
from difflib import SequenceMatcher


class QueriesParser:
    def __init__(self, query_file):
        self.file = query_file
        self.tree = eT.parse(self.file)
        self.root = self.tree.getroot()
        # query number: "Full command"
        self.queries_full_commands = defaultdict(str)
        # query number: ['query', 'words']
        self.queries = defaultdict(list)
        # query number: ['Query', 'Words'] first capital letter
        self.Queries = defaultdict(list)
        # query number: ['Query', 'words'] a combination of all possible lists
        self.Queries_combos = defaultdict(list)
        # query number: ['query', 'words'] a combination of all possible lists
        self.queries_combos = defaultdict(list)
        # query number: [sentences_lower_letter]
        self.query_sentences = defaultdict(list)
        # query number: [sentences_with_capital_letter]
        self.Query_sentences = defaultdict(list)
        self.__parse_queries()
        self.__generate_query_sentences()
        self.__generate_Query_sentences()
        self.synonyms = defaultdict(list)
        self.__initialize_synonyms()

    def __parse_queries(self):
        for query in self.root.iter('query'):
            self.queries_full_commands[query.find('number').text] = query.find('text').text

        for key, st in self.queries_full_commands.items():
            self.queries[key] = st.strip('#combine()').strip().split()

    def __generate_query_sentences(self):
        for key, lst in self.queries.items():
            for word in lst:
                self.queries_combos[key].append([word] + list(filter(lambda x: x != word.lower(), lst)))
        for key, lst in self.queries_combos.items():
            for combo in lst:
                self.query_sentences[key].append('_'.join(combo))

    def __generate_Query_sentences(self):
        for key, lst in self.queries.items():
            self.Queries[key] = [word.title() for word in lst]

        for key, lst in self.queries.items():
            for word in self.Queries[key]:
                self.Queries_combos[key].append([word] + list(filter(lambda x: x != word.lower(), lst)))

        for key, lst in self.Queries_combos.items():
            for combo in lst:
                self.Query_sentences[key].append('_'.join(combo))

    def __initialize_synonyms(self):
        for word_list in self.queries.values():
            for word in word_list:
                self.synonyms[word] = []

    def get_title_query_sentences(self):
        return self.Query_sentences

    def get_query_sentences(self):
        return self.query_sentences

    def get_query_words_dict(self):
        """
        Returns a dictionary of all the query words (in lower case)
        :return: query number:[query words]
        """
        return self.queries

    def get_query_title_words_dict(self):
        """
        Returns a dictionary of all the query words (Titled)
        :return: query number:[Query Words]
        """
        return self.Queries

    def add_similar_words(self, num_words=1):
        """
        Adds the similar words from word2vec model to original queries
        :parameter: num_words: number of similar words to add to query for each query word (default is 3)
        """
        for index, words in self.queries.items():
            for word in words:
                syn = ['<', word]
                similar_words = self.synonyms.get(word, [])
                num_sim = min(num_words, len(similar_words))
                for i in range(num_sim):
                    syn.append(similar_words[i])
                syn.append('>')
                self.queries[index].extend(' '.join(syn))

    @staticmethod
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def write_to_file(self):
        for query in self.root.iter('query'):
            words = ' '.join(self.queries[query.find('number').text])
            text = '#combine( {} )'.format(words)
            query.find('text').text = text
        self.tree.write(self.file)

    def get_voc_words(self):
        return self.synonyms.keys()
