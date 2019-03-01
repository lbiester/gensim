from gensim.models.sequence2vec import Sequence2Vec, SequenceContext
from gensim.models.word2vec import Word2Vec, LineSentence  # avoid referencing __main__ in pickle

sentences = [['I', 'like', 'dogs'],
             ['I', 'have', 'two', 'cats'],
             ['The', 'beach', 'is', 'warm'],
             ['Trees', 'are', 'green'],
             ['Dogs', 'do', 'not', 'like', 'cats'],
             ['Dogs', 'like', 'trees'],
             ['My', 'cat', 'is', 'brown']]

# just making some simple arbitrary test data
contexts = [SequenceContext('dry', ['wet', 'cold', 'hair', 'air']),
            SequenceContext('dog', ['cat', 'pet', 'animal', 'hair']),
            SequenceContext('tree', ['nature', 'grass', 'park', 'brown', 'leaves'])]

# model = Word2Vec(sentences, min_count=1, sg=True)
model = Sequence2Vec(contexts, min_count=1, sg=True)
print(model)

