import argparse

from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

import gensim
from gensim.models import Word2Vec

CS_CATEGORIES = {
    'semantic': {'Antonyms-nouns', 'Antonyms-adjectives', 'Antonyms-verbs', 'States-cities', 'Family-relations'},
    'syntactic': {'gram1-Nouns-plural', 'gram2-Jobs', 'gram3-Verb-past', 'gram4-Pronouns',
                  'gram5-Antonyms-adjectives,gradation', 'gram5-Nationalities'}
}


def _read_embeddings(embedding_file):
    # embedding_file = '/Users/lbiester/Desktop/final_project/pretrained/wiki.{}.bin'.format(language)
    # embedding_file = '/Users/lbiester/Desktop/final_project/pretrained/wiki.{}.vec'.format(language)
    # data = open(embedding_file).read()
    print('Loading embeddings...')
    # model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    if embedding_file.endswith('.model'):
        model = gensim.models.Word2Vec.load(embedding_file)
    # model = gensim.models.FastText.load_fasttext_format(embedding_file)
    print('Finished loading embeddings')
    return model


def _load_similarity_data(language):
    if language == 'ro':
        file = '/Users/lbiester/Desktop/final_project/experiment_1/ro_WS353.txt'
    elif language == 'de':
        file = '/Users/lbiester/Desktop/final_project/experiment_1/GUR65.txt'
    else:
        raise Exception('Language not supported')
    raw = open(file).read()
    similarities = {}
    for row in raw.splitlines():
        first, second, sim, _, _ = row.split(':')
        similarities[(first, second)] = float(sim)
    return similarities


def _load_analogy_data(language):
    if language == 'cs':
        file = '/Users/lbiester/Desktop/final_project/experiment_2/CZ.txt'
    raw = open(file).read()
    syntactic = []
    semantic = []
    is_syntactic = False
    for row in raw.splitlines():
        if row[0] == ':':
            category = row[2:]
            if row[2:] in CS_CATEGORIES['syntactic']:
                is_syntactic = True
            elif row[2:] in CS_CATEGORIES['semantic']:
                is_syntactic = False
            else:
                raise Exception('Analogy category not handled')
        else:
            analogy_tuple = tuple(string.lower() for string in row.split())
            if len(analogy_tuple) != 4:
                raise Exception('Analogy should have four items')
            if is_syntactic:
                syntactic.append(analogy_tuple)
            else:
                semantic.append(analogy_tuple)
    return syntactic, semantic


def similarity(language):
    similarity_data = _load_similarity_data(language)
    model = _read_embeddings(language)

    corr_input_similarity = []
    corr_input_vector = []
    for pair, sim in similarity_data.items():
        corr_input_vector.append(model.similarity(pair[0], pair[1]))
        corr_input_similarity.append(sim)

    print('Correlation:', spearmanr(corr_input_similarity, corr_input_vector))
    pass


def analogy(embedding_file):
    syntactic_analogies, semantic_analogies = _load_analogy_data('cs')
    model = _read_embeddings(embedding_file)
    correct_syntactic = 0
    for syn_analogy in syntactic_analogies:
        computed = model.similar_by_vector(model[syn_analogy[0]] - model[syn_analogy[1]] + model[syn_analogy[3]], 1)[0][0]
        if computed == syn_analogy[2]:
            correct_syntactic += 1
    print(correct_syntactic, len(syntactic_analogies))
    # print(syntactic_analogies)
    pass


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--language')
    parser.add_argument('--embedding_file')
    return parser.parse_args()


def main():
    args = _parse_args()
    # similarity('de')
    analogy(args.embedding_file)
    pass


if __name__ == '__main__':
    main()