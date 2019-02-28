import argparse

from gensim.models import Word2Vec, FastText

LANGUAGE_PREFIX_MAP = {
    'arabic': 'ar',
    'czech': 'cs',
    'german': '',
    'english': 'en',
    'spanish': 'es',
    'french': 'fr',
    'italian': 'it',
    'romanian': 'ro',
    'russian': 'ru'
}
VECTOR_SIZE = 300
NEGATIVE = 5
WINDOW_SIZE = 5
MIN_COUNT = 5
SAMPLE_RATE = 0.0001


def _get_wiki_filename(language):
    return f'wiki_data/{LANGUAGE_PREFIX_MAP[language]}wiki_text'


def _train_model(language, percentage_of_data, algorithm):
    filename = _get_wiki_filename(language)
    if percentage_of_data is not None:
        raise NotImplemented('Varied percentage of data is not yet implemented!')

    step_size = 0.025 if algorithm == 'skipgram' else 0.05
    sg = 1 if algorithm == 'skipgram' else 0
    if algorithm in {'skipgram', 'cbow'}:
        model = Word2Vec(corpus_file=filename, size=VECTOR_SIZE, negative=NEGATIVE, window=WINDOW_SIZE,
                         min_count=MIN_COUNT, alpha=step_size, sample=SAMPLE_RATE, sg=sg)
    elif algorithm == 'fasttext':
        model = FastText(corpus_file=filename, size=VECTOR_SIZE, negative=NEGATIVE, window=WINDOW_SIZE,
                         min_count=MIN_COUNT, alpha=step_size, sample=SAMPLE_RATE, sg=sg)
    elif algorithm == 'fasttext_laura':
        raise NotImplemented("Laura's fasttext algorithm is not yet implemented")
    pass


def _parse_args():
    parser = argparse.ArgumentParser(description='Create word embeddings of wikipedia data')
    parser.add_argument('--language',
                        choices=['arabic', 'czech', 'german', 'english', 'spanish', 'french', 'italian',
                                 'romanian', 'russian'],
                        required=True)
    parser.add_argument('--percentage_of_data',
                        choices=list(range(0, 101)),
                        default=None)
    parser.add_argument('--algorithm',
                        choices=['fasttext_laura', 'fasttext', 'skipgram', 'cbow'],
                        required=True)
    parser.add_argument('--max_lines',
                        help='The maximum number of lines')

    return parser.parse_args()


def main():
    args = _parse_args()

    _train_model(args.language, args.percentage_of_data, args.algorithm)


if __name__ == '__main__':
    main()
