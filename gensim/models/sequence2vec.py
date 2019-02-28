"""
This module extends word2vec to allow training outside of the traditional sentence setting
The goal is to be able to include context without the restrictions of the sentence format

For example, in the building example, if we want to train each building to predict functionalities of adjacent buildings
in the student schedule, this is not really possible with a sentence.
"""
import logging
from collections import defaultdict

from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.word2vec import Word2VecVocab

logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
    from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000

    def _do_train_job(self, contexts, alpha, inits):
        """Train the model on a single batch of sentences.

        Parameters
        ----------
        sentences : iterable of list of str
            Corpus chunk to be used in this training batch.
        alpha : float
            The learning rate used in this batch.
        inits : (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        Returns
        -------
        (int, int)
             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).

        """
        pass


class Sequence2Vec(BaseWordEmbeddingsModel):
    def __init__(self, context_iterable, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
                 max_final_vocab=None):

        self.vocabulary = Sequence2VecVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample, sorted_vocab=bool(sorted_vocab),
            null_word=null_word, max_final_vocab=max_final_vocab, ns_exponent=ns_exponent)

        super(Sequence2Vec, self).__init__(context_iterable=context_iterable, workers=workers, vector_size=size,
                                           epochs=iter, callbacks=callbacks, batch_words=batch_words,
                                           trim_rule=trim_rule, sg=sg, alpha=alpha, window=window, seed=seed, hs=hs,
                                           negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha,
                                           compute_loss=compute_loss, fast_version=FAST_VERSION)


class Sequence2VecVocab(Word2VecVocab):
    def __init__(
            self, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0,
            max_final_vocab=None, ns_exponent=0.75):
        super(Sequence2VecVocab, self).__init__(max_final_vocab, min_count, sample, sorted_vocab, null_word,
                                                max_final_vocab, ns_exponent)

    def scan_vocab(self, sentences=None, corpus_file=None, context_iterable=None, progress_per=10000, workers=None,
                   trim_rule=None):
        # TODO (laurabiester): finish this section!
        logger.info("collecting all words and their counts")
        # if corpus_file:
        #     sentences = LineSentence(corpus_file)

        total_words, corpus_count = self._scan_vocab_context_iterable(context_iterable, progress_per)

        logger.info(
            "collected %i word types from a corpus of %i raw words and %i sentences",
            len(self.raw_vocab), total_words, corpus_count
        )

        return total_words, corpus_count

    def _scan_vocab_context_iterable(self, context_iterable, progress_per):
        context_no = -1
        total_words = 0
        vocab = defaultdict(int)
        for context_no, sequence_context in enumerate(context_iterable):
            if context_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                    context_no, total_words, len(vocab)
                )

            vocab[sequence_context.word] += 1
            total_words += 1

            # TODO (lbiester): handle max vocab?
            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                raise Exception('Max vocab not yet handled properly')

        corpus_count = context_no + 1
        self.raw_vocab = vocab
        return total_words, corpus_count


class SequenceContext(object):
    def __init__(self, word, contexts, context_weights=None):
        self.word = word
        self.contexts = contexts
