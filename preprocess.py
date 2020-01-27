import collections
import nltk
from nltk.corpus import stopwords
from nltk.metrics import TrigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
import itertools


def label_feats_from_corpus(corp):
    label_feats = collections.defaultdict(list)
    for fileid in range(corp['sentiment'].count()):
        feats = bag_of_non_stopwords(str(corp['text'][fileid]).split())
        label_feats[int(corp['sentiment'][fileid])].append(feats)
    return label_feats


def bag_of_non_stopwords(words, stopfile='spanish'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

def bag_of_words_not_in_set(words, badwords):
    return create_word_features(set(words) - set(badwords))


def create_word_features(words):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    score = TrigramAssocMeasures.chi_sq
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    finder = TrigramCollocationFinder.from_words(words)
    trigrams = finder.score_ngrams(trigram_measures.raw_freq)

    return dict([(word, True) for word in itertools.chain(words, trigrams)])

def split_label_feats(lfeats):
    test_feats = []
    for label, feats in lfeats.items():
        test_feats.extend([(feat, label) for feat in feats])
#         print(train_feats)
    return test_feats

