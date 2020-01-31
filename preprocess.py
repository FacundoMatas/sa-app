import collections
import nltk
from nltk.corpus import stopwords
from nltk.metrics import TrigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
import itertools
import re

def label_feats_from_corpus(corp):
    # corp = preprocesamiento_df(corp_aux)
    # corp['text'] = corp.apply(preprocesamiento_row, axis=1)
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


def preprocesamiento_row(row):
    replaced = re.sub("(?:^|\s)[＃#]{1}(\w+)", '', str(row['text']))
    replaced_wo_user = re.sub("(?:^|\s)[＠ @]{1}([^\s#<>[\]|{}]+)", '', replaced)
    replaced_links = re.sub("/[\w.-]*", '', replaced_wo_user)
    replaced_links = replaced_links.replace("\r", " ")
    replaced_links = replaced_links.replace("\n", " ")
    replaced_links = ''.join([i for i in replaced_links if not i.isdigit()])
    a, b = 'áéíóúüàèìòù', 'aeiouuaeiou'
    trans = str.maketrans(a, b)
    replaced_links = replaced_links.translate(trans)
    replaced_links = replaced_links.translate(str.maketrans('', '', '1234567890'))
    replaced_links = replaced_links.replace('[^\w\s]', '')

    return replaced_links

def preprocesamiento_df(df):
    df["text"] = df["text"].str.replace("RT @.*:", "")
    df["text"] = df["text"].str.replace('https://t.co*',"")
    df["text"] = df["text"].str.replace('//t.co*', "")
    df['text'] = df['text'].str.lower()
    return df



