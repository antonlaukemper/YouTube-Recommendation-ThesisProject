import re
from collections import Counter

import fasttext
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from spellchecker import SpellChecker


def get_corpus_description(data):
    corpus = ''.join(data["text"])
    corpus_tokens = fasttext.tokenize(corpus)
    print("raw number of tokens: %d" % len(corpus_tokens))
    counts = Counter(corpus_tokens)
    print("raw number of disctinct tokens: %d" % len(counts))

    print("#### running the spellchecker###")
    # the only spellcheck I'm doing is removing repeated letters
    checked_corpus = re.sub(r"(.)\1{2,}", r"\1\1", corpus)
    corpus_tokens = fasttext.tokenize(checked_corpus)
    print("raw number of tokens: %d" % len(corpus_tokens))
    counts = Counter(corpus_tokens)
    print("raw number of disctinct tokens: %d" % len(counts))


    print('#### removing html tags')
    html_free_corpus = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\\u201c|\\u2019', '', checked_corpus)
    html_free_corpus_tokens = fasttext.tokenize(html_free_corpus)
    print("number of tokens: %d" % len(html_free_corpus_tokens))
    html_counts = Counter(html_free_corpus_tokens)
    print("number of disctinct tokens: %d" % len(html_counts))

    print('#### removing links')
    link_free_corpus = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', html_free_corpus)
    link_free_corpus_tokens = fasttext.tokenize(link_free_corpus)
    print("number of tokens: %d" % len(link_free_corpus_tokens))
    link_counts = Counter(link_free_corpus_tokens)
    print("number of disctinct tokens: %d" % len(link_counts))

    print("#### removing special symbols and numbers ###")
    # remove special characters and numbers
    clean_corpus = re.sub('[^A-Za-z\s]+', '', link_free_corpus)
    clean_corpus_tokens = fasttext.tokenize(clean_corpus)
    print("number of tokens: %d" % len(clean_corpus_tokens))
    clean_counts = Counter(clean_corpus_tokens)
    print("number of disctinct tokens: %d" % len(clean_counts))

    # only the special symbols
    dirty_corpus = re.sub('[A-Za-z\s]+', '', corpus)
    distinct_symbols = Counter(dirty_corpus)
    print("Number of distinct removed special symbols: %d" % len(distinct_symbols))



    print("#### removing stop words ####")
    stop_words = set(stopwords.words('english'))
    stop_words = [re.sub('[^A-Za-z\s]+', '', word) for word in stop_words]
    print("number of stop words: %d" % len(stop_words))
    corpus_wo_stop_words = [token for token in clean_corpus_tokens if not token in stop_words]
    counts_wo_stopwords = Counter(corpus_wo_stop_words)
    print("Number of tokens wo stopwords: %d" % len(corpus_wo_stop_words))
    print("Number of distinct tokens: %d" % len(counts_wo_stopwords))

    print("#### lemmatization ####")
    lemmatizer = WordNetLemmatizer()
    lemmatized_corpus = [lemmatizer.lemmatize(x) for x in tqdm(corpus_wo_stop_words)]
    counts_lemmatized = Counter(lemmatized_corpus)
    print("Number lemmatized tokens: %d" % len(lemmatized_corpus))
    print("Number of distinct lemmatized tokens: %d" % len(counts_lemmatized))
    return counts_lemmatized.most_common(25000)