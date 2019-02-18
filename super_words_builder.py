import pickle
from os import listdir
import os.path
from utilitarian import QuickDataFrame

""" We named our list of classifier-words "Super words"
    The data structure is a Python dictionary of years -> dict of classifiers-> dict of words -> frequencies
    e.g. s_words[1987]['份']['日志'] is 4, meaning that in google ngrams dataset the frequency of the use of 
    the classifier '份' with the context noun '日志' in the year of 1987 is 4
"""


def save_classifier_nouns():
    # this super words list is not like the final super_words (i.e. does not have time stamps and frequency data)
    super_words = dict()
    with open('./data/gwc2016_classifiers/lemma_dictionary_tao1.txt', encoding='utf-8') as infile:
        # ChineseLemma   \t   Classifier   \t   FrequencyCount
        for line in infile:
            if len(line) < 2 or line[0] == '#':
                continue
            context, classifier, freq = line.split('\t')
            if int(freq) > 1:
                if classifier not in super_words:
                    super_words[classifier] = set()
                super_words[classifier].add(context)

    # deleting categories with less than 10 instances
    bad_cats = set()
    for cat, words in super_words.items():
        if len(words) < 10:
            bad_cats.add(cat)
    for cat in bad_cats:
        del super_words[cat]
    print(len(bad_cats), 'categories removed due to having less than 10 instances')
    super_words = {2010: super_words}
    with open('./data/super_words-chi-Luis.pkl', 'wb') as super_file:
        pickle.dump(super_words, super_file)


def save_time_stamps():
    with open('./data/super_words-chi-Luis.pkl', 'rb') as super_file:
        super_words = pickle.load(super_file)

    if os.path.isfile('./data/time_stamp_data_luis(w2v).pkl'):
        with open('./data/time_stamp_data_luis(w2v).pkl', 'rb') as infile:
            time_stamps = pickle.load(infile)
    else:
        # initialise time_stamps for each combination of "classifier context"
        time_stamps = dict()
        for decade, cats in super_words.items():
            for cat, words in cats.items():
                for word in words:
                    query = cat + word
                    time_stamps[query] = []

    if os.path.isfile('./data/checked_files(w2v).pkl'):
        with open('./data/checked_files(w2v).pkl', 'rb') as infile:
            checked_files = pickle.load(infile)
    else:
        checked_files = set()

    ngram_path = '/media/disk_ngram2/ngram_classifier_data/'  # path to the folder of google chinese ngrams
    file_names_list = listdir(ngram_path)
    i = 0
    for file_name in file_names_list:
        i += 1
        if file_name in checked_files:
            continue
        print(i, 'of', len(file_names_list))
        with open(ngram_path + file_name, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    ngram = line.partition('\t')[0]
                    ngram_ref = ngram.replace(' ', '')
                    ngram_ref = ngram_ref.replace('_NOUN', '')
                    if ngram_ref in time_stamps:
                        time_stamps[ngram_ref].append(line.strip())
                except Exception as e:
                    print(e, line)
            checked_files.add(file_name)
        if i % 50 == 0:
            with open('./data/checked_files(w2v).pkl', 'wb') as outfile:
                pickle.dump(checked_files, outfile)
            with open('./data/time_stamp_data_luis(w2v).pkl', 'wb') as outfile:
                pickle.dump(time_stamps, outfile)

    with open('./data/checked_files(w2v).pkl', 'wb') as outfile:
        pickle.dump(checked_files, outfile)
    # save time_stamp data to file:
    with open('./data/time_stamp_data_luis(w2v).pkl', 'wb') as outfile:
        pickle.dump(time_stamps, outfile)


def build_super_words():
    """make a dict of dict of dict : super_word[year][classifier][context]=frequency"""

    with open('./data/time_stamp_data_luis(w2v).pkl', 'rb') as infile:
        timestamps = pickle.load(infile)

    # load the list of w2v to filter out the words that are not covered by our w2v
    with open('./data/w2v-chi.pkl', 'rb') as in_file:
        word_list = pickle.load(in_file)

    with open('./data/super_words-chi-Luis.pkl', 'rb') as super_file:
        # super words without time data
        s_words = pickle.load(super_file)

    # make lists of all contexts that are also in FasText and all classifiers
    classifiers = []
    for clsf, cntx in s_words[2010].items():
        classifiers.append(clsf)

    # make a dict of all years and create dicts for each
    super_words = dict()
    for year in range(1940, 2010):
        super_words[year] = dict()
        for cat in classifiers:
            super_words[year][cat] = dict()

    # fill in the frequencies
    for cat, wrds in s_words[2010].items():
        for wrd in wrds:
            if cat + wrd not in timestamps or wrd not in word_list or len(wrd) < 2:
                continue
            query_ngram = cat + wrd + '_NOUN'
            for item in timestamps[cat + wrd]:
                try:
                    ngram, year, freq, books = item.split('\t')
                    ngram = ngram.replace(' ', '')
                    if ngram != query_ngram:
                        continue
                    year = int(year)
                    if year in super_words:
                        super_words[year][cat][wrd] = int(freq)
                except Exception as e:
                    print(e)

    with open('./data/super_words-chi-Luis-YbyY(w2v).pkl', 'wb') as super_file:
        pickle.dump(super_words, super_file)

