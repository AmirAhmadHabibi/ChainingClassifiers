import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def reduce_dimensions(start=1940, threshold=None, method='LDA'):
    # read the super words create hitherto for each year then make word vectors for each year's hitherto
    from utilitarian import Progresser

    with open('./data/super_words-chi-Luis-YbyY(w2v).pkl', 'rb') as super_file:
        s_words = pickle.load(super_file)

    # remove context from next years
    for y in range(start, 2010):
        for c, words in s_words[y].items():
            for word in words.keys():
                # remove the word from the next years
                for yr in range(y + 1, 2010):
                    s_words[yr][c].pop(word, None)
    print('done removing!')

    with open('./data/w2v-chi-yby.pkl', 'rb') as w2v_file:
        w2v = pickle.load(w2v_file)

    # create the list of words to be predicted in each year
    word_list = dict()
    for yr in range(start, 2010):
        word_list[yr] = set()
        for cat, words in s_words[yr].items():
            word_list[yr] = word_list[yr] | words.keys()

    categories = dict()
    for i, cat in enumerate(s_words[1960].keys()):
        categories[cat] = i

    # create the list of the words that the predictions are based on in each year
    s_words_hitherto = dict()
    for cat in categories:
        s_words_hitherto[cat] = set()

    word_vectors = dict()
    prog = Progresser(2009 - start)

    # iterate on years and make the cumulative list of base words and make the word vector for each year's words
    for year in range(start, 2009):
        # add the words for this year to the best cat in hitherto list
        for word in word_list[year]:
            best_cat = ''
            max_f = 0
            for cat in categories:
                if word in s_words[year][cat]:
                    if s_words[year][cat][word] > max_f:
                        max_f = s_words[year][cat][word]
                        best_cat = cat
            s_words_hitherto[best_cat].add(word)

        words = []
        X = []
        y = []
        classifier_number = 0

        for cat in categories:
            if len(s_words_hitherto[cat]) > 0:
                classifier_number += 1
            for word in s_words_hitherto[cat]:
                words.append(word)
                X.append(w2v[word])
                y.append(categories[cat])

        if method == 'PCA':
            transformer = PCA(n_components=classifier_number - 1)
            transformer.fit(np.array(X))
        elif threshold is not None and year > threshold:
            pass
        else:
            transformer = LinearDiscriminantAnalysis(n_components=classifier_number - 1, solver='eigen', shrinkage=0.5)
            transformer.fit(np.array(X), np.array(y))

        # transform ad add both hitherto words of this year and new words of the next year
        X_new = transformer.transform(np.array(X))
        # make the new word vector dict for this year
        word_vectors[year + 1] = dict()
        for i in range(len(words)):
            word_vectors[year + 1][words[i]] = X_new[i]

        for word in word_list[year + 1]:
            if word not in word_vectors[year + 1]:
                word_vectors[year + 1][word] = transformer.transform(np.array([w2v[word]]))[0]

        prog.count()

    # write the word vectors in file
    if method == 'PCA':
        with open('./data/w2v-chi-yby-PCA.pkl', 'wb') as outfile:
            pickle.dump(word_vectors, outfile)
    elif threshold is not None:
        with open('./data/w2v-chi-yby-s0.5LDA-fixed.pkl', 'wb') as outfile:
            pickle.dump(word_vectors, outfile)
    else:
        with open('./data/w2v-chi-yby-s0.5LDA.pkl', 'wb') as outfile:
            pickle.dump(word_vectors, outfile)


def tsne_embed(dimention=2):
    # load word2vecs
    with open('./data/w2v-chi-yby.pkl', 'rb') as w2v_file:
        w2v = pickle.load(w2v_file)
    word_list = []
    vector_list = []
    for key, value in w2v.items():
        word_list.append(key)
        vector_list.append(value)
    vector_embedded = TSNE(n_components=dimention, verbose=3).fit_transform(vector_list)
    w2v_reduced = dict()
    for i in range(0, len(word_list)):
        w2v_reduced[word_list[i]] = vector_embedded[i]
    with open('./data/w2v-chi-yby(' + str(dimention) + 'D).pkl', 'wb') as out_file:
        pickle.dump(w2v_reduced, out_file)
    print(str(dimention) + 'D done!')
