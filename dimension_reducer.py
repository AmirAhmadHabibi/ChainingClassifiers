import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from paths import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def read_dimension_and_context_size_for_each_year(path='./data/w2v-chi-yby-PCA.pkl'):
    with open(path, 'rb') as infile:
        word_vec = pickle.load(infile)
    dims, contexts = dict(), dict()
    for year, vecs in word_vec.items():
        contexts[year] = len(vecs)
        dims[year] = len(vecs[next(iter(vecs))])
    return dims, contexts


def reduce_dimensions(threshold=None, method='LDA', resize=False):
    # read the super words create hitherto for each year then make word vectors for each year's hitherto
    from utilitarian import Progresser

    if resize:
        dim_size, cntxt_size = read_dimension_and_context_size_for_each_year()

    with open(super_words_path, 'rb') as super_file:
        s_words = pickle.load(super_file)

    # remove context from next years
    for y in range(START, END):
        for c, words in s_words[y].items():
            for word in words.keys():
                # remove the word from the next years
                for yr in range(y + 1, END):
                    s_words[yr][c].pop(word, None)
    # print('done removing!')

    with open(w2v_path, 'rb') as w2v_file:
        w2v = pickle.load(w2v_file)

    # create the list of words to be predicted in each year
    word_list = dict()
    for yr in range(START, END):
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
    prog = Progresser(END-1 - START)

    # iterate on years and make the cumulative list of base words and make the word vector for each year's words
    for year in range(START, END-1):
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

        if resize:
            # d = D x n / N
            n = len(set(words) | set(word_list[year + 1]))
            n_dimension = int(dim_size[year + 1] * n / cntxt_size[year + 1])
        else:
            n_dimension = classifier_number - 1

        if method == 'PCA':
            transformer = PCA(n_components=n_dimension)
            transformer.fit(np.array(X))
        elif threshold is not None and year > threshold:
            pass
        else:
            transformer = LinearDiscriminantAnalysis(n_components=n_dimension, solver='eigen', shrinkage=0.5)
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
        with open(pca_reduced_path, 'wb') as outfile:
            pickle.dump(word_vectors, outfile)
    elif threshold is not None:
        with open(lda_f_reduced_path, 'wb') as outfile:
            pickle.dump(word_vectors, outfile)
    else:
        with open(lda_reduced_path, 'wb') as outfile:
            pickle.dump(word_vectors, outfile)


def tsne_embed(dimension=2):
    # load word2vecs
    with open('./data/w2v-chi-yby.pkl', 'rb') as w2v_file:
        w2v = pickle.load(w2v_file)
    word_list = []
    vector_list = []
    for key, value in w2v.items():
        word_list.append(key)
        vector_list.append(value)
    vector_embedded = TSNE(n_components=dimension, verbose=3).fit_transform(vector_list)
    w2v_reduced = dict()
    for i in range(0, len(word_list)):
        w2v_reduced[word_list[i]] = vector_embedded[i]
    with open('./data/w2v-chi-yby(' + str(dimension) + 'D).pkl', 'wb') as out_file:
        pickle.dump(w2v_reduced, out_file)
    print(str(dimension) + 'D done!')


def compare_word_vectors():
    w2v_path1 = './data/w2v-chi-yby.pkl'
    # w2v_path2 = './data/w2v-chi-en-yby.pkl'
    w2v_path2 = './data/w2v-chi-en2-yby.pkl'
    with open(w2v_path1, 'rb') as w2v_file:
        w2v1 = pickle.load(w2v_file)
    with open(w2v_path2, 'rb') as w2v_file:
        w2v2 = pickle.load(w2v_file)

    words = list(set(w2v1.keys()) & set(w2v2.keys()))
    dist1 = []
    dist2 = []
    for i in range(len(words) - 1):
        for j in range(i + 1, len(words)):
            dist1.append(cosine_similarity(w2v1[words[i]].reshape(1, -1), w2v1[words[j]].reshape(1, -1))[0][0])
            dist2.append(cosine_similarity(w2v2[words[i]].reshape(1, -1), w2v2[words[j]].reshape(1, -1))[0][0])
    print(pearsonr(dist1, dist2))
    # en1 (0.31718718532181123, 0.0)
#     0.36445544192266743, 0.0)

if __name__ == "__main__":
    # dims1, contexts1 = read_dimension_and_context_size_for_each_year()
    # dims2, contexts2 = read_dimension_and_context_size_for_each_year('./data/w2v-chi-en-yby-PCA_resz.pkl')
    # print()
    compare_word_vectors()
