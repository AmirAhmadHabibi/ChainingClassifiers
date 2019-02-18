import copy
from the_predictor_II_I import SuperPredictor
import pickle

# load super words
with open('./super-words-chi/super_words-chi-Luis-YbyY(w2v).pkl', 'rb') as super_file:
    s_words = pickle.load(super_file)

# save a copy with the frequencies in all years
sw_complete = copy.deepcopy(s_words)

# remove context from next years so to have only the first year of appearance of each context word
for y in range(1940, 2010):
    for c, words in s_words[y].items():
        for appeared_word in words.keys():
            # remove the word from the next years
            for next_year in range(y + 1, 2010):
                s_words[next_year][c].pop(appeared_word, None)
print('done removing!')

# remove categories that are empty after 1940
for empty_cat in ['枝', '桌', '课', '进', '丝', '记', '盏', '夥', '轴', '尾', '针']:
    for y in range(1940, 2010):
        s_words[y].pop(empty_cat, None)


def make_predictor(path, w2v_version, s, t, e):
    """find the word vector to load and then make a predictor object with it"""

    if 'LDA' in w2v_version or 'PCA' in w2v_version:
        if w2v_version == 'LDA':
            with open('./super-words-chi/w2v-chi-yby-LDA.pkl', 'rb') as infile:
                w2v_yby = pickle.load(infile)
        elif w2v_version == 's0.5LDA':
            with open('./super-words-chi/w2v-chi-yby-s0.5LDA.pkl', 'rb') as infile:
                w2v_yby = pickle.load(infile)
        elif w2v_version == 's0.5LDA-F':
            with open('./super-words-chi/w2v-chi-yby-s0.5LDA-fixed.pkl', 'rb') as infile:
                w2v_yby = pickle.load(infile)
        elif w2v_version == 'PCA':
            with open('./super-words-chi/w2v-chi-yby-PCA.pkl', 'rb') as infile:
                w2v_yby = pickle.load(infile)
        else:
            raise Exception("Word2vec version not found!")
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, path=path,
                                   word_vectors=None, word_vector_yby=w2v_yby, swc=sw_complete)
    else:
        if w2v_version == '2D':
            with open('./super-words-chi/w2v-chi-yby(2D).pkl', 'rb') as w2v_file:
                w2v = pickle.load(w2v_file)
        elif w2v_version == '5D':
            with open('./super-words-chi/w2v-chi-yby(5D).pkl', 'rb') as w2v_file:
                w2v = pickle.load(w2v_file)
        elif w2v_version == 'orig':
            with open('./super-words-chi/w2v-chi-yby.pkl', 'rb') as w2v_file:
                w2v = pickle.load(w2v_file)
        else:
            raise Exception("Word2vec version not found!")
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, path=path,
                                   word_vectors=w2v, word_vector_yby=None, swc=sw_complete)
    return predictor


def predict_all_models(path, w2v_version, kw_i=0.0, kw_u=0.0, s=1940, t=1951, e=2010):
    print('--' + path)
    predictor = make_predictor(path, w2v_version, s, t, e)

    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='wavg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='frequency', cat_sim_method='wavg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='wavg', vec_sim_method='exp_euc_sq', kw=kw_i[2])
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='pt-avg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='pt-wavg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='frequency', cat_sim_method='pt-wavg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='frequency', cat_sim_method='wavg', vec_sim_method='exp_euc_sq', kw=kw_i[2])
    # predictor.predict_them_all(prior_dist='frequency_sqr', cat_sim_method='wavg_sqr', vec_sim_method='exp_euc_sq', kw=kw_i[0])
    # predictor.predict_them_all(prior_dist='frequency_log', cat_sim_method='wavg_log', vec_sim_method='exp_euc_sq', kw=kw_i[0])

    # predictor.predict_them_all(prior_dist='items', cat_sim_method='nn', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='nn', vec_sim_method='exp_euc_sq', kw=kw_i[0])
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='5nn', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='5nn', vec_sim_method='exp_euc_sq', kw=kw_i[1])
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='avg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='avg', vec_sim_method='exp_euc_sq', kw=kw_i)
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt0-avg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt-avg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt-wavg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt-wavg', vec_sim_method='exp_euc_sq', kw=kw_i[3])
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt-avg', vec_sim_method='exp_euc_sq', kw=kw_i[0])
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt0-mode', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt-mode', vec_sim_method='exp_euc_sq')

    # predictor.predict_them_all(prior_dist='items', cat_sim_method='gkde', vec_sim_method='')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='wavg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='items', cat_sim_method='eavg', vec_sim_method='exp_euc_sq')

    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='nn', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='5nn', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='avg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='avg', vec_sim_method='exp_euc_sq', kw=kw_u)
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='pt0-avg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='pt-avg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='pt0-mode', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='pt-mode', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='pt-wavg', vec_sim_method='exp_euc_sq')

    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='gkde', vec_sim_method='')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='wavg', vec_sim_method='exp_euc_sq')
    # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='eavg', vec_sim_method='exp_euc_sq')

    # predictor.predict_them_all(prior_dist='items', cat_sim_method='one', vec_sim_method='')
    predictor.predict_them_all(prior_dist='frequency', cat_sim_method='one', vec_sim_method='')


def predict_kernel_widths(path, w2v_version, s=1940, t=1941, e=1950):
    print('--' + path)

    predictor = make_predictor(path, w2v_version, s, t, e)
    kws = [i / 10.0 for i in range(1, 11)] + [float(i) for i in range(2, 101)]

    for kw in kws:
        predictor.predict_them_all(prior_dist='items', cat_sim_method='nn', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='items', cat_sim_method='5nn', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt-avg', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='items', cat_sim_method='pt-wavg', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='items', cat_sim_method='avg', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='avg', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='frequency', cat_sim_method='wavg', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='uniform', cat_sim_method='wavg', vec_sim_method='exp_euc_sq', kw=kw)
        # predictor.predict_them_all(prior_dist='frequency_sqr', cat_sim_method='wavg_sqr', vec_sim_method='exp_euc_sq',
        #                            kw=kw)
        # predictor.predict_them_all(prior_dist='frequency_log', cat_sim_method='wavg_log', vec_sim_method='exp_euc_sq',
        #                            kw=kw)


def find_neighbours(path, w2v_version, s=1940, t=1951, e=2010):
    predictor = make_predictor(path, w2v_version, s, t, e)
    # predictor.find_neighbouring_words(word='博客', category='个', year=2003)
    # predictor.find_neighbouring_words(word='网民', category='名', year=2001)
    # predictor.find_neighbouring_words(word='鼠标', category='个', year=1996)
    # predictor.find_neighbouring_words(word='便利店', category='家', year=1992)

    # predictor.find_neighbouring_words(word='浏览器', category='个', year=1997)
    # predictor.find_neighbouring_words(word='防火墙', category='个', year=2003)
    # predictor.find_neighbouring_words(word='终端', category='个', year=2003)

    # predictor.find_neighbouring_words(word='互联网', category='代', year=1995)

    # predictor.find_neighbouring_words(word='玩家', category='名', year=1997)
    # predictor.find_neighbouring_words(word='技师', category='名', year=1997)
    # predictor.find_neighbouring_words(word='同伴', category='名', year=1997)

    # predictor.find_neighbouring_words(word='路由器', category='个', year=1995)

    predictor.find_neighbouring_words(word='相机', category='款', year=1993)
    # 相机    款 1993
    # 电脑 1.6881220515509874e-22 	 [('部', 1958), ('台', 1978), ('代', 1981), ('款', 1988)]
    # 光盘 1.2122898876345795e-23 	 [('张', 1949), ('套', 1988)]
    predictor.find_neighbouring_words(word='电脑', category='款', year=1988)
    predictor.find_neighbouring_words(word='光盘', category='套', year=1988)
    # predictor.find_neighbouring_words(word='软件', category='款', year=1993)
    # predictor.find_neighbouring_words(word='服务器', category='组', year=1992)
    # predictor.find_neighbouring_words(word='防火墙', category='道', year=1992)
    # predictor.find_neighbouring_words(word='视频', category='段', year=1997)


# predict for all 4 word vector versions:
predict_all_models(path='chi-w2v-yby-all', w2v_version='orig', kw_i=52.0, kw_u=0.8, s=1940, t=1951, e=2010)
predict_all_models(path='chi-w2v-yby-pca-all', w2v_version='PCA', kw_i=48.0, kw_u=0.5, s=1940, t=1951, e=2010)
predict_all_models(path='chi-w2v-yby-s0.5lda-fixed-all', w2v_version='s0.5LDA-F', kw_i=4.0, kw_u=0.2, s=1940, t=1951, e=2010)
predict_all_models(path='chi-w2v-yby-s0.5lda-all', w2v_version='s0.5LDA', kw_i=4.0, kw_u=0.2, s=1940, t=1951, e=2010)

# predict for different kernel widths in the 40s for all 4 word vector versions
# predict_kernel_widths(path='chi-w2v-yby-40s', w2v_version='orig', s=1940, t=1941, e=1950)
# predict_kernel_widths(path='chi-w2v-yby-pca-40s', w2v_version='PCA', s=1940, t=1941, e=1950)
# predict_kernel_widths(path='chi-w2v-yby-s0.5lda-fixed-40s', w2v_version='s0.5LDA-F', s=1940, t=1941, e=1950)
# predict_kernel_widths(path='chi-w2v-yby-s0.5lda-40s', w2v_version='s0.5LDA', s=1940, t=1941, e=1950)

# predict for different kernel widths in all years for all 4 word vector versions
# predict_kernel_widths(path='chi-w2v-yby-all', w2v_version='orig', s=1940, t=1941, e=2010)
# predict_kernel_widths(path='chi-w2v-yby-pca-all', w2v_version='PCA', s=1940, t=1941, e=2010)
# predict_kernel_widths(path='chi-w2v-yby-s0.5lda-fixed-all', w2v_version='s0.5LDA-F', s=1940, t=1941, e=2010)
# predict_kernel_widths(path='chi-w2v-yby-s0.5lda-all', w2v_version='s0.5LDA', s=1940, t=1941, e=2010)

# # predict for all 4 word vector versions with dynamic kernel width:
# models = {'orig': 'chi-w2v-yby-all',
#           'PCA': 'chi-w2v-yby-pca-all',
#           's0.5LDA': 'chi-w2v-yby-s0.5lda-all',
#           's0.5LDA-F': 'chi-w2v-yby-s0.5lda-fixed-all'}
#
# # models = {'s0.5LDA': 'chi-w2v-yby-s0.5lda-all'}
#
# for wv, path in models.items():
#     with open('./predictions/' + path + '/kernels.pkl', 'rb') as k_file:
#         k1 = pickle.load(k_file)
#     # method_names = ['items-nn-exp_euc_sq_', 'items-5nn-exp_euc_sq_', 'items-pt-avg-exp_euc_sq_',
#     #                 'items-pt-wavg-exp_euc_sq_','frequency-wavg-exp_euc_sq_']
#
#     # method_names = ['frequency-wavg-exp_euc_sq_']
#     # method_names = ['frequency_sqr-wavg_sqr-exp_euc_sq_']
#     # method_names = ['frequency_log-wavg_log-exp_euc_sq_']
#     method_names = [
#         # 'items-pt-avg-exp_euc_sq_',
#         'uniform-pt-avg-exp_euc_sq_',
#         'frequency-pt-wavg-exp_euc_sq_',
#         'uniform-wavg-exp_euc_sq_',
#         # 'frequency-wavg-exp_euc_sq_',
#     ]
#
#     # 0.9 2.0 7.0
#     ki = [k1[mn] for mn in method_names]
#     # ki = [k1[mn][1951] for mn in method_names]
#     # print(ki)
#     # ki = k1['items']
#     # ku = k1['uniform']
#     print(path)
#     print(ki[2][1950])
#     predict_all_models(path=path, w2v_version=wv, kw_i=ki)
