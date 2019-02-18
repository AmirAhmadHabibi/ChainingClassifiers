import copy
from the_predictor import SuperPredictor
import pickle


def load_super_words():
    # load super words
    with open('./data/super_words-chi-Luis-YbyY(w2v).pkl', 'rb') as super_file:
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
    return s_words, sw_complete


def make_predictor_object(path, w2v_version, s, t, e):
    """find the word vector to load and then make a predictor object with it"""
    s_words, sw_complete = load_super_words()

    if 'LDA' in w2v_version or 'PCA' in w2v_version:
        with open('./data/w2v-chi-yby-' + w2v_version + '.pkl', 'rb') as infile:
            w2v_yby = pickle.load(infile)
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, path=path,
                                   word_vectors=None, word_vector_yby=w2v_yby, swc=sw_complete)
    else:
        with open('./data/w2v-chi-yby.pkl', 'rb') as w2v_file:
            w2v = pickle.load(w2v_file)
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, path=path,
                                   word_vectors=w2v, word_vector_yby=None, swc=sw_complete)
    return predictor


def predict_with_all_kernel_widths(path, w2v_version, kws, models, s=1940, t=1941, e=1950):
    predictor = make_predictor_object(path, w2v_version, s, t, e)

    for kw in kws:
        # do each kernel width for all different prior distribution and category similarity method
        for m in models:
            predictor.predict_them_all(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2], kw=kw)


def predict_all_models(path, w2v_version, models, s=1940, t=1951, e=2010):
    predictor = make_predictor_object(path, w2v_version, s, t, e)

    with open('./predictions/' + path + '/kernels.pkl', 'rb') as k_file:
        best_kernel_widths = pickle.load(k_file)

    for m in models:
        if m[3] == 'adj':
            m[3] = best_kernel_widths[m[0] + '-' + m[1] + '-' + m[2] + '_'][t - 1]
        predictor.predict_them_all(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2], kw=m[3])
