import copy
import os

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from the_analyser import SuperPredictionAnalyser
from the_predictor import SuperPredictor
import pickle
from paths import *


def load_super_words():
    # load super words
    with open(super_words_path, 'rb') as super_file:
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
    # print('done removing!')

    # remove categories that are empty after 1940
    for empty_cat in ['枝', '桌', '课', '进', '丝', '记', '盏', '夥', '轴', '尾', '针']:
        for y in range(1940, 2010):
            s_words[y].pop(empty_cat, None)
    return s_words, sw_complete


def make_predictor_object(path, w2v_version, s, t, e):
    """find the word vector to load and then make a predictor object with it"""
    s_words, sw_complete = load_super_words()

    if 'LDA' in w2v_version or 'PCA' in w2v_version:
        with open(w2v_init + w2v_version + '.pkl', 'rb') as infile:
            w2v_yby = pickle.load(infile)
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, path=path,
                                   word_vectors=None, word_vector_yby=w2v_yby, swc=sw_complete)
    else:
        with open(w2v_path, 'rb') as w2v_file:
            w2v = pickle.load(w2v_file)
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, path=path,
                                   word_vectors=w2v, word_vector_yby=None, swc=sw_complete)
    return predictor


def predict_with_all_kernel_widths(path, w2v_version, kws, models, s=1940, t=1941, e=2010):
    predictor = make_predictor_object(path, w2v_version, s, t, e)

    for kw in kws:
        # do each kernel width for all different prior distribution and category similarity method
        for m in models:
            predictor.predict_them_all(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2], kw=kw)


def predict_and_get_precision(path, name, s_words, pred, m, kw, year):
    if kw <= 0:
        return 0
    file_name = './predictions/' + path + '/' + m[0] + '-' + m[1] + '-' + m[2] + '_' + str(kw) + '.pkl'
    if not os.path.exists(file_name):
        pred.predict_them_all(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2], kw=kw)
    with open(file_name, 'rb') as p_file:
        prs = {name: pickle.load(p_file)}
    # return SuperPredictionAnalyser.compute_precision(s_words, prs[name], first_year=1940, last_year=year) #TODO
    return SuperPredictionAnalyser.compute_precision(s_words, prs[name], first_year=year, last_year=year)


def predict_and_get_precision_preset(kw):
    # print('||', kw)
    if type(kw) != float:
        kw = kw[0]
    global o_path, o_name, o_s_words, o_pred, o_m, o_year
    if kw <= 0:
        return 0
    kw = float(round(kw, tol))
    file_name = './predictions/' + o_path + '/' + o_m[0] + '-' + o_m[1] + '-' + o_m[2] + '_' + str(kw) + '.pkl'
    if not os.path.exists(file_name):
        o_pred.predict_them_all(prior_dist=o_m[0], cat_sim_method=o_m[1], vec_sim_method=o_m[2], kw=kw)
    with open(file_name, 'rb') as p_file:
        prs = {o_name: pickle.load(p_file)}
    # return - SuperPredictionAnalyser.compute_precision(o_s_words, prs[o_name], first_year=1940, last_year=o_year) #TODO
    r = - SuperPredictionAnalyser.compute_precision(o_s_words, prs[o_name], first_year=o_year, last_year=o_year)
    print('\\\\', kw, r)
    return r


def optimize_kernel_widths(path, w2v_version, models, kwmax, kwmin, s=1940, t=1941, e=2010, std=False):
    global o_path, o_name, o_s_words, o_pred, o_m, o_year, tol
    # perform a binary search
    s_words, sw_complete = load_super_words()
    kernels = dict()
    method_names = {m[0] + '-' + m[1] + '-' + m[2] + '_': m for m in models}
    if len(method_names) == 0:
        return
    for name, m in method_names.items():
        # print('--', name)
        kernels[name] = dict()
        predictor = make_predictor_object(path, w2v_version, s, t, e)
        for year in range(THRESHOLD, END):
            if not std:
                decimal_points = 2
                min_step = pow(0.1, decimal_points)
                kw_min = kwmin
                kw_max = kwmax
                step_size = round((kw_max - kw_min) / 10.0, decimal_points)
                best_kw = 1.0
                best_prc = 0.0
                while step_size > min_step:
                    current_kw = kw_min
                    while current_kw <= kw_max:
                        sf = '%.' + str(decimal_points) + 'f'
                        current_kw = float(sf % round(current_kw, decimal_points))
                        # find the precision for the point
                        prc = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                                                        kw=current_kw, year=year - 1)
                        print(name, current_kw, '|', prc)
                        if prc > best_prc:
                            best_prc = prc
                            best_kw = current_kw
                        current_kw += step_size
                    kw_min = best_kw - 1.0 * step_size
                    kw_max = best_kw + 1.0 * step_size
                    step_size = round(step_size / 5.0, decimal_points)
                kernels[name][year] = best_kw
            else:
                o_path, o_name, o_s_words, o_pred, o_m, o_year, tol = path, name, s_words, predictor, m, year - 1, 2
                # res = minimize_scalar(fun=predict_and_get_precision_preset, bracket=(0, 30), bounds=(0, 100),
                #                       method='brent', tol=0.1, options={'maxiter': 100, 'disp': True})
                # res = minimize_scalar(fun=predict_and_get_precision_preset, bounds=(0, 100),
                #                       method='Bounded', tol=0.001, options={'maxiter': 200, 'disp': 3})
                res = minimize(fun=predict_and_get_precision_preset, x0=np.array([1.0]), bounds=[(0.0, 100.0)],
                               tol=0.001, options={'maxiter': 200, 'disp': False})
                print(res, res.x)
                kernels[name][year] = res.x[0]
        print('best_kw for', name, ':', kernels[name])
    if std:
        sufx = '-opt'
    else:
        sufx = ''
    with open('./predictions/' + path + '/kernels' + sufx + '.pkl', 'wb') as k_file:
        pickle.dump(kernels, k_file)


def predict_all_models(path, w2v_version, models, s=1940, t=1950, e=2010, std=False):
    predictor = make_predictor_object(path, w2v_version, s, t, e)

    if std:
        sufx = '-opt'
    else:
        sufx = ''
    with open('./predictions/' + path + '/kernels' + sufx + '.pkl', 'rb') as k_file:
        best_kernel_widths = pickle.load(k_file)
    print('\n'.join(list(best_kernel_widths.keys())))
    for m in models:
        kws = m[3]
        if m[3] == 'adj':
            # kws = best_kernel_widths[m[0] + '-' + m[1] + '-' + m[2] + '_'][t - 1]
            kws = best_kernel_widths[m[0] + '-' + m[1] + '-' + m[2] + '_']
            print(m[0] + '-' + m[1] + '-' + m[2] + '_', kws)
        print('predicting for', m)
        predictor.predict_them_all(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2], kw=kws)
