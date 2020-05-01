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
    for y in range(START, END, STEP):
        for c, words in s_words[y].items():
            for appeared_word in words.keys():
                # remove the word from the next years
                for next_year in range(y + STEP, END, STEP):
                    s_words[next_year][c].pop(appeared_word, None)
    # print('done removing!')

    # # remove categories that are empty after 1940
    # for empty_cat in ['枝', '桌', '课', '进', '丝', '记', '盏', '夥', '轴', '尾', '针']:
    #     for y in range(1940, 2010):
    #         s_words[y].pop(empty_cat, None)
    return s_words, sw_complete


def make_predictor_object(path, w2v_version, s, t, e, step):
    """find the word vector to load and then make a predictor object with it"""
    s_words, sw_complete = load_super_words()
    if 'en-w2v-all' in path:
        with open(w2v_path, 'rb') as infile:
            w2v_yby = pickle.load(infile)
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, step=step, path=path,
                                   word_vectors=None, word_vector_yby=w2v_yby, swc=sw_complete)
    elif 'LDA' in w2v_version or 'PCA' in w2v_version:
        with open(w2v_init + w2v_version + '.pkl', 'rb') as infile:
            w2v_yby = pickle.load(infile)
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, step=step, path=path,
                                   word_vectors=None, word_vector_yby=w2v_yby, swc=sw_complete)
    else:
        with open(w2v_path, 'rb') as w2v_file:
            w2v = pickle.load(w2v_file)
        predictor = SuperPredictor(super_words=s_words, hbc=None, start=s, threshold=t, end=e, step=step, path=path,
                                   word_vectors=w2v, word_vector_yby=None, swc=sw_complete)
    return predictor


def predict_with_all_kernel_widths(path, w2v_version, kws, models, s=1940, t=1941, e=2010, step=1):
    predictor = make_predictor_object(path, w2v_version, s, t, e, step)

    for kw in kws:
        # do each kernel width for all different prior distribution and category similarity method
        for m in models:
            predictor.predict_them_all(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2], kw=kw)


def predict_and_get_precision(path, name, s_words, pred, m, kw, year, start_year=0):
    if kw <= 0:
        return 0
    file_name = './predictions/' + path + '/' + m[0] + '-' + m[1] + '-' + m[2] + '_' + str(kw) + '.pkl'
    if not os.path.exists(file_name):
        pred.predict_them_all(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2], kw=kw)
    with open(file_name, 'rb') as p_file:
        prs = {name: pickle.load(p_file)}
    # return SuperPredictionAnalyser.compute_precision(s_words, prs[name], first_year=1940, last_year=year) #TODO
    return SuperPredictionAnalyser.compute_precision(s_words, prs[name], first_year=year - start_year, last_year=year)


def get_llp_preset(kw):
    # print('||', kw)
    if type(kw) not in {np.float64, float, int}:
        kw = kw[0]
    global o_path, o_name, o_s_words, o_pred, o_m, o_year, llp
    if kw <= 0:
        return 10000000000000000
    # kw = float(round(kw, tol))
    # file_name = './predictions/' + o_path + '/llp- ' + o_m[0] + '-' + o_m[1] + '-' + o_m[2] + '_' + str(kw) + '.pkl'
    # if not os.path.exists(file_name):
    #     o_pred.predict_them_all(prior_dist=o_m[0], cat_sim_method=o_m[1], vec_sim_method=o_m[2], kw=kw)
    #
    # with open(file_name, 'rb') as p_file:
    #     prs = {o_name: pickle.load(p_file)}
    # # return - SuperPredictionAnalyser.compute_precision(o_s_words, prs[o_name], first_year=1940, last_year=o_year) #TODO
    # r = - SuperPredictionAnalyser.compute_precision(o_s_words, prs[o_name], first_year=o_year, last_year=o_year)
    # print('\\\\', kw, r)

    # if not os.path.exists(file_name):
    res = - o_pred.log_likelihood_of_posterior(year=o_year, kw=kw)
    # print('||', kw, res)
    # else:
    #     with open(file_name, 'rb') as p_file:
    #         llp = pickle.load(p_file)
    #         res = -llp[o_year]
    #     print('|', kw, res)
    return res


def optimize_kernel_widths(path, w2v_version, models, kwmax, kwmin, s=1940, t=1941, e=2010, step=1, mode='mine'):
    global o_path, o_name, o_s_words, o_pred, o_m, o_year, tol, llp, o_kw
    # perform a binary search
    s_words, sw_complete = load_super_words()
    kernels = dict()
    method_names = {m[0] + '-' + m[1] + '-' + m[2] + '_': m for m in models}
    if len(method_names) == 0:
        return
    llp = {}
    if mode == 'sanity':
        sufx = '-snt'
        with open('./predictions/' + path + '/kernels' + sufx + '.pkl', 'rb') as k_file:
            best_kernel_widths = pickle.load(k_file)
    for name, m in method_names.items():
        # print('--', name)
        kernels[name] = dict()
        predictor = make_predictor_object(path, w2v_version, s, t, e, step)
        predictor.set_params(prior_dist=m[0], cat_sim_method=m[1], vec_sim_method=m[2])
        o_path, o_name, o_s_words, o_pred, o_m, tol = path, name, s_words, predictor, m, 2
        for year in range(THRESHOLD, END, STEP):
            o_year = year - STEP
            o_kw = kernels[name]
            if mode == 'mine':
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
                        # prc = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                        #                                 kw=current_kw, year=year - 1)

                        prc = -get_llp_preset(current_kw)
                        # print(name, current_kw, '|', prc)
                        if prc > best_prc:
                            best_prc = prc
                            best_kw = current_kw
                        current_kw += step_size
                    kw_min = best_kw - 1.0 * step_size
                    kw_max = best_kw + 1.0 * step_size
                    step_size = round(step_size / 5.0, decimal_points)
                print('--', year, best_kw)
                kernels[name][year] = best_kw
            if mode == 'prc':
                decimal_points = 1
                step_size = 0.1
                best_kw = 1.0
                best_prc = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                                                     kw=best_kw, year=year - STEP)
                current_kw = kwmin + step_size
                while current_kw <= kwmax:
                    if current_kw == 1.0:
                        step_size = 1.0
                    sf = '%.' + str(decimal_points) + 'f'
                    current_kw = float(sf % round(current_kw, decimal_points))
                    # find the precision for the point
                    prc = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                                                    kw=current_kw, year=year - STEP)
                    # prc = -get_llp_preset(current_kw)
                    # print(name, current_kw, '|', prc)
                    if prc > best_prc:
                        best_prc = prc
                        best_kw = current_kw
                    current_kw += step_size
                print('--', year, best_kw)
                kernels[name][year] = best_kw
            if mode == 'prc-10':
                decimal_points = 1
                step_size = 0.1
                best_kw = 1.0
                best_prc = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                                                     kw=best_kw, year=year - STEP, start_year=year - 10 * STEP)
                current_kw = kwmin + step_size
                while current_kw <= kwmax:
                    if current_kw == 1.0:
                        step_size = 1.0
                    sf = '%.' + str(decimal_points) + 'f'
                    current_kw = float(sf % round(current_kw, decimal_points))
                    # find the precision for the point
                    prc = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                                                    kw=current_kw, year=year - STEP)
                    # prc = -get_llp_preset(current_kw)
                    # print(name, current_kw, '|', prc)
                    if prc > best_prc:
                        best_prc = prc
                        best_kw = current_kw
                    current_kw += step_size
                print('--', year, best_kw)
                kernels[name][year] = best_kw
            elif mode == 'std':
                # if year < 1967 or year > 1973:
                # res = minimize_scalar(fun=get_llp_preset, bracket=(0, 30), bounds=(0, 100),
                #                       method='brent', tol=0.1, options={'maxiter': 100, 'disp': True})
                res = minimize_scalar(fun=get_llp_preset, bounds=(0, 100), method='Bounded', tol=0.001)
                # res = minimize(fun=get_llp_preset, method='L-BFGS-B', x0=np.array([1.0]),
                #                bounds=[(0.01, 100.0)])  # , tol=0.01 , options={'maxiter': 200, 'disp': False}
                print('--', year, res.x)
                # kernels[name][year] = res.x[0]
                kernels[name][year] = res.x
            elif mode == 'sanity':
                o_year = year
                x = best_kernel_widths[name][year + 1]
                # res = minimize_scalar(fun=get_llp_preset, bounds=(0, 100), method='Bounded', tol=0.001)
                # llp1 = - o_pred.log_likelihood_of_posterior(year=o_year, kw=res.x)
                llp1 = - o_pred.log_likelihood_of_posterior(year=o_year, kw=x)
                llp2 = - o_pred.log_likelihood_of_posterior(year=o_year, kw=1.0)
                if llp2 < llp1:
                    print('||||||>>>>>', llp1, llp2)
                # kernels[name][year]=res.x
                # prc1 = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                #                                                                 kw=kernels[name][year], year=year)
                # kernels[name][year]=1.0
                # prc2 = predict_and_get_precision(path=path, name=name, s_words=s_words, pred=predictor, m=m,
                #                                                                 kw=kernels[name][year], year=year)
                # print('::::::::::::::', prc1, prc2)
                # print('--', year, res.x)
                # kernels[name][year] = res.x
                kernels[name][year] = x
        print('best_kw for', name, ':', kernels[name])

    if mode == 'std':
        sufx = '-opt'
    elif mode == 'mine':
        sufx = ''
    elif mode == 'sanity':
        sufx = '-snt'
    elif mode == 'prc':
        sufx = '-prc'
    elif mode == 'prc-10':
        sufx = '-prc-10'

    if os.path.exists('./predictions/' + path + '/kernels' + sufx + '.pkl'):
        with open('./predictions/' + path + '/kernels' + sufx + '.pkl', 'rb') as k_file:
            best_kernel_widths = pickle.load(k_file)
    else:
        best_kernel_widths = {}

    for name, kws in kernels.items():
        best_kernel_widths[name] = kws
    with open('./predictions/' + path + '/kernels' + sufx + '.pkl', 'wb') as k_file:
        pickle.dump(best_kernel_widths, k_file)


def predict_all_models(path, w2v_version, models, s=1940, t=1950, e=2010, step=1, mode='std'):
    predictor = make_predictor_object(path, w2v_version, s, t, e, step)

    if mode == 'std':
        sufx = '-opt'
    elif mode == 'mine':
        sufx = ''
    elif mode == 'sanity':
        sufx = '-snt'
    elif mode == 'prc':
        sufx = '-prc'
    elif mode == 'prc-10':
        sufx = '-prc-10'
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
