import pickle
from the_analyser import SuperPredictionAnalyser

import matplotlib.pyplot as plt

c = ['#E58606', '#99C945', '#52BCA3', '#5D69B1', '#CC61B0', '#DAA51B', '#24796C', '#2F8AC4', '#764E9F', '#ED645A']


def load_super_words():
    # load super words
    with open('./data/super_words-chi-Luis-YbyY(w2v).pkl', 'rb') as super_file:
        s_words = pickle.load(super_file)

    # remove context from next years
    for y in range(1940, 2010):
        for cat, words in s_words[y].items():
            for appeared_word in words.keys():
                # remove the word from the next years
                for next_year in range(y + 1, 2010):
                    s_words[next_year][cat].pop(appeared_word, None)
    print('done removing!')
    return s_words


def do_analysis(path, models, t=1951):
    s_words = load_super_words()
    with open('./predictions/' + path + '/kernels.pkl', 'rb') as k_file:
        best_kernel_widths = pickle.load(k_file)

    colors = [c[0], c[1], c[2], c[3], c[4]]

    all_methods = []
    for m in models:
        if m[3] == 'adj':
            m[3] = best_kernel_widths[m[0] + '-' + m[1] + '-' + m[2] + '_'][t - 1]
        all_methods.append(m[0] + '-' + m[1] + '-' + m[2] + '_' + str(m[3]))
    prs = dict()
    for m in all_methods:
        with open('./predictions/' + path + '/' + m + '.pkl', 'rb') as p_file:
            prs[m] = pickle.load(p_file)

    plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=colors, baselines=None,
                                      all_methods=all_methods, lang=path)
    plotter.just_get_precision()


def plot_time_series():
    s_words = load_super_words()

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    method_i = [
        {'lang': 'chi-w2v-yby-all', 'kw_i': 52.0, 'kw_u': 0.8,
         'name': 'Full'},
        {'lang': 'chi-w2v-yby-pca-all', 'kw_i': 48.0, 'kw_u': 0.5,
         'name': 'PCA-reduced'},
        {'lang': 'chi-w2v-yby-s0.5lda-fixed-all', 'kw_i': 4.0, 'kw_u': 0.2,
         'name': 'FDA-reduced (static)'},
        {'lang': 'chi-w2v-yby-s0.5lda-all', 'kw_i': 4.0, 'kw_u': 0.2,
         'name': 'FDA-reduced (dynamic)'}]

    for i in range(len(method_i)):
        names = ['Baseline (s)']
        all_methods = ['items-one-_1.0']

        names += ['1nn', '5nn', 'Exemplar (s=1)', 'Exemplar', 'Prototype']
        all_methods += ['items-nn-exp_euc_sq_1.0', 'items-5nn-exp_euc_sq_1.0', 'items-avg-exp_euc_sq_1.0']
        all_methods += ['items-avg-exp_euc_sq_' + str(method_i[i]['kw_i'])]
        all_methods += ['items-pt-avg-exp_euc_sq_1.0']

        prs = dict()
        for m in all_methods:
            with open('./predictions/' + method_i[i]['lang'] + '/' + m + '.pkl', 'rb') as p_file:
                prs[m] = pickle.load(p_file)

        plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=c, baselines=None,
                                          all_methods=all_methods, lang=method_i[i]['lang'])
        # plotter.precision_over_time(name=method_i[i]['name'])

        # plotter.precision_over_time_plot(name=method_i[i]['name'], ax=ax[int(i / 2), i % 2], items=method_i[i]['pr'],
        #                                  items_names=names)
        # plotter.precision_over_time_plot(name=method_i[i]['name'], ax=ax[int(i / 2), i % 2], items_names=names)
        plotter.precision_over_time_plot(name=method_i[i]['name'], ax=ax[int(i / 2), i % 2], items_names=names,
                                         average=5)

    ax[1, 0].legend(loc='lower left', ncol=2, fontsize=11)
    ax[1, 0].set_xlabel('Years', fontsize=12)
    ax[1, 1].set_xlabel('Years', fontsize=12)
    ax[0, 0].set_ylabel('Predictive Accuracy (%)', fontsize=12)
    ax[1, 0].set_ylabel('Predictive Accuracy (%)', fontsize=12)
    print('saving...')
    plt.tight_layout()
    name = 'precisions_over_time'
    name = 'precisions_over_time-5ave'
    fig.savefig('./predictions/' + name + '.png', bbox_inches='tight')
    fig.savefig('./predictions/' + name + '.pdf', format='pdf', transparent=True, bbox_inches='tight')
    fig.savefig('./predictions/' + name + '.eps', format='eps', transparent=True, bbox_inches='tight')
    fig.clear()
    plt.clf()


def find_best_kernel_widths(path, kws, models):
    s_words = load_super_words()

    # find the best kernel width until each year
    method_names = [m[0] + '-' + m[1] + '-' + m[2] + '_' for m in models]

    all_methods = []
    for kw in kws:
        for name in method_names:
            all_methods.append(name + str(kw))

    prs = dict()
    for m in all_methods:
        with open('./predictions/' + path + '/' + m + '.pkl', 'rb') as p_file:
            pr = pickle.load(p_file)
        prs[m] = pr

    plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=[], baselines=None,
                                      all_methods=all_methods, lang=path)
    kernels = dict()
    for name in method_names:
        print('--', name)
        kernels[name] = dict()
        for year in range(1949, 2009):
            kernels[name][year + 1] = plotter.best_kernel_width(kws=kws, end_year=year, method=name)

    with open('./predictions/' + path + '/kernels.pkl', 'wb') as k_file:
        pickle.dump(kernels, k_file)


def compare_predictions():
    path = 'chi-w2v-yby-s0.5lda-all'
    ms = ['items-pt-wavg-exp_euc_sq_dynKW', 'items-avg-exp_euc_sq_dynKW']

    prs = ['', '']
    with open('./predictions/' + path + '/' + ms[0] + '.pkl', 'rb') as p_file:
        prs[0] = pickle.load(p_file)
    with open('./predictions/' + path + '/' + ms[1] + '.pkl', 'rb') as p_file:
        prs[1] = pickle.load(p_file)

    SuperPredictionAnalyser.overlap(prs[0], prs[1])
