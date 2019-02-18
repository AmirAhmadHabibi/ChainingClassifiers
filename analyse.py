import pickle
from the_analyser import SuperPredictionAnalyser

import matplotlib.pyplot as plt

c = ['#E58606', '#99C945', '#52BCA3', '#5D69B1', '#CC61B0', '#DAA51B', '#24796C', '#2F8AC4', '#764E9F', '#ED645A']

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


def do_analysis(path, name, kw_i, kw_u):
    print(path)
    all_methods = []
    # all_methods = ['items-nn-exp_euc_sq_1.0', 'items-5nn-exp_euc_sq_1.0', 'items-avg-exp_euc_sq_1.0']
    # all_methods += ['items-avg-exp_euc_sq_' + str(kw_i)]
    # all_methods += ['items-nn-exp_euc_sq_' + str(kw_i[0])]
    # all_methods += ['items-5nn-exp_euc_sq_' + str(kw_i[1])]
    # all_methods += ['items-pt-avg-exp_euc_sq_1.0']
    # all_methods += ['items-pt-avg-exp_euc_sq_' + str(kw_i[2])]
    # all_methods += ['items-pt-wavg-exp_euc_sq_1.0']
    # all_methods += ['items-pt-wavg-exp_euc_sq_' + str(kw_i[3])]
    # all_methods += ['items-pt-wavg-exp_euc_sq_dynKW']
    # all_methods += ['items-nn-exp_euc_sq_' + str(kw_i)]
    # all_methods += ['items-5nn-exp_euc_sq_' + str(kw_i)]
    # all_methods += ['items-pt-avg-exp_euc_sq_' + str(kw_i)]
    # all_methods += ['items-pt0-avg-exp_euc_sq_1.0']
    # all_methods += ['items-pt-mode-exp_euc_sq_1.0']
    # all_methods += ['items-wavg-exp_euc_sq_1.0', 'items-eavg-exp_euc_sq_1.0']
    # all_methods += ['uniform-nn-exp_euc_sq_1.0', 'uniform-5nn-exp_euc_sq_1.0', 'uniform-avg-exp_euc_sq_1.0']
    # all_methods += ['uniform-avg-exp_euc_sq_' + str(kw_u)]
    # all_methods += ['uniform-pt-avg-exp_euc_sq_1.0']
    # all_methods += ['uniform-pt0-avg-exp_euc_sq_1.0']
    # all_methods += ['uniform-pt-mode-exp_euc_sq_1.0']
    # all_methods += ['uniform-wavg-exp_euc_sq_1.0', 'uniform-eavg-exp_euc_sq_1.0']
    # all_methods += ['uniform-pt-wavg-exp_euc_sq_1.0']
    # all_methods += ['items-one-_1.0', 'frequency-one-_1.0']
    # all_methods += ['frequency-wavg-exp_euc_sq_dynKW']
    # all_methods += ['frequency_sqr-wavg_sqr-exp_euc_sq_dynKW']
    # all_methods += ['frequency_log-wavg_log-exp_euc_sq_dynKW']
    all_methods += [
        'items-pt-avg-exp_euc_sq_1.0',
        'uniform-pt-avg-exp_euc_sq_1.0',

        'uniform-wavg-exp_euc_sq_1.0',
        'uniform-wavg-exp_euc_sq_dynKW',
        'uniform-pt-wavg-exp_euc_sq_1.0',

        'frequency-wavg-exp_euc_sq_1.0',
        'frequency-wavg-exp_euc_sq_dynKW',
        'frequency-pt-wavg-exp_euc_sq_1.0',
        'items-one-_1.0',
        'frequency-one-_1.0'
    ]

    colors = [c[0], c[1], c[2], c[3], c[4]]

    prs = dict()
    for m in all_methods:
        with open('./predictions/' + path + '/' + m + '.pkl', 'rb') as p_file:
            prs[m] = pickle.load(p_file)

    plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=colors, baselines=None,
                                      all_methods=all_methods, lang=path)
    # plotter.find_100_most_recent()
    plotter.just_get_precision()
    # plotter.precision_over_time_plot(name=name)
    # plotter.bar_chart_it()
    # plotter.box_plot_it()
    # plotter.false_positive_it()
    # plotter.bar_plot_category_precision(category_number=138 - 12, name=name)
    # plotter.scatter_plot_category_precision_recall(name=name)


def do_the_kws(path):
    print(path)
    colors = []

    all_methods = []
    # kws = []
    # for i in range(1, 11):
    #     kws.append(i / 10.0)
    # for i in range(2, 100, 4):
    #     kws.append(float(i))
    kws = []
    for i in range(1, 11):
        kws.append(i / 10.0)
    # for i in range(2, 10, 1):
    #     kws.append(float(i))
    for i in range(2, 51, 2):
        kws.append(float(i))

    for kw in kws:
        all_methods.append('items-avg-exp_euc_sq_' + str(kw))
        all_methods.append('uniform-avg-exp_euc_sq_' + str(kw))

    clr = {'items': c[0], 'uniform': c[1]}

    prs = dict()
    for m in all_methods:
        with open('./predictions/' + path + '/' + m + '.pkl', 'rb') as p_file:
            pr = pickle.load(p_file)
        prs[m] = pr

    plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=colors, baselines=None,
                                      all_methods=all_methods, lang=path)
    # plotter.just_get_precision()
    plotter.plot_series(kws, colour=clr)

    # yby
    # Max for items: 14.0 0.3359050445103858
    # Max for uniform: 0.6 0.3195845697329377

    # yby 50s
    # Max for items: 30.0 0.45160642570281123
    # Max for uniform: 0.6 0.37248995983935745
    # fixed:
    # Max for items: 22.0 0.463855421686747
    # Max for uniform: 0.6 0.37248995983935745

    # yby 50s 2d
    # Max for items: 1.1
    # Max for uniform: 0.01

    # yby 50s 2d
    # Max for items: 34.0
    # Max for uniform: 0.1

    # yby 50s lda
    # Max for items: 6.0 0.44457831325301206
    # Max for uniform: 0.5 0.3240963855421687

    # yby 50s slda
    # Max for items: 8.0 0.49417670682730924
    # Max for uniform: 0.3 0.40883534136546185

    # yby 50s pca
    # Max for items: 26.0 0.4714859437751004
    # Max for uniform: 0.4 0.37730923694779117

    # yby 50s s0.5lda
    # Max for items: 5.0 0.521285140562249
    # Max for uniform: 0.2 0.41967871485943775

    # yby 40s
    # Max for items: 52.0 0.4744437763078773
    # Max for uniform: 0.8 0.31930246542393265

    # yby 40s s0.5lda
    # Max for items: 4.0 0.5147324113048707
    # Max for uniform: 0.2 0.42994588093806374

    # yby 40s s0.5lda fixed
    # Max for items: 4.0 0.5147324113048707
    # Max for uniform: 0.2 0.42994588093806374

    # yby 40s pca
    # Max for items: 48.0 0.4846662657847264
    # Max for uniform: 0.5 0.3367408298256164


def plot_time_series():
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    # method_i = [{'lang': 'chi-w2v-yby-51', 'kw_i': 52.0, 'kw_u': 0.8, 'name': 'Vanilla word2vec models'},
    #             {'lang': 'chi-w2v-yby-pca-51', 'kw_i': 48.0, 'kw_u': 0.5,
    #              'name': 'Dimension-reduced models (time-varying)'},
    #             {'lang': 'chi-w2v-yby-s0.5lda-fixed-51', 'kw_i': 4.0, 'kw_u': 0.2,
    #              'name': 'Category-biased models (static)'},
    #             {'lang': 'chi-w2v-yby-s0.5lda-51', 'kw_i': 4.0, 'kw_u': 0.2,
    #              'name': 'Category-biased models (time-varying)'}]

    # method_i = [
    #     {'lang': 'chi-w2v-yby-all', 'kw_i': 52.0, 'kw_u': 0.8,
    #      'name': 'Vanilla word2vec models', 'pr': vanilla},
    #     {'lang': 'chi-w2v-yby-pca-all', 'kw_i': 48.0, 'kw_u': 0.5,
    #      'name': 'Dimension-reduced models (dynamic)', 'pr': pca},
    #     {'lang': 'chi-w2v-yby-s0.5lda-fixed-all', 'kw_i': 4.0, 'kw_u': 0.2,
    #      'name': 'Category-weighted models (static)', 'pr': lda_s},
    #     {'lang': 'chi-w2v-yby-s0.5lda-all', 'kw_i': 4.0, 'kw_u': 0.2,
    #      'name': 'Category-weighted models (dynamic)', 'pr': lda_t}]
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

        # if method_i[i]['lang'] == 'chi-w2v-yby-s0.5lda-all':
        #     names += ['Prototype Kernel_s (s)', 'Prototype Kernel_t (s)']
        #     all_methods += ['items-pt-avg-exp_euc_sq_' + str(7.0)]
        #     all_methods += ['items-pt-avg-exp_euc_sq_dynKW']

        colors = c

        prs = dict()
        for m in all_methods:
            with open('./predictions/' + method_i[i]['lang'] + '/' + m + '.pkl', 'rb') as p_file:
                prs[m] = pickle.load(p_file)

        plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=colors, baselines=None,
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


def find_best_kernel_widths(path, kws):
    # find the best kernel width until each year
    models = [['uniform', 'avg', 'exp_euc_sq'],
              ['items', 'avg', 'exp_euc_sq'],
              ['uniform', 'wavg', 'exp_euc_sq'],
              ['frequency', 'wavg', 'exp_euc_sq']]
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

# do_analysis(path='chi-w2v-yby-51', kw_i=52.0, kw_u=0.8, name='Vanilla')
# do_analysis(path='chi-w2v-yby-pca-51', kw_i=48.0, kw_u=0.5, name='Dimension-reduced')
# do_analysis(path='chi-w2v-yby-s0.5lda-51', kw_i=4.0, kw_u=0.2, name='Category-biased')
# do_analysis(path='chi-w2v-yby-s0.5lda-fixed-51', kw_i=4.0, kw_u=0.2, name='Category-biased(Fixed)')
#
# do_analysis(path='chi-w2v-yby', kw_i=22.0, kw_u=0.6)
# do_analysis(path='chi-w2v-yby-s0.5lda', kw_i=5.0, kw_u=0.2)
# do_analysis(path='chi-w2v-yby-pca', kw_i=26.0, kw_u=0.4)
#
# do_the_kws(path='chi-w2v-yby-40s')
# do_the_kws(path='chi-w2v-yby-s0.5lda-40s')
# do_the_kws(path='chi-w2v-yby-s0.5lda-fixed-40s')
# do_the_kws(path='chi-w2v-yby-pca-40s')
#
# plot_time_series()
#

# analise dynamic kernels
# do_analysis(path='chi-w2v-yby-all', kw_i='dynKW', kw_u=0.9, name='Vanilla')
# do_analysis(path='chi-w2v-yby-pca-all', kw_i='dynKW', kw_u=0.5, name='Dimension-reduced')
# do_analysis(path='chi-w2v-yby-s0.5lda-fixed-all', kw_i='dynKW', kw_u=0.2, name='Category-biased(Fixed)')
# do_analysis(path='chi-w2v-yby-s0.5lda-all', kw_i='dynKW', kw_u=0.2, name='Category-biased')
# do_analysis(path='chi-w2v-yby-s0.5lda-all', kw_i=[0.9, 2.0, 7.0, 2.0], kw_u='', name='Category-biased')
#
# x_names = [r'$full$', r'$PCA-reduced$', r'$FDA-reduced\:(static)$', r'$FDA-reduced\:(dynamic)$']
# y_names = [r'$1nn$', r'$5nn$', r'$exemplar\:(s=1)$', r'$exemplar$', r'$prototype$']
# precisions = [[39.1, 39.2, 38.6, 34.5, 26.1],
#               [39.1, 39.2, 38.7, 35.0, 23.3],
#               [40.8, 41.2, 42.0, 44.0, 38.6],
#               [42.9, 43.5, 44.2, 46.7, 40.7]]
# SuperPredictionAnalyser.barplot3d(precisions, x_names, y_names, baseline=29.9)

# compare_predictions()
