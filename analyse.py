import pickle
from the_analyser import SuperPredictionAnalyser
from paths import *
import matplotlib.pyplot as plt

c = ['#E58606', '#99C945', '#52BCA3', '#5D69B1', '#CC61B0', '#DAA51B', '#24796C', '#2F8AC4', '#764E9F', '#ED645A']


def load_super_words():
    # load super words
    with open(super_words_path, 'rb') as super_file:
        s_words = pickle.load(super_file)

    # remove context from next years
    for y in range(1940, 2010):
        for cat, words in s_words[y].items():
            for appeared_word in words.keys():
                # remove the word from the next years
                for next_year in range(y + 1, 2010):
                    s_words[next_year][cat].pop(appeared_word, None)
    # print('done removing!')
    return s_words


def do_analysis(path, models, path_n, llp=False):
    s_words = load_super_words()

    colors = [c[0], c[1], c[2], c[3], c[4]]

    all_methods = []
    for m in models:
        all_methods.append(m[0] + '-' + m[1] + '-' + m[2] + '_' + str(m[3]))
    prs = dict()
    for m in all_methods:
        try:
            with open('./predictions/' + path + '/' + m + '.pkl', 'rb') as p_file:
                prs[m] = pickle.load(p_file)
        except:
            print()

    analyser = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=colors, baselines=None,
                                       all_methods=all_methods, lang=path)
    if llp:
        s = analyser.just_get_llp(first_year=THRESHOLD, last_year=END)
        for i in range(int(len(s) / 2)):
            un = str(s[2 * i][1])
            it = str(s[2 * i + 1][1])
            print(path_n, models[2 * i][4], '&', un, '&', it, '\\\\', sep='\t')
    else:
        s = analyser.just_get_precision(first_year=THRESHOLD, last_year=END)
        for i in range(int(len(s) / 2)):
            un = str(s[2 * i][1]) + ' \% (' + str(s[2 * i][2]) + ')'
            it = str(s[2 * i + 1][1]) + ' \% (' + str(s[2 * i + 1][2]) + ')'
            print(path_n, models[2 * i][4], '&', un, '&', it, '\\\\', sep='\t')
            # print(s[2 * i], s[2 * i + 1])


def plot_time_series():
    s_words = load_super_words()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # method_i = [
    #     {'lang': 'chi-en-w2v-yby-all',
    #      'name': 'Full'},
    #     {'lang': 'chi-w2v-yby-pca-all',
    #      'name': 'PCA-reduced'},
    #     {'lang': 'chi-w2v-yby-s0.5lda-fixed-all',
    #      'name': 'FDA-reduced (static)'},
    #     {'lang': 'chi-w2v-yby-s0.5lda-all',
    #      'name': 'FDA-reduced (dynamic)'}]
    method_i = [
        {'lang': 'chi-en-w2v-yby-all',
         'name': 'Full'},
        # {'lang': 'chi-en-w2v-yby-pca_resz-all',
        #  'name': 'PCA-reduced'},
        # {'lang': 'chi-en-w2v-yby-s0.5lda-fixed_resz-all',
        #  'name': 'FDA-reduced (static)'},
        # {'lang': 'chi-en-w2v-yby-s0.5lda_resz-all',
        #  'name': 'FDA-reduced (dynamic)'}
    ]

    names = ['Baseline (s)', '1nn', '10nn', 'Exemplar (s=1)', 'Exemplar', 'Prototype']
    all_methods = ['items-one-_1.0', 'items-s1nn-exp_euc_sq_1.0', 'items-s10nn-exp_euc_sq_1.0',
                   'items-avg-exp_euc_sq_1.0', 'items-avg-exp_euc_sq_adj', 'items-pt-avg-exp_euc_sq_1.0']
    prs = dict()
    for m in all_methods:
        with open('./predictions/' + method_i[0]['lang'] + '/' + m + '.pkl', 'rb') as p_file:
            prs[m] = pickle.load(p_file)
    plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=c, baselines=None,
                                      all_methods=all_methods, lang=method_i[0]['lang'])
    plotter.precision_over_time_plot(name='size-based', ax=ax[1], items_names=names, average=5)

    all_methods = ['uniform-one-_1.0', 'uniform-s1nn-exp_euc_sq_1.0', 'uniform-s10nn-exp_euc_sq_1.0',
                   'uniform-avg-exp_euc_sq_1.0', 'uniform-avg-exp_euc_sq_adj', 'uniform-pt-avg-exp_euc_sq_1.0']
    prs = dict()
    for m in all_methods:
        with open('./predictions/' + method_i[0]['lang'] + '/' + m + '.pkl', 'rb') as p_file:
            prs[m] = pickle.load(p_file)
    plotter = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=c, baselines=None,
                                      all_methods=all_methods, lang=method_i[0]['lang'])
    plotter.precision_over_time_plot(name='uniform', ax=ax[0], items_names=names, average=5)

    # plotter.precision_over_time(name=method_i[i]['name'])

    # plotter.precision_over_time_plot(name=method_i[i]['name'], ax=ax[int(i / 2), i % 2], items=method_i[i]['pr'],
    #                                  items_names=names)
    # plotter.precision_over_time_plot(name=method_i[i]['name'], ax=ax[int(i / 2), i % 2], items_names=names)

    ax[1].legend(loc='upper right', ncol=2, fontsize=11)
    ax[0].set_xlabel('Years', fontsize=12)
    ax[1].set_xlabel('Years', fontsize=12)
    ax[0].set_ylabel('Predictive Accuracy (%)', fontsize=12)
    # ax[1, 0].set_ylabel('Predictive Accuracy (%)', fontsize=12)
    print('saving...')
    plt.tight_layout()
    # name = 'precisions_over_time'
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


def make_bar_chart(path, llp=False, tokens=False):
    if tokens:
        name = 'model' + ('-llp' if llp else '-precision')+'-token'
        models=[]
        k_list = list(range(1, 11))
        # k_list = [15, 20, 25, 30, 50]
        for i in k_list:
            models.append(['uniform', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
            models.append(['frequency', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
        models += [
            ['uniform', 'wavg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
            ['frequency', 'wavg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
            ['uniform', 'wavg', 'exp_euc_sq', 'adj', 'exemplar'],
            ['frequency', 'wavg', 'exp_euc_sq', 'adj', 'exemplar'],
            ['uniform', 'pt-wavg', 'exp_euc_sq', '1.0', 'prototype'],
            ['frequency', 'pt-wavg', 'exp_euc_sq', '1.0', 'prototype'],
            ['uniform', 'one', '', '1.0','Baseline'],
            ['frequency', 'one', '', '1.0','Baseline']
        ]
    else:
        name = 'model' + ('-llp' if llp else '-precision')
        models=[]
        k_list = list(range(1, 11))
        # k_list = [15, 20, 25, 30, 50]
        for i in k_list:
            models.append(['uniform', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
            models.append(['items', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
        models += [
            ['uniform', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
            ['items', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
            ['uniform', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
            ['items', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
            ['uniform', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],
            ['items', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],
            ['uniform', 'one', '', '1.0', 'Baseline'],
            ['items', 'one', '', '1.0', 'Baseline'],
        ]
    s_words = load_super_words()

    colors = [c[0], c[1], c[2], c[3], c[4]]

    method_names = []
    all_methods = []
    for m in models:
        all_methods.append(m[0] + '-' + m[1] + '-' + m[2] + '_' + str(m[3]))
        method_names.append(m[4])
    method_names = [method_names[2 * i] for i in range(len(method_names) // 2)]
    prs = dict()
    for m in all_methods:
        try:
            with open('./predictions/' + path + '/' + m + '.pkl', 'rb') as p_file:
                prs[m] = pickle.load(p_file)
        except:
            print()

    analyser = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=colors, baselines=None,
                                       all_methods=all_methods, lang=path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    if llp:
        s = analyser.just_get_llp(first_year=THRESHOLD, last_year=END)
        unif = []
        unif_er = []
        size = []
        size_er = []
        for i in range(int(len(s) / 2)):
            unif.append(s[2 * i][1] / 100)
            unif_er.append(s[2 * i][2] / 100)
            size.append(s[2 * i + 1][1] / 100)
            size_er.append(s[2 * i + 1][2] / 100)
        ax[0].set_ylabel('Log likelihood of the posteriors', fontsize=12)
        bot=min(unif+size)
        bot*=1.1
        ax[0].set_ylim(bottom=bot, top=0)
        ax[1].set_ylim(bottom=bot, top=0)
    else:
        s = analyser.just_get_precision(first_year=THRESHOLD, last_year=END)
        unif = []
        unif_er = []
        size = []
        size_er = []
        for i in range(int(len(s) / 2)):
            unif.append(s[2 * i][1] / 100)
            unif_er.append(s[2 * i][2] / 100)
            size.append(s[2 * i + 1][1] / 100)
            size_er.append(s[2 * i + 1][2] / 100)
        # top = max(unif + size)
        # top *= 1.1
        top=0.42
        ax[0].set_ylim(bottom=0, top=top)
        ax[1].set_ylim(bottom=0, top=top)
        # ax[0].set_yticks([i / 100 for i in range(0, 46, 5)])
        # ax[1].set_yticks([i / 100 for i in range(0, 46, 5)])
        ax[0].set_ylabel('Predictive Accuracy (%)', fontsize=12)

    ax[0].bar(method_names, unif, yerr=unif_er, error_kw=dict(ecolor='black', lw=1, capsize=4, capthick=1, alpha=0.5))
    ax[0].set_xticklabels(method_names, rotation=30, ha="right")
    ax[0].set_title('uniform')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    ax[1].bar(method_names, size, yerr=size_er, error_kw=dict(ecolor='black', lw=1, capsize=4, capthick=1, alpha=0.5))
    ax[1].set_xticklabels(method_names, rotation=30, ha="right")
    ax[1].set_title('size-based')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    ax[0].set_xlabel('models', fontsize=12)
    ax[1].set_xlabel('models', fontsize=12)
    print('saving...')
    plt.tight_layout()

    fig.savefig('./predictions/' + name + '.png', bbox_inches='tight')
    # fig.savefig('./predictions/' + name + '.pdf', format='pdf', transparent=True, bbox_inches='tight')
    fig.savefig('./predictions/' + name + '.eps', format='eps', transparent=True, bbox_inches='tight')
    fig.clear()
    plt.clf()


def find_predicted_cat_for_nouns(path, m, nouns):
    method = m[0] + '-' + m[1] + '-' + m[2] + '_' + str(m[3])
    try:
        with open('./predictions/' + path + '/' + method + '.pkl', 'rb') as p_file:
            preds = pickle.load(p_file)
    except:
        return
    # s = """博客 & blog & 2003 & 个 & 个\\ \hline
    # 网民 & netizen & 2001 & 名 & 名\\ \hline
    # 公投 & referendum & 2000 & 次 & 次\\ \hline
    # 帖子 & (Internet) post & 1999 & 篇 & 份\\ \hline
    # 股权 & equity & 1998 & 批 & 笔\\ \hline
    # 世博会 & Expo & 1998 & 次 & 次\\ \hline
    # 并购 & merger & 1998 & 宗 & 次\\ \hline
    # 网友 & (Internet) user & 1998 & 名 & 名\\ \hline
    # 机型 & (aircraft) model & 1997 & 款 & 架\\ \hline
    # 玩家 & player & 1997 & 名 & 名\\ \bottomrule"""
    # nouns = [x.split()[0] for x in s.split('\n')]
    for year, cats in preds.items():
        best_cat = dict()  # dictionary of word: [best category, probability]
        for cat, words in cats.items():
            for word, p in words.items():
                if word not in best_cat:
                    best_cat[word] = [cat, p]
                elif p > best_cat[word][1]:
                    best_cat[word] = [cat, p]
        for word, cat in best_cat.items():
            if word in nouns:
                print(word, year, cat)
    # s = """博客 & blog & 2003 & 个 & 个\\ \hline
    # 网民 & netizen & 2001 & 名 & 名\\ \hline
    # 公投 & referendum & 2000 & 次 & 次\\ \hline
    # 帖子 & (Internet) post & 1999 & 篇 & 名\\ \hline
    # 股权 & equity & 1998 & 批 & 项\\ \hline
    # 世博会 & Expo & 1998 & 次 & --\\ \hline
    # 并购 & merger & 1998 & 宗 & 起\\ \hline
    # 网友 & (Internet) user & 1998 & 名 & 个\\ \hline
    # 机型 & (aircraft) model & 1997 & 款 & 位\\ \hline
    # 玩家 & player & 1997 & 名 & 名\\ \bottomrule"""


def make_precision_recall_plot(path, models):
    s_words = load_super_words()

    all_methods = []
    for m in models:
        all_methods.append(m[0] + '-' + m[1] + '-' + m[2] + '_' + str(m[3]))
    prs = dict()
    for m in all_methods:
        try:
            with open('./predictions/' + path + '/' + m + '.pkl', 'rb') as p_file:
                prs[m] = pickle.load(p_file)
        except:
            print(m)

    analyser = SuperPredictionAnalyser(super_words=s_words, predictions=prs, colors=[], baselines=None,
                                       all_methods=all_methods, lang=path)

    analyser.scatter_plot_category_precision_recall()


def plot_s_values_over_time():
    all_methods = [('uniform-avg-exp_euc_sq_', 'uniform prior'), ('items-avg-exp_euc_sq_','size-based prior')]
    file_name = 's-over-time'
    # all_methods = [('uniform-wavg-exp_euc_sq_', 'uniform prior'), ('frequency-wavg-exp_euc_sq_', 'size-based prior')]
    # file_name = 's-over-time-token'

    # method_i = [
    #     {'lang': 'chi-en-w2v-yby-all',
    #      'name': 'Full'},
    #     {'lang': 'chi-w2v-yby-pca-all',
    #      'name': 'PCA-reduced'},
    #     {'lang': 'chi-w2v-yby-s0.5lda-fixed-all',
    #      'name': 'FDA-reduced (static)'},
    #     {'lang': 'chi-w2v-yby-s0.5lda-all',
    #      'name': 'FDA-reduced (dynamic)'}]
    method_i = [
        {'lang': 'chi-en-w2v-yby-all',
         'name': 'Full'},
        # {'lang': 'chi-en-w2v-yby-pca_resz-all',
        #  'name': 'PCA-reduced'},
        # {'lang': 'chi-en-w2v-yby-s0.5lda-fixed_resz-all',
        #  'name': 'FDA-reduced (static)'},
        # {'lang': 'chi-en-w2v-yby-s0.5lda_resz-all',
        #  'name': 'FDA-reduced (dynamic)'}
    ]

    kws = dict()
    for m in method_i:
        with open('./predictions/' + m['lang'] + '/kernels-prc-10.pkl', 'rb') as k_file:
        # with open('./predictions/' + m['lang'] + '/kernels-opt.pkl', 'rb') as k_file:
            kws[m['name']] = pickle.load(k_file)

    first_year = 1950
    print('plotting...')
    fig, ax = plt.subplots(1, figsize=(12, 6))
    x_list = list(range(first_year, 2004))

    i = 0
    for name, kw in kws.items():
        kw_list_uni = [1 / kw[all_methods[0][0]][year] for year in x_list]
        kw_list_itm = [1 / kw[all_methods[1][0]][year] for year in x_list]
        ax.plot(x_list, kw_list_uni, linewidth=1.5, color=c[i], label=all_methods[0][1], alpha=1, dashes=[5, 2])
        ax.plot(x_list, kw_list_itm, linewidth=1.5, color=c[i], label=all_methods[1][1], alpha=1)
        i += 1

    # ax.set_xticks(x_list)
    # ax.set_yticks([i for i in range(0, 20)])
    # ax.set_ylim(bottom=0, top=10)
    ax.set_ylim(bottom=0, top=11)
    ax.set_xlim(right=1995)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    # ax.yaxis.grid(alpha=0.3, linestyle='solid', linewidth=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax.set_title('Precision over time for ' + name + ' models', fontsize=16)
    # ax.set_title('stimated optimal values of the sensitivity parameter (s) from the type-based models.', fontsize=13)

    ax.legend(loc='upper left', ncol=2, fontsize=11)
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Estimated s', fontsize=12)
    # ax[1, 0].set_ylabel('Predictive Accuracy (%)', fontsize=12)
    print('saving...')
    plt.tight_layout()
    fig.savefig('./predictions/' + file_name + '.png', bbox_inches='tight')
    fig.savefig('./predictions/' + file_name + '.pdf', format='pdf', transparent=True, bbox_inches='tight')
    # fig.savefig('./predictions/' + file_name + '.eps', format='eps', transparent=True, bbox_inches='tight')
    fig.clear()
    plt.clf()


if __name__ == "__main__":
    plot_time_series()
    # find_predicted_cat_for_nouns()
    # plot_s_values_over_time()
    pass
