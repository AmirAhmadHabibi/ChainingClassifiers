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


def do_analysis(path, models, path_n):
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
    s = analyser.just_get_precision(first_year=THRESHOLD, last_year=END)
    for i in range(int(len(s) / 2)):
        un = str(s[2 * i][1]) + ' \% (' + str(s[2 * i][2]) + ')'
        it = str(s[2 * i + 1][1]) + ' \% (' + str(s[2 * i + 1][2]) + ')'
        print(path_n, models[2 * i][4], '&', un, '&', it, '\\\\', sep='\t')
        # print(s[2 * i], s[2 * i + 1])


def plot_time_series():
    s_words = load_super_words()

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
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
        {'lang': 'chi-en-w2v-yby-pca_resz-all',
         'name': 'PCA-reduced'},
        {'lang': 'chi-en-w2v-yby-s0.5lda-fixed_resz-all',
         'name': 'FDA-reduced (static)'},
        {'lang': 'chi-en-w2v-yby-s0.5lda_resz-all',
         'name': 'FDA-reduced (dynamic)'}]
    for i in range(len(method_i)):

        names = ['Baseline (s)', '1nn', '5nn', 'Exemplar (s=1)', 'Exemplar', 'Prototype']
        all_methods = ['items-one-_1.0', 'items-nn-exp_euc_sq_1.0', 'items-5nn-exp_euc_sq_1.0',
                       'items-avg-exp_euc_sq_1.0', 'items-avg-exp_euc_sq_adj', 'items-pt-avg-exp_euc_sq_1.0']

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

    ax[0, 1].legend(loc='upper right', ncol=2, fontsize=11)
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


if __name__ == "__main__":
    plot_time_series()
