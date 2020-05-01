import super_words_builder as sw_builder
import model_saver
import dimension_reducer
from paths import *
from analyse import find_best_kernel_widths, do_analysis, make_bar_chart, make_precision_recall_plot
from predict import predict_with_all_kernel_widths, predict_all_models, optimize_kernel_widths

# prepare the data and the word2vec
# model_saver.save_chi_w2vs()

# these three lines would produce the file ./data/super_words-chi-Luis-YbY(w2v).pkl:
# sw_builder.save_classifier_nouns()
# sw_builder.save_time_stamps()
# sw_builder.build_super_words()
#
# model_saver.w2v_remove_non_superword()

# dimension_reducer.reduce_dimensions(method='PCA', resize=resize)
# dimension_reducer.reduce_dimensions(method='LDA', resize=resize)
# dimension_reducer.reduce_dimensions(method='LDA', threshold=THRESHOLD, resize=True)

# models that need kernel width adjustment:
# each model is described by 4 elements:
#   1.prior dist 2.category similarity method 3.vector similarity function 4.kernel width
models = []
k_list = list(range(1, 11))
# k_list = [15, 20, 25, 30, 50]
for i in k_list:
    models.append(['uniform', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
    models.append(['items', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
    models.append(['frequency', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
models += [
    # ['uniform', 'nn', 'exp_euc_sq', '1.0'],
    # ['items', 'nn', 'exp_euc_sq', '1.0'],
    # ['uniform', '5nn', 'exp_euc_sq', '1.0'],
    # ['items', '5nn', 'exp_euc_sq', '1.0'],

    ['uniform', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    ['items', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    ['uniform', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
    ['items', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
    ['uniform', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],
    ['items', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],

    ['uniform', 'wavg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    ['frequency', 'wavg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    ['uniform', 'wavg', 'exp_euc_sq', 'adj', 'exemplar'],
    ['frequency', 'wavg', 'exp_euc_sq', 'adj', 'exemplar'],
    ['uniform', 'pt-wavg', 'exp_euc_sq', '1.0', 'prototype'],
    ['frequency', 'pt-wavg', 'exp_euc_sq', '1.0', 'prototype'],

    ['uniform', 'one', '', '1.0', 'Baseline'],
    ['items', 'one', '', '1.0', 'Baseline'],
    ['uniform', 'one', '', '1.0','Baseline'],
    ['frequency', 'one', '', '1.0','Baseline']
]

kw_models = []
for m in models:
    if m[3] == 'adj':
        kw_models.append(m)

# all the kernel widths that we want to try
kernel_widths = [i / 10.0 for i in range(1, 11)] + [float(i) for i in range(2, 101)]

# for each type of vector space
for v, p in paths.items():
    print(v, p)
    # for all kernel_widths try to predict and then analyse to find the best KW
    # predict_with_all_kernel_widths(path=p, w2v_version=v, kws=kernel_widths, models=kw_models, s=START, t=START + 1,
    #                                e=END, step=STEP)
    # find_best_kernel_widths(path=p, kws=kernel_widths, models=kw_models)
    mode = 'prc'
    # mode = 'prc-10'
    # mode = 'std'
    # # mode = 'sanity'
    optimize_kernel_widths(path=p, w2v_version=v, models=kw_models, kwmin=0.0, kwmax=100.0, s=START, t=THRESHOLD, e=END,
                           step=STEP, mode=mode)

    # having the best kernel widths, predict with all models
    predict_all_models(path=p, w2v_version=v, models=models, s=START, t=THRESHOLD, e=END,step=STEP, mode=mode)
    # print('results for', v, p)
    # do_analysis(path=p, models=models, path_n=path_name[v], llp=True)

#
# print('==' * 20)
# print('==' * 20)
# # for each type of vector space
# for v, p in paths.items():
#     print(v)
#     # analyse the predictions and get the precision values
#     do_analysis(path=p, models=models, path_n=path_name[v], llp=False)
#     # do_analysis(path=p, models=models, path_n=path_name[v], llp=True)

# make_bar_chart('chi-en-w2v-yby-all', llp=False, tokens=False)
# make_bar_chart('chi-en-w2v-yby-all', llp=False, tokens=True)
# make_bar_chart('chi-en-w2v-yby-all', models,llp=True)
# make_precision_recall_plot('chi-en-w2v-yby-all', [['items', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)']])
