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
for i in range(1, 11):
    models.append(['uniform', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
    # models.append(['items', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
    models.append(['frequency', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
models += [
    # ['uniform', 'nn', 'exp_euc_sq', '1.0'],
    # ['items', 'nn', 'exp_euc_sq', '1.0'],
    # ['uniform', '5nn', 'exp_euc_sq', '1.0'],
    # ['items', '5nn', 'exp_euc_sq', '1.0'],

    # ['uniform', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    # ['items', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    # ['uniform', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
    # ['items', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
    # ['uniform', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],
    # ['items', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],

    ['uniform', 'wavg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    ['frequency', 'wavg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    ['uniform', 'wavg', 'exp_euc_sq', 'adj', 'exemplar'],
    ['frequency', 'wavg', 'exp_euc_sq', 'adj', 'exemplar'],
    ['uniform', 'pt-wavg', 'exp_euc_sq', '1.0', 'prototype'],
    ['frequency', 'pt-wavg', 'exp_euc_sq', '1.0', 'prototype'],
    #
    # ['uniform', 'one', '', '1.0','Baseline'],
    # ['items', 'one', '', '1.0','Baseline'],
    ['uniform', 'one', '', '1.0','Baseline'],
    ['frequency', 'one', '', '1.0','Baseline']
]

kw_models = []
for m in models:
    if m[3] == 'adj':
        kw_models.append(m)

# all the kernel widths that we want to try
kernel_widths = [i / 10.0 for i in range(1, 11)] + [float(i) for i in range(2, 101)]

# # for each type of vector space
# for v, p in paths.items():
#     # for all kernel_widths try to predict and then analyse to find the best KW
#     # predict_with_all_kernel_widths(path=p, w2v_version=v, kws=kernel_widths, models=kw_models, s=START, t=START + 1,
#     #                                e=END)
#     # find_best_kernel_widths(path=p, kws=kernel_widths, models=kw_models)
#     optimize_kernel_widths(path=p, w2v_version=v, models=kw_models, kwmin=0.0, kwmax=100.0, s=START, t=THRESHOLD, e=END,
#                            std=True)
#
#     # having the best kernel widths, predict with all models
#     predict_all_models(path=p, w2v_version=v, models=models, s=START, t=THRESHOLD, e=END, std=True)
#     print('results for', v, p)
#     # do_analysis(path=p, models=models, path_n=path_name[v])

# print('=='*20)
# print('=='*20)
# # for each type of vector space
# for v, p in paths.items():
#     print(v)
#     # analyse the predictions and get the precision values
#     do_analysis(path=p, models=models, path_n=path_name[v])

make_bar_chart('chi-en-w2v-yby-all', models)
# make_precision_recall_plot('chi-en-w2v-yby-all', [['items', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)']])

# decimal =2:
# orig
# Full	exemplar	&	18.0 \% (1.19)	&	36.5 \% (1.49)	\\
# PCA
# PCA-reduced	exemplar	&	23.2 \% (1.31)	&	37.7 \% (1.5)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	20.3 \% (1.24)	&	35.8 \% (1.48)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	19.3 \% (1.22)	&	37.2 \% (1.5)	\\


# after fixing default from 0.0 to 1.0
# orig
# Full	exemplar	&	18.3 \% (1.2)	&	37.1 \% (1.5)	\\
# PCA
# PCA-reduced	exemplar	&	23.3 \% (1.31)	&	38.3 \% (1.51)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	20.2 \% (1.24)	&	36.3 \% (1.49)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	19.4 \% (1.22)	&	37.7 \% (1.5)	\\

# LLP minimize This is not right:
# orig
# Full	exemplar	&	2.6 \% (0.49)	&	30.1 \% (1.42)	\\
# PCA
# PCA-reduced	exemplar	&	4.8 \% (0.66)	&	29.8 \% (1.42)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	4.4 \% (0.63)	&	30.8 \% (1.43)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	8.0 \% (0.84)	&	30.3 \% (1.42)	\

# LLP minimize - fixed the negative issue-
# Full	exemplar	&	3.7 \% (0.58)	&	30.9 \% (1.43)	\\
# PCA
# PCA-reduced	exemplar	&	5.3 \% (0.7)	&	31.3 \% (1.44)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	6.8 \% (0.78)	&	33.6 \% (1.46)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	9.9 \% (0.93)	&	34.4 \% (1.47)	\\

# LLP minimize scalar
# orig
# Full	exemplar	&	6.6 \% (0.77)	&	33.9 \% (1.47)	\\
# PCA
# PCA-reduced	exemplar	&	7.2 \% (0.8)	&	34.6 \% (1.47)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	7.3 \% (0.81)	&	33.8 \% (1.47)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	10.2 \% (0.94)	&	34.8 \% (1.48)	\\

# RESIZED VERSOINS: LLP minimize scalar
# orig
# Full	exemplar	&	6.6 \% (0.77)	&	33.9 \% (1.47)	\\
# PCA_resz
# PCA-reduced	exemplar	&	7.5 \% (0.81)	&	34.9 \% (1.48)	\\
# s0.5LDA-fixed_resz
# FDA-reduced	exemplar	&	7.3 \% (0.81)	&	33.8 \% (1.47)	\\
# s0.5LDA_resz
# FDA-reduced	exemplar	&	10.5 \% (0.95)	&	35.0 \% (1.48)	\\


# RESIZED VERSOINS: LLP my optimizer:
#orig
# Full	exemplar	&	14.4 \% (1.09)	&	38.9 \% (1.51)	\\
# PCA_resz
# PCA-reduced	exemplar	&	10.1 \% (0.93)	&	38.6 \% (1.51)	\\
# s0.5LDA-fixed_resz
# FDA-reduced	exemplar	&	6.0 \% (0.74)	&	32.9 \% (1.46)	\\
# s0.5LDA_resz
# FDA-reduced	exemplar	&	9.8 \% (0.92)	&	33.6 \% (1.46)	\\

# TODO Run frequncy
