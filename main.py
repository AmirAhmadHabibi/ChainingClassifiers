import super_words_builder as sw_builder
import model_saver
import dimension_reducer
from paths import *
from analyse import find_best_kernel_widths, do_analysis
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
# for i in range(1, 11):
#     models.append(['uniform', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
#     models.append(['items', 's' + str(i) + 'nn', 'exp_euc_sq', '1.0', str(i) + 'nn'])
models += [
    # ['uniform', 'nn', 'exp_euc_sq', '1.0'],
    # ['items', 'nn', 'exp_euc_sq', '1.0'],
    # ['uniform', '5nn', 'exp_euc_sq', '1.0'],
    # ['items', '5nn', 'exp_euc_sq', '1.0'],
    # ['uniform', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    # ['items', 'avg', 'exp_euc_sq', '1.0', 'exemplar (s=1)'],
    ['uniform', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
    ['items', 'avg', 'exp_euc_sq', 'adj', 'exemplar'],
    # ['uniform', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],
    # ['items', 'pt-avg', 'exp_euc_sq', '1.0', 'prototype'],

    # ['uniform', 'wavg', 'exp_euc_sq', '1.0'],
    # ['frequency', 'wavg', 'exp_euc_sq', '1.0'],
    # ['uniform', 'wavg', 'exp_euc_sq', 'adj'],
    # ['frequency', 'wavg', 'exp_euc_sq', 'adj'],
    # ['uniform', 'pt-wavg', 'exp_euc_sq', '1.0'],
    # ['frequency', 'pt-wavg', 'exp_euc_sq', '1.0'],
    #
    # ['uniform', 'one', '', '1.0','Baseline'],
    # ['items', 'one', '', '1.0','Baseline'],
    # ['uniform', 'one', '', '1.0','Baseline'],
    # ['frequency', 'one', '', '1.0','Baseline']
]

kw_models = []
for m in models:
    if m[3] == 'adj':
        kw_models.append(m)

# all the kernel widths that we want to try
kernel_widths = [i / 10.0 for i in range(1, 11)] + [float(i) for i in range(2, 101)]

# for each type of vector space
for v, p in paths.items():
    # for all kernel_widths try to predict and then analyse to find the best KW
    # predict_with_all_kernel_widths(path=p, w2v_version=v, kws=kernel_widths, models=kw_models, s=START, t=START + 1,
    #                                e=END)
    # find_best_kernel_widths(path=p, kws=kernel_widths, models=kw_models)
    optimize_kernel_widths(path=p, w2v_version=v, models=kw_models, kwmin=0.0, kwmax=100.0, s=START, t=THRESHOLD, e=END,
                           std=False)

    # having the best kernel widths, predict with all models
    predict_all_models(path=p, w2v_version=v, models=models, s=START, t=THRESHOLD, e=END, std=False)

# for each type of vector space
for v, p in paths.items():
    print(v)
    # analyse the predictions and get the precision values
    do_analysis(path=p, models=models, path_n=path_name[v])

# decimal =2:
# orig
# Full	exemplar	&	18.0 \% (1.19)	&	36.5 \% (1.49)	\\
# PCA
# PCA-reduced	exemplar	&	23.2 \% (1.31)	&	37.7 \% (1.5)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	20.3 \% (1.24)	&	35.8 \% (1.48)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	19.3 \% (1.22)	&	37.2 \% (1.5)	\\

# decimal =1
# orig
# Full	exemplar	&	15.1 \% (1.11)	&	36.4 \% (1.49)	\\
# PCA
# PCA-reduced	exemplar	&	16.6 \% (1.15)	&	37.6 \% (1.5)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	9.1 \% (0.89)	&	34.3 \% (1.47)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	11.7 \% (1.0)	&	36.5 \% (1.49)	\\

# decimal =1 40-now
# orig
# Full	exemplar	&	18.0 \% (1.19)	&	33.9 \% (1.47)	\\
# PCA
# PCA-reduced	exemplar	&	17.1 \% (1.17)	&	32.2 \% (1.45)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	9.8 \% (0.92)	&	31.4 \% (1.44)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	13.2 \% (1.05)	&	32.6 \% (1.45)	\\

# decimal 2 40-now
# ll	exemplar	&	22.2 \% (1.29)	&	33.8 \% (1.47)	\\
# PCA
# PCA-reduced	exemplar	&	24.4 \% (1.33)	&	32.2 \% (1.45)	\\
# s0.5LDA-fixed
# FDA-reduced	exemplar	&	22.1 \% (1.29)	&	31.8 \% (1.44)	\\
# s0.5LDA
# FDA-reduced	exemplar	&	23.0 \% (1.3)	&	33.1 \% (1.46)	\\