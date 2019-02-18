import super_words_builder as sw_builder
import model_saver
import dimension_reducer
from analyse import find_best_kernel_widths, do_analysis
from predict import predict_with_all_kernel_widths, predict_all_models

START = 1940
THRESHOLD = 1950
END = 2010
# prepare the data and the word2vec
model_saver.save_chi_w2vs()

sw_builder.save_classifier_nouns()
sw_builder.save_time_stamps()
sw_builder.build_super_words()

model_saver.w2v_remove_non_superword()

dimension_reducer.reduce_dimensions(method='PCA')
dimension_reducer.reduce_dimensions(method='LDA')
dimension_reducer.reduce_dimensions(method='LDA', threshold=THRESHOLD)

# 4 different vector spaces
paths = {'orig': 'chi-w2v-yby-all', 'PCA': 'chi-w2v-yby-pca-all', 's0.5LDA-fixed': 'chi-w2v-yby-s0.5lda-fixed-all',
         's0.5LDA': 'chi-w2v-yby-s0.5lda-all'}

# models that need kernel width adjustment:
# each model is described by 4 elements:
#   1.prior dist 2.category similarity method 3.vector similarity function 4.kernel width
models = [['uniform', 'nn', 'exp_euc_sq', '1.0'],
          ['uniform', '5nn', 'exp_euc_sq', '1.0'],
          ['uniform', 'avg', 'exp_euc_sq', '1.0'],
          ['uniform', 'avg', 'exp_euc_sq', 'adj'],
          ['uniform', 'pt-avg', 'exp_euc_sq', '1.0'],

          ['items', 'nn', 'exp_euc_sq', '1.0'],
          ['items', '5nn', 'exp_euc_sq', '1.0'],
          ['items', 'avg', 'exp_euc_sq', '1.0'],
          ['items', 'avg', 'exp_euc_sq', 'adj'],
          ['items', 'pt-avg', 'exp_euc_sq', '1.0'],

          ['uniform', 'wavg', 'exp_euc_sq', '1.0'],
          ['uniform', 'wavg', 'exp_euc_sq', 'adj'],
          ['uniform', 'pt-wavg', 'exp_euc_sq', '1.0'],

          ['frequency', 'wavg', 'exp_euc_sq', '1.0'],
          ['frequency', 'wavg', 'exp_euc_sq', 'adj'],
          ['frequency', 'pt-wavg', 'exp_euc_sq', '1.0'],

          ['items', 'one', '', '1.0'],
          ['uniform', 'one', '', '1.0'],
          ['frequency', 'one', '', '1.0']
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
    predict_with_all_kernel_widths(path=p, w2v_version=v, kws=kernel_widths, models=kw_models, s=START, t=START + 1,
                                   e=THRESHOLD)
    find_best_kernel_widths(path=p, kws=kernel_widths, models=kw_models)

    # having the best kernel widths, predict with all models
    predict_all_models(path=p, w2v_version=v, models=models, s=START, t=THRESHOLD + 1, e=END)

    # analyse the predictions and get the precision values
    do_analysis(path=p, models=models)
