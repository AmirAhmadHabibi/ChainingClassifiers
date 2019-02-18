import super_words_builder as sw_builder
import model_saver
import dimension_reducer
from analyse import find_best_kernel_widths
from predict import predict_with_all_kernel_widths, predict_all_models

# prepare the data and the word2vec
model_saver.save_chi_w2vs()

sw_builder.save_classifier_nouns()
sw_builder.save_time_stamps()
sw_builder.build_super_words()

model_saver.w2v_remove_non_superword()

dimension_reducer.reduce_LDA()
dimension_reducer.reduce_LDA(threshold=1950)

# 4 different vector spaces
paths = {'orig': 'chi-w2v-yby-all', 'PCA': 'chi-w2v-yby-pca-all', 's0.5LDA-F': 'chi-w2v-yby-s0.5lda-fixed-all',
         's0.5LDA': 'chi-w2v-yby-s0.5lda-all'}

# all the widths that we want to try
kernel_widths = [i / 10.0 for i in range(1, 11)] + [float(i) for i in range(2, 101)]

# for each type of vector space
for v, p in paths.items():
    # for all kernel_widths try to predict and then analyse to find the best KW
    predict_with_all_kernel_widths(path=p, w2v_version=v, kws=kernel_widths, s=1940, t=1941, e=1950)
    find_best_kernel_widths(path=p, kws=kernel_widths)

    # having the best kernel widths, predict with all models
    predict_all_models(path=p, w2v_version=v, s=1940, t=1951, e=2010)

