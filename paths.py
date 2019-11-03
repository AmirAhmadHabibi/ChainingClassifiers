# # 4 different vector spaces
# paths = {
#     'orig': 'chi-w2v-yby-all',
#     'PCA': 'chi-w2v-yby-pca-all',
#     's0.5LDA-fixed': 'chi-w2v-yby-s0.5lda-fixed-all',
#     's0.5LDA': 'chi-w2v-yby-s0.5lda-all'
# }
#
# super_words_path = './data/super_words-chi-Luis-YbyY(w2v).pkl'
# w2v_path = './data/w2v-chi-yby.pkl'
#
# pca_reduced_path = './data/w2v-chi-yby-PCA.pkl'
# lda_f_reduced_path = './data/w2v-chi-yby-s0.5LDA-fixed.pkl'
# lda_reduced_path = './data/w2v-chi-yby-s0.5LDA.pkl'
# w2v_init = './data/w2v-chi-yby-'
# resize = False

paths = {
    'orig': 'chi-en-w2v-yby-all',
    'PCA': 'chi-en-w2v-yby-pca-all',
    's0.5LDA-fixed': 'chi-en-w2v-yby-s0.5lda-fixed-all',
    's0.5LDA': 'chi-en-w2v-yby-s0.5lda-all'
}
path_name={
    'orig': 'Full',
    'PCA': 'PCA-reduced',
    's0.5LDA-fixed': 'FDA-reduced',
    's0.5LDA': 'FDA-reduced'}
super_words_path = './data/super_words-chi-Luis-YbyY(w2v-en).pkl'
w2v_path = './data/w2v-chi-en-yby.pkl'
w2v_init = './data/w2v-chi-en-yby-'

pca_reduced_path = './data/w2v-chi-en-yby-PCA.pkl'
lda_f_reduced_path = './data/w2v-chi-en-yby-s0.5LDA-fixed.pkl'
lda_reduced_path = './data/w2v-chi-en-yby-s0.5LDA.pkl'
resize = False

# paths = {
#     'orig': 'chi-en-w2v-yby-all',
#     'PCA_resz': 'chi-en-w2v-yby-pca_resz-all',
#     's0.5LDA-fixed_resz': 'chi-en-w2v-yby-s0.5lda-fixed_resz-all',
#     's0.5LDA_resz': 'chi-en-w2v-yby-s0.5lda_resz-all'
# }
# super_words_path = './data/super_words-chi-Luis-YbyY(w2v-en).pkl'
# w2v_path = './data/w2v-chi-en-yby.pkl'
# w2v_init = './data/w2v-chi-en-yby-'
#
# pca_reduced_path = './data/w2v-chi-en-yby-PCA_resz.pkl'
# lda_f_reduced_path = './data/w2v-chi-en-yby-s0.5LDA-fixed_resz.pkl'
# lda_reduced_path = './data/w2v-chi-en-yby-s0.5LDA_resz.pkl'
# resize = True


START = 1940
THRESHOLD = 1950
END = 2010