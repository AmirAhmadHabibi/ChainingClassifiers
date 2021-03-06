import copy
import pickle
import random
from time import time
from gensim import matutils
from math import exp, log, inf, sqrt
import numpy as np
from sklearn.cluster import estimate_bandwidth, MeanShift

from utilitarian import Progresser
from scipy.stats import gaussian_kde
from sklearn.neighbors.kde import KernelDensity


class SuperPredictor:
    """Super Predictor

    Parameters
    ----------
    super_words : dict
        super_words[year][cat][context] is the frequency of that category-context pair in that year.
        Only the new context words should be in this dictionary.

    path : string
        The path to save the pickle file of predictions.

    start : int
        The year to start considering the words.

    threshold: int
        The year to start predicting the words.

    end : int
        The ending year.

    word_vectors : dict
        dictionary of words to their vector representations

    word_vector_yby: dict
        dictionary of years to word vectors in case of LDA and PCA dimension reduction in each year.

    swc : dict
        super-words-complete. a version of the super-words that all the context words and their frequencies are present,
        not just the new ones.

    new_words : dict
        A dictionary of years and the newly appearing words in that year.
    """

    def __init__(self, super_words, path, start, threshold, end, step, hbc, word_vectors, word_vector_yby, swc=None):
        self.word_vectors = word_vectors
        self.super_words = super_words
        self.hbc = hbc
        self.path = path
        self.kernel_width = None
        self.categories = super_words[threshold].keys()
        self.threshold = threshold
        self.start = start
        self.end = end
        self.step = step
        self.new_words = None
        self.predictions = None
        self.prior_dist = None
        self.cat_sim_method = None
        self.vec_sim_method = None
        self.super_words_so_far = None  # super words until current year for each category
        self.word_freq_so_far = None
        self.prg = None
        self.word_vectors_yby = word_vector_yby
        self.kde_functions = dict()
        self.super_words_c = swc
        self.prototypes = None

        # initiate frequency of each category in each year
        self.frequency = dict()
        for yr in range(start, end, step):
            self.frequency[yr] = dict()
            for cat in self.categories:
                self.frequency[yr][cat] = sum(self.super_words_c[yr][cat].values())

        # initialise new word list for each year
        self.new_words = dict()
        for yr in range(start, end, step):
            self.new_words[yr] = set()
            for cat, wrds in self.super_words[yr].items():
                self.new_words[yr] = self.new_words[yr] | wrds.keys()

    def predict_them_all(self, prior_dist, cat_sim_method, vec_sim_method, kw=1.0):
        """
        Predicts for new words of all years from the threshold until the end.

        Parameters
        ----------
        prior_dist : string
            valid prior distributions are
            ['uniform'|'frequency'|'items']

        cat_sim_method : string
            valid category similarity methods are
            ['one'|'gkde'|'nn'|'avg'|'wavg'|'eavg'|
            'tnn'|'tnn-avg'|'*nn'|'*tnn'|
            'pt0-avg'|'pt-avg'|'pt0-mode'|'pt-mode']

        vec_sim_method: string
            valid vector similarity methods are
            ['cos'|'euc'|'exp_euc'|'exp_euc_sq'|'hbc']

        kw: float or dict, (default 1.0)
            kernel width for exp_euc and exp_euc_sq similarity methods
        """

        t = time()
        if type(kw) == float or type(kw) == int or type(kw) == str:
            kw = float(kw)
            self.kernel_width = dict()
            for year in range(self.start, self.end, self.step):
                self.kernel_width[year] = kw
            kw_name = str(kw)
        else:
            self.kernel_width = kw
            kw_name = 'adj'
        self.prior_dist = prior_dist
        self.cat_sim_method = cat_sim_method
        self.vec_sim_method = vec_sim_method

        # initialise super words until current year for each category
        if 'tnn' not in cat_sim_method:
            self.__init_super_words_so_far()

        # iterate on years and predict for each one
        self.prg = Progresser(sum([len(self.new_words[yr]) for yr in range(self.threshold, self.end, self.step)]),
                              msg='predicting for ' + prior_dist + '-' + cat_sim_method + '-' +
                                  vec_sim_method + '_' + kw_name)

        # do the predictions for each year
        self.predictions = dict()
        for year in range(self.threshold, self.end, self.step):
            self.__predict_one_step(year)

        # write the results to file
        with open('./predictions/' + self.path + '/' + prior_dist + '-' + cat_sim_method + '-' + vec_sim_method +
                  '_' + kw_name + '.pkl', 'wb') as prediction_file:
            pickle.dump(self.predictions, prediction_file)

        load_time = time() - t
        print('\t prediction for ' + prior_dist + '-' + cat_sim_method + '-' + vec_sim_method + '_' + kw_name +
              ' done in', int(round(load_time / 60, 0)), ':', int(round(load_time % 60, 0)), 'sec')

    def __init_super_words_so_far(self):
        self.super_words_so_far = dict()
        for cat in self.categories:
            self.super_words_so_far[cat] = set()
        for yr in range(self.start, self.threshold, self.step):
            for cat, wrds in self.super_words[yr].items():
                self.super_words_so_far[cat] = self.super_words_so_far[cat] | wrds.keys()

        if 'pt-wavg' in self.cat_sim_method:
            self.word_freq_so_far = dict()
            for cat in self.categories:
                self.word_freq_so_far[cat] = dict()
            for yr in range(self.start, self.threshold, self.step):
                for cat, wrds in self.super_words[yr].items():
                    for w, f in wrds.items():
                        try:
                            self.word_freq_so_far[cat][w] += f
                        except:
                            self.word_freq_so_far[cat][w] = f

    def __update_prototypes(self, yr):
        self.prototypes = dict()
        for cat in self.categories:
            X = []
            if 'wavg' in self.cat_sim_method:
                ws = []
            for word in self.super_words_so_far[cat]:
                X.append(self.word_vectors[word])
                try:
                    ws.append(self.super_words[yr][cat][word])
                except KeyError:
                    ws.append(0)
                except:
                    continue

            pt_vec = np.nan
            if len(X) == 0:
                pass
            elif 'wavg' in self.cat_sim_method:
                if sum(ws) == 0:
                    pt_vec = np.mean(X, axis=0)
                else:
                    pt_vec = np.average(X, axis=0, weights=ws)
            elif 'avg' in self.cat_sim_method:
                pt_vec = np.mean(X, axis=0)
            elif 'mode' in self.cat_sim_method:
                try:
                    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10000)
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    ms.fit(X)
                    labels_unique, l_counts = np.unique(ms.labels_, return_counts=True)
                    best_label = labels_unique[np.where(l_counts == max(l_counts))]
                    pt_vec = ms.cluster_centers_[best_label]
                except:
                    # if there are any errors just use mean instead
                    pt_vec = np.mean(X, axis=0)

            # print(pt_vec)
            self.prototypes[cat] = pt_vec

    def __predict_one_step(self, year):
        """
        update the super_words_so_far with new words if needed, prepare the word vector of this year if it is year by
        year and calculate the prior for each category and the similarities for each word.

        Parameters
        ----------
        year: (int)
            predict for this year
        """
        # prepare w2v list for this year if the vectors are Year by Year
        if self.word_vectors_yby is not None:
            self.word_vectors = self.word_vectors_yby[year]

        # update super words until now if the method is not based on only the immediate previous year
        if 'tnn' not in self.cat_sim_method:
            for cat, wrds in self.super_words[year - self.step].items():
                self.super_words_so_far[cat] = self.super_words_so_far[cat] | wrds.keys()
            if 'pt-wavg' in self.cat_sim_method:
                for cat, wrds in self.super_words[year - self.step].items():
                    for w, f in wrds.items():
                        try:
                            self.word_freq_so_far[cat][w] += f
                        except:
                            self.word_freq_so_far[cat][w] = f

        # initialise prototypes for each category in this year
        if 'pt' in self.cat_sim_method:
            if '0' not in self.cat_sim_method:
                self.__update_prototypes(yr=year - self.step)
            elif self.prototypes is None:
                self.__update_prototypes(yr=year - self.step)

        # prepare kde functions
        if self.cat_sim_method == 'gkde':
            ne = len(self.new_words[year])
            ol = sum(len(ws) for ws in self.super_words_so_far.values())
            print(year, '\t', '%' + str(int(100 * ne / (ol + ne))), ne, '\t', '%' + str(int(100 * ol / (ol + ne))), ol)
            for cat in self.categories:

                vectors = [self.word_vectors[w] for w in self.super_words_so_far[cat]]
                if len(vectors) > 1:
                    self.kde_functions[cat] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(vectors)
                else:
                    self.kde_functions[cat] = None

        # p_c = self.__calculate_prior(year=year - self.step,log_res=True)
        p_c = self.__calculate_prior(year=year - self.step, log_res=False)

        p_c_i = dict()  # P(category | item) ~ P(category) * P(item | category)
        # initialise the 2D dict
        for category in self.categories:
            p_c_i[category] = dict()

        for item in self.new_words[year]:
            self.prg.count()
            # standard knn methods:
            if self.cat_sim_method.startswith('s') and self.cat_sim_method.endswith('nn'):
                k = int(self.cat_sim_method[1:-2])
                max_sim = [(0, '') for _ in range(k)]
                for cat in self.categories:
                    for word in self.super_words_so_far[cat]:
                        try:
                            sim = self.__vector_similarity(word, item, year=year)
                            if sim > max_sim[0][0]:
                                max_sim[0] = (sim, cat)
                            max_sim = sorted(max_sim, key=lambda t: t[0])
                        except Exception as e:
                            print(10, e)
                nn_count = {cat: 0 for cat in self.categories}
                max_n = 0
                for x in max_sim:
                    nn_count[x[1]] += 1
                    max_n = max(max_n, nn_count[x[1]])
                best_cats = [cat for cat in self.categories if nn_count[cat] == max_n]
                if self.prior_dist == 'uniform':
                    best_cat = random.choice(best_cats)
                elif self.prior_dist == 'items' or self.prior_dist == 'frequency':
                    best_cats = {bc: p_c[bc] for bc in best_cats}
                    best_cat = sorted(best_cats.items(), key=lambda kv: kv[1])[-1][0]

                for category in self.categories:
                    p_c_i[category][item] = 0
                p_c_i[best_cat][item] = 1
                # TODO: check this
            else:  # other methods
                # for category in self.categories:
                #     # don't predict for categories that had it before
                #     if item in self.super_words_so_far[category]:
                #         p_c_i[category][item] = - inf
                #     else:
                #         # the values are in logarithm so the multiplication would be the sum of the two values
                #         p_c_i[category][item] = p_c[category] + self.__get_similarity(item, year - self.step, category)
                # TODO: the next lines or the previous lines
                sum_over_categories = 0.0
                for category in self.categories:
                    if len(self.super_words_so_far[category]) == 0:
                        p_c_i[category][item] = 0
                    # don't predict for categories that had it before
                    if item in self.super_words_so_far[category]:
                        p_c_i[category][item] = 0
                    else:
                        # the values are in logarithm so the multiplication would be the sum of the two values
                        p_c_i[category][item] = p_c[category] * self.__get_similarity(item, year - self.step, category,
                                                                                      False)
                        sum_over_categories += p_c_i[category][item]

                for category in self.categories:
                    if len(self.super_words_so_far[category]) == 0 or item in self.super_words_so_far[category]:
                        p_c_i[category][item] = -inf
                    else:
                        # p_c_i[category][item] = log(p_c_i[category][item]) - log(sum_over_categories)

                        try:
                            p_c_i[category][item] = log(p_c_i[category][item]) - log(sum_over_categories)
                        except:
                            p_c_i[category][item] = -inf
                            # a=self.__get_similarity(item, year - self.step, category, False)
                            # a=self.__get_similarity(item, year - self.step, category, False)
                            # p_c_i[item][category] = -1000000000.0

        self.predictions[year] = p_c_i

    def set_params(self, prior_dist, cat_sim_method, vec_sim_method):
        self.prior_dist = prior_dist
        self.cat_sim_method = cat_sim_method
        self.vec_sim_method = vec_sim_method

    def build_super_words_until(self):
        self.super_words_until = {self.threshold - 2 * self.step: {}}
        for cat in self.categories:
            self.super_words_until[self.threshold - 2 * self.step][cat] = set()
        for yr in range(self.start, self.threshold - self.step):
            for cat, wrds in self.super_words[yr].items():
                self.super_words_until[self.threshold - 2 * self.step][cat] = \
                    self.super_words_until[self.threshold - 2 * self.step][cat] | wrds.keys()
        for yr in range(self.threshold - self.step, self.end, self.step):
            self.super_words_until[yr] = copy.deepcopy(self.super_words_until[yr - self.step])
            for cat, wrds in self.super_words[yr].items():
                self.super_words_until[yr][cat] = self.super_words_until[yr][cat] | wrds.keys()

    def log_likelihood_of_posterior(self, year, kw):
        """
        to find the best kernel width ...
        """
        kw = float(kw)
        self.kernel_width = dict()
        for y in range(self.start, self.end, self.step):
            self.kernel_width[y] = kw
        # self.prg = Progresser(len(self.new_words[year]),
        #                       msg='llp for ' + self.prior_dist + '-' + self.cat_sim_method + '-' + self.vec_sim_method
        #                           + '_' + str(self.kernel_width[year]) + 'y:' + str(year))

        # prepare w2v list for this year if the vectors are Year by Year
        if self.word_vectors_yby is not None:
            self.word_vectors = self.word_vectors_yby[year]

        # update super words until now if the method is not based on only the immediate previous year
        try:
            self.super_words_so_far = self.super_words_until[year - self.step]
        except:
            self.build_super_words_until()
            self.super_words_so_far = self.super_words_until[year - self.step]

        p_c = self.__calculate_prior(year=year - self.step, log_res=False)

        p_c_i = dict()  # P(category | item) ~ P(category) * P(item | category)
        posterior_list = []

        # For each new noun
        for item in self.new_words[year]:
            p_c_i[item] = dict()
            true_cats = []
            sum_over_categories = 0.0
            for category in self.categories:
                if len(self.super_words_so_far[category]) == 0:
                    continue
                # don't count categories that had it before
                # if item in self.super_words_so_far[category]:
                #     p_c_i[item][category] = 0
                # else:
                # the values are in logarithm so the multiplication would be the sum of the two values
                p_c_i[item][category] = p_c[category] * self.__get_similarity(item, year - self.step, category, False)
                sum_over_categories += p_c_i[item][category]
                if item in self.super_words[year][category]:
                    true_cats.append(p_c_i[item][category])

            for x in true_cats:
                try:
                    posterior_list.append(log(x) - log(sum_over_categories))
                except:
                    print('EXCEPTION IN LOG POSTERIOR!')

            # self.prg.count()

        # log_of_posteriors = sum(posterior_list)
        # print('length of the list:', len(posterior_list))
        log_of_posteriors = np.mean(posterior_list)

        return log_of_posteriors

    def __calculate_prior(self, year=None, log_res=True):
        """
        Parameters
        ----------
        year : (int)
            For the case of frequency based prior.

        Returns
        -------
        p_c : (dict)
            A dictionary of the categories and the logarithm of their prior.

        """
        p_c = dict()  # prior: P(category)

        # first methods - probability of each category with uniform distribution
        if self.prior_dist == 'uniform':
            for category in self.categories:
                p_c[category] = 1

        # second method - probability of each category based on frequency
        elif self.prior_dist == 'frequency':
            for category in self.categories:
                p_c[category] = 0
                for yr in range(self.start, year + self.step, self.step):
                    p_c[category] += sum(self.super_words[yr][category].values())

        elif self.prior_dist == 'frequency_sqr':
            for category in self.categories:
                p_c[category] = 0
                for yr in range(self.start, year + self.step, self.step):
                    p_c[category] += sum([sqrt(a) for a in self.super_words[yr][category].values()])

        elif self.prior_dist == 'frequency_log':
            for category in self.categories:
                p_c[category] = 0
                for yr in range(self.start, year + self.step, self.step):
                    p_c[category] += sum([log(a) for a in self.super_words[yr][category].values()])

        # third method - based on longer list of items
        elif self.prior_dist == 'items':
            for category in self.categories:
                p_c[category] = len(self.super_words_so_far[category])
        else:
            raise Exception('Prior calculation method not defined!')

        if log_res:
            # convert to log(p_c)
            for key in p_c.keys():
                if p_c[key] == 0:
                    p_c[key] = - inf
                else:
                    p_c[key] = log(p_c[key])
        return p_c

    def __get_similarity(self, item, year, category, log_res=True):
        """
        Parameters
        ----------
        item : (string)
            The word which its similarity is needed.

        category : (string)
            The category which the item is going to be compared for getting the similarity.

        year : (int)
            For the case of frequency based prior and kernel widths.

        Returns
        -------
        similarity : (float)
            The logarithm of the similarity between the word and the category.

        """
        similarity = 0
        if self.cat_sim_method == 'one':
            similarity = 1
        elif self.cat_sim_method == 'gkde':
            if self.kde_functions[category] is None:
                a = 0
            else:
                # a = self.kde_functions[category].evaluate(self.word_vectors[item])[0]
                a = self.kde_functions[category].score_samples([self.word_vectors[item]])[0]
            # print('--', a)
            # if a == -inf or a == np.nan or a == inf:
            #     a = 0
            return a
        elif self.cat_sim_method == 'nn':
            # nearest neighbour method
            max_similarity = 0
            for word in self.super_words_so_far[category]:
                try:
                    sim = self.__vector_similarity(word, item, year=year)
                    max_similarity = max(sim, max_similarity)
                except Exception as e:
                    print('1', e)
            similarity = max_similarity

        elif self.cat_sim_method == 'avg':
            # average similarity method
            similarity_sum = 0
            words_count = 0
            for word in self.super_words_so_far[category]:
                try:
                    similarity_sum += self.__vector_similarity(word, item, year=year)
                    words_count += 1
                except Exception as e:
                    print('2', e)
            similarity = similarity_sum / words_count if words_count != 0 else 0

        elif self.cat_sim_method == 'wavg':
            # weighted average similarity method
            similarity_sum = 0
            for word in self.super_words_c[year][category].keys():
                try:
                    similarity_sum += self.__vector_similarity(word, item, year=year) * \
                                      self.super_words_c[year][category][word]
                except Exception as e:
                    print('3', e)
            similarity = similarity_sum / self.frequency[year][category] if self.frequency[year][
                                                                                category] != 0 else 0
        elif self.cat_sim_method == 'wavg_sqr':
            # weighted average similarity method
            similarity_sum = 0
            for word in self.super_words_c[year][category].keys():
                try:
                    similarity_sum += self.__vector_similarity(word, item, year=year) * \
                                      sqrt(self.super_words_c[year][category][word])
                except Exception as e:
                    print('4', e)
            total_freq = sum([sqrt(a) for a in self.super_words[year][category].values()])
            similarity = similarity_sum / total_freq if total_freq != 0 else 0

        elif self.cat_sim_method == 'wavg_log':
            # weighted average similarity method
            similarity_sum = 0
            for word in self.super_words_c[year][category].keys():
                try:
                    similarity_sum += self.__vector_similarity(word, item, year=year) * \
                                      log(self.super_words_c[year][category][word])
                except Exception as e:
                    print('5', e)
            total_freq = sum([log(a) for a in self.super_words[year][category].values()])
            similarity = similarity_sum / total_freq if total_freq != 0 else 0

        elif self.cat_sim_method == 'eavg':
            # average similarity method except the 0 frequencies (not use hitherto)
            similarity_sum = 0
            words_count = 0
            for word in self.super_words_c[year][category].keys():
                try:
                    similarity_sum += self.__vector_similarity(word, item, year=year)
                    words_count += 1
                except Exception as e:
                    print('6', e)
            similarity = similarity_sum / words_count if words_count != 0 else 0

        elif self.cat_sim_method == 'tnn':
            # nearest neighbour method for the last decade
            max_similarity = 0
            for word in self.super_words[year][category]:
                try:
                    if word == item:
                        continue
                    sim = self.__vector_similarity(word, item, year=year)
                    max_similarity = max(sim, max_similarity)
                except Exception as e:
                    print('7', e)
            similarity = max_similarity

        elif self.cat_sim_method == 'tnn_avg':
            # average similarity method for the last decade
            similarity_sum = 0
            words_count = 0
            for word in self.super_words[year][category]:
                try:
                    if word == item:
                        continue
                    similarity_sum += self.__vector_similarity(word, item, year=year)
                    words_count += 1
                except Exception as e:
                    print('8', e)
            similarity = similarity_sum / words_count if words_count != 0 else 0

        elif 'tnn' in self.cat_sim_method:
            # avg of K nearest neighbours in each category in the last time point
            max_sim = [0 for _ in range(int(self.cat_sim_method[:-3]))]
            for word in self.super_words[year][category]:
                try:
                    if word == item:
                        continue
                    sim = self.__vector_similarity(word, item, year=year)

                    max_sim[0] = max(sim, max_sim[0])
                    max_sim = sorted(max_sim)
                except Exception as e:
                    print('9', e)
            similarity = np.mean(max_sim)

        elif 'nn' in self.cat_sim_method:
            # avg of K nearest neighbours in each category
            max_sim = [0 for _ in range(int(self.cat_sim_method[:-2]))]
            for word in self.super_words_so_far[category]:
                try:
                    sim = self.__vector_similarity(word, item, year=year)

                    max_sim[0] = max(sim, max_sim[0])
                    max_sim = sorted(max_sim)
                except Exception as e:
                    print(10, e)
            similarity = np.mean(max_sim)
        elif 'pt' in self.cat_sim_method:
            similarity = self.__vector_similarity(w1=item, v2=self.prototypes[category], year=year)
        else:
            raise Exception('Category similarity method not defined!')

        if log_res:
            if similarity == 0:
                return - inf
            return log(similarity) - log(self.kernel_width[year + self.step])  # TODO: I changed this
        else:
            return similarity / self.kernel_width[year + self.step]  # TODO: I changed this

    def __vector_similarity(self, w1=None, w2=None, v1=None, v2=None, year=None):
        """
        Parameters
        ----------
        w1 : (string)
            The first word to get the similarity.

        w2 : (string)
            The Second word to get the similarity.

        year : (int)
            The year parameter for kernel width value

        Returns
        -------
        similarity : (float)
            The similarity between the vectors of the two words.

        """
        if w1 is not None:
            vec1 = self.word_vectors[w1]
        elif v1 is not None:
            vec1 = v1
        if w2 is not None:
            vec2 = self.word_vectors[w2]
        elif v2 is not None:
            vec2 = v2

        if vec2 is np.nan:
            vec2 = np.zeros(len(vec1))

        if self.vec_sim_method == 'cos':
            return (np.dot(matutils.unitvec(vec1), matutils.unitvec(vec2)) + 1) / 2
        elif self.vec_sim_method == 'euc':
            return 1 / np.sqrt(np.sum((vec1 - vec2) ** 2))
        elif self.vec_sim_method == 'exp_euc':
            return exp(
                - np.sqrt(np.sum((vec1 - vec2) ** 2)) / self.kernel_width[year + self.step])
        elif self.vec_sim_method == 'exp_euc_sq':
            return exp(- np.sum((vec1 - vec2) ** 2) / self.kernel_width[year + self.step])
        elif self.vec_sim_method == 'hbc':
            try:
                return self.hbc[w1][w2]
            except IndexError:
                return 0.1
        else:
            raise Exception('Vector similarity method not defined!')

    def find_neighbouring_words(self, word, category, year, starting_year=1988):
        print(word, category, year)

        self.vec_sim_method = 'exp_euc_sq'
        self.kernel_width = dict()
        for year in range(self.threshold, self.end, self.step):
            self.kernel_width[year] = 1.0
        kw_name = str(1.0)
        self.word_vectors = self.word_vectors_yby[year]

        # find nearest neighbour in the specified category
        distances = dict()
        for yr in range(starting_year, year, self.step):
            for cat, wrds in self.super_words[yr].items():
                if cat == category:
                    for w in wrds.keys():
                        if w == word:
                            continue
                        distances[w] = self.__vector_similarity(word, w)
        sorted_words = sorted(distances, key=distances.get, reverse=True)
        print('--', sorted_words[0], distances[sorted_words[0]], '\t', category)
        # words = []
        # for i in range(5):
        #     try:
        #         words.append(sorted_words[i])
        #         print('+', sorted_words[i], distances[sorted_words[i]])
        #     except:
        #         pass
        # words.append('...')
        # for i in range(3):
        #     ind = len(sorted_words) - i - 1
        #     words.append(sorted_words[ind])
        #     print('-', sorted_words[ind], distances[sorted_words[ind]])
        # print(words)

        # find nearest neighbour in all categories
        distances = dict()
        for yr in range(starting_year, year, self.step):
            for cat, wrds in self.super_words[yr].items():
                for w in wrds.keys():
                    if w == word:
                        continue
                    distances[w] = self.__vector_similarity(word, w)
        sorted_words = sorted(distances, key=distances.get, reverse=True)
        for i in range(10):
            # find its real categories
            real_cats = []
            for yr in range(self.start, year, self.step):
                for cat, wrds in self.super_words[yr].items():
                    if sorted_words[i] in wrds:
                        real_cats.append((cat, yr))
            print(i + 1, ')', sorted_words[i], distances[sorted_words[i]], '\t', real_cats)
