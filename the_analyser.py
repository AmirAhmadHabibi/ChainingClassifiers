import pickle
from copy import copy
from math import sqrt
from random import randint
import pandas as pd
import numpy as np
from scipy import stats

from paths import *
from utilitarian import QuickDataFrame

from matplotlib.font_manager import FontProperties
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from palettable.cartocolors import qualitative
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('agg')
plt.rcdefaults()

c = ['#E58606', '#99C945', '#52BCA3', '#5D69B1', '#CC61B0', '#24796C', '#DAA51B', '#2F8AC4', '#764E9F', '#ED645A']
names = {'items-nn-exp_euc_sq_1.0': '1nn (s)', 'items-5nn-exp_euc_sq_1.0': '5nn (s)',
         'items-avg-exp_euc_sq_1.0': 'Exemplar (s)', 'items-avg-exp_euc_sq_': 'Kernel (s)',
         'items-one-_1.0': 'Baseline (s)',
         'uniform-nn-exp_euc_sq_1.0': '1nn (u)', 'uniform-5nn-exp_euc_sq_1.0': '5nn (u)',
         'uniform-avg-exp_euc_sq_1.0': 'Exemplar (u)', 'uniform-avg-exp_euc_sq_': 'Kernel (u)'}


class SuperPredictionAnalyser:
    """This class takes a list of predictions and does analyses and creates plots """

    def __init__(self, super_words, predictions, colors, baselines, all_methods, lang):
        self.super_words = super_words
        self.predictions = predictions  # dictionary of methods to predictions
        self.colors = colors
        self.all_methods = all_methods
        self.baselines = baselines
        self.lang = lang
        context = set()
        for y in range(START, END):
            for c, words in super_words[y].items():
                for appeared_word in words.keys():
                    context.add(appeared_word)
        self.context_count = len(context)

    def just_get_precision(self, first_year, last_year):
        """print the precision of each model"""
        # print('precisions:')
        output = []
        for method in self.all_methods:
            precision = self.__compute_precision(self.predictions[method], first_year=first_year, last_year=last_year)
            # compute the binomial confidence interval for 95%
            interval = 197 * (sqrt((precision * (1 - precision) / self.context_count)))
            # print(method, ':', round(precision * 100, 1), '\% (' + str(round(interval, 2)) + ')')
            output.append((method, round(precision * 100, 1), round(interval, 2)))
        return output

    def bar_chart_it(self):
        methods = []
        precisions = []

        category_count = 0
        print('precisions:')
        for method in self.all_methods:
            if category_count == 0:
                category_count = len(next(iter(self.predictions[method].values())))
            precision = self.__compute_precision(self.predictions[method])
            methods.append(method)
            precisions.append(precision)
            print(method, ':', precision)

        fig, ax = plt.subplots(figsize=(0.8 * len(methods), 6))
        y_pos = np.arange(len(methods))

        if self.baselines is not None:
            bls = []
            for m in methods:
                bls.append(self.baselines[m])
                # if m.startswith('uniform'):
                #     bls.append(self.baselines[m])
                # else:
                #     bls.append(0)

            ax.bar(y_pos, bls, align='center', alpha=0.3, color='grey')

        rects = ax.bar(y_pos, precisions, align='center', alpha=0.5, color='g')
        for i, c in enumerate(self.colors):
            rects[i].set_color(c)

        ax.set_xticklabels(methods, rotation='vertical')
        ax.set_xticks(y_pos)
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height - 0.02, round(height, 3), ha='center',
                    va='bottom', alpha=0.3)

        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel('precision')
        ax.set_title('Precision of each model')

        fig.savefig('./predictions/' + self.lang + '/stats/bar-chart.png', bbox_inches='tight')
        fig.clear()
        plt.clf()

    def plot_series(self, kws, colour):
        precisions = {'items': [], 'uniform': []}

        for kw in kws:
            m = 'items-avg-exp_euc_sq_' + str(kw)
            p = self.__compute_precision(self.predictions[m], first_year=1940)
            precisions['items'].append(p)
            # print(m, ':', p)
            m = 'uniform-avg-exp_euc_sq_' + str(kw)
            p = self.__compute_precision(self.predictions[m], first_year=1940)
            precisions['uniform'].append(p)
            # print(m, ':', p)

        # find the maximum
        max_p = 0
        max_kw = 0
        for i, p in enumerate(precisions['items']):
            if p > max_p:
                max_p = p
                max_kw = kws[i]
        print('Max for items:', max_kw, max_p)
        max_p = 0
        max_kw = 0
        for i, p in enumerate(precisions['uniform']):
            if p > max_p:
                max_p = p
                max_kw = kws[i]
        print('Max for uniform:', max_kw, max_p)

        # plot it
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(kws, precisions['items'], linewidth=1, color=colour['items'], label='# items')
        ax.plot(kws, precisions['uniform'], linewidth=1, color=colour['uniform'], label='uniform')

        # ax.set_xticklabels(methods, rotation='vertical')
        # ax.set_xticks(y_pos)

        plt.legend(loc='lower right', ncol=1, fontsize=8)

        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel('precision')
        ax.set_title('Precision over Kernel widths')

        fig.savefig('./predictions/' + self.lang + '/stats/plot-kw.png', bbox_inches='tight')
        fig.clear()
        plt.clf()

    @staticmethod
    def compute_precision(super_words, preds, first_year=1951, last_year=2009):
        total_number = 0
        true_positive = 0

        # counting the total number of predicted words
        for year, cats in preds.items():
            if year < first_year or last_year < year:
                continue
            for cat, words in cats.items():
                total_number += len(words)
                break
        if total_number == 0:
            return 0.0
        # counting the number of correct predictions
        for year, cats in preds.items():
            if year > last_year or year < first_year:
                continue
            best_cat = dict()  # dictionary of word: [best category, probability]
            for cat, words in cats.items():
                for word, p in words.items():
                    if word not in best_cat:
                        best_cat[word] = [cat, p]
                    elif p > best_cat[word][1]:
                        best_cat[word] = [cat, p]

            for word, cat_p in best_cat.items():
                if word in super_words[year][cat_p[0]]:  # cat_p[1] != 0 and
                    true_positive += 1
        # print('total number of preds:', total_number)
        return true_positive / total_number

    def __compute_precision(self, preds, first_year=1951, last_year=2009):
        total_number = 0
        true_positive = 0

        # counting the total number of predicted words
        for year, cats in preds.items():
            if year < first_year or last_year < year:
                continue
            for cat, words in cats.items():
                total_number += len(words)
                break
        if total_number == 0:
            return 0.0
        # counting the number of correct predictions
        for year, cats in preds.items():
            if year > last_year or year < first_year:
                continue
            best_cat = dict()  # dictionary of word: [best category, probability]
            for cat, words in cats.items():
                for word, p in words.items():
                    if word not in best_cat:
                        best_cat[word] = [cat, p]
                    elif p > best_cat[word][1]:
                        best_cat[word] = [cat, p]

            for word, cat_p in best_cat.items():
                if word in self.super_words[year][cat_p[0]]:  # cat_p[1] != 0 and
                    true_positive += 1
        # print('total number of preds:', total_number)
        return true_positive / total_number

    def box_plot_it(self):
        num_of_cats = 0
        # making a list of all words
        all_words = set()
        for method, preds in self.predictions.items():
            for decade, cats in preds.items():
                for cat, words in cats.items():
                    for word, p in words.items():
                        all_words.add(word)
            break

        for method, preds in self.predictions.items():
            if num_of_cats == 0:
                num_of_cats = len(next(iter(preds.values())))

            probs = dict()
            count = dict()  # dict of (tp, all)s
            for cat, words in next(iter(preds.values())).items():
                probs[cat] = []
                count[cat] = [0, 0]

            for decade, cats in self.super_words.items():
                for cat, words in cats.items():
                    for word in words:
                        if word in all_words:
                            count[cat][1] += 1  # count the total number of real items in each category
            for decade, cats in preds.items():
                best_cat = dict()  # dictionary of word: [best category, probability]
                for cat, words in cats.items():
                    for word, p in words.items():
                        if word not in best_cat:
                            best_cat[word] = [cat, p]
                        elif p > best_cat[word][1]:
                            best_cat[word] = [cat, p]
                for word, cat_p in best_cat.items():
                    if word in self.super_words[decade][cat_p[0]]:
                        probs[cat_p[0]].append(cat_p[1])
                        count[cat_p[0]][0] += 1  # count the TP number of items in each category

            # build the plot
            data_to_plot = []
            axes_name = []
            for category, probs in probs.items():
                data_to_plot.append(probs)
                axes_name.append(category + '\n' + str(count[category][0]) + '/' + str(count[category][1]) + '\n(' +
                                 '%.1f' % (100 * round(count[category][0] / count[category][1], 3)) + '%)')

            # Create a figure and axes instances
            fig, ax = plt.subplots(figsize=(0.9 * num_of_cats, 8))
            ax.boxplot(data_to_plot, showfliers=False)

            ax.set_xticklabels(axes_name)
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

            # horizontal line for random
            ax.axhline(y=self.baselines[method], xmin=0, xmax=3, c="red", linestyle='dashed', linewidth=1, alpha=0.5)

            ax.set_axisbelow(True)
            ax.set_title('Predicted probability distributions for each category using ' + method)
            ax.set_ylabel('probability')

            # plotting scattered circles
            for i in range(num_of_cats):
                y = data_to_plot[i]
                x = np.random.normal(1 + i, 0.05, size=len(y))
                plt.plot(x, y, '.', color='lightgrey', alpha=0.3, zorder=-1)

            # Save the figure
            fig.savefig('./predictions/' + self.lang + '/stats/' + method + '.png', bbox_inches='tight')
            fig.clear()
            plt.clf()

    def false_positive_it(self):
        for method, preds in self.predictions.items():
            fp = pd.DataFrame(columns={'item', 'true_category', 'predicted_category'})
            for decade, cats in preds.items():
                best_cat = dict()  # dictionary of word: [best category, probability]
                for cat, words in cats.items():
                    for word, p in words.items():
                        if word not in best_cat:
                            best_cat[word] = [cat, p]
                        elif p > best_cat[word][1]:
                            best_cat[word] = [cat, p]

                # for each word and its best category
                for word, cat_p in best_cat.items():
                    # if its a false positive
                    if word not in self.super_words[decade][cat_p[0]]:
                        t_cat = ''
                        # find the true category
                        for cat, words in self.super_words[decade].items():
                            if word in words:
                                t_cat = cat
                                break
                        # add the case to the list
                        fp = fp.append({'item': word, 'true_category': t_cat, 'predicted_category': cat_p[0]},
                                       ignore_index=True)
            fp.to_csv('./predictions/' + self.lang + '/stats/false_positives-' + method + '.csv', index=False)

    def precision_over_time(self, name, first_year=1951, average=None):
        first_year = 1951
        items = []
        for method in self.all_methods:
            if not method.startswith('items'):
                continue
            print('computing precision for', method)
            prs = []
            if average is None:
                for y in range(first_year, 2004):
                    prs.append(self.__compute_precision(self.predictions[method], first_year=first_year, last_year=y))
            else:
                for y in range(first_year, 2000, average):
                    prs.append(
                        self.__compute_precision(self.predictions[method], first_year=y, last_year=y + average - 1))
                # print('-', prs[-1])
            items.append(prs)

        print(self.lang)
        print(name)
        print(items)
        print('_____')
        return items

    def precision_over_time_plot(self, name, ax=None, first_year=1951, items=None, items_names=None, average=None):
        if items is None:
            items = self.precision_over_time(name, first_year, average)
        print('plotting...')
        save = False
        if ax is None:
            save = True
            fig, ax = plt.subplots(figsize=(12, 8))

        if average is None:
            x = list(range(first_year, 2004))
        else:
            x = list(range(first_year + average - 1, 2004, average))

        for i in range(len(items)):
            if 'Baseline' in items_names[i]:
                ax.plot(x, items[i], linewidth=1.5, color=self.colors[i], label=items_names[i], alpha=1, dashes=[5, 2])
            else:
                ax.plot(x, items[i], linewidth=1.5, color=self.colors[i], label=items_names[i], alpha=1)

        # for i in range(len(unifm)):
        #     ax.plot(x, unifm[i], linewidth=1.5, color=self.colors[i], label=unifm_names[i], alpha=1, dashes=[5, 2])

        # ax.plot(x, word_count, linewidth=1.5, color='#DAA51B', label='word count', alpha=1)

        if average is None:
            ax.set_xticks(list(range(first_year, 2004, 5)))
        else:
            ax.set_xticks(x)
        ax.set_yticks([i / 100 for i in range(20, 61, 5)])
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        # ax.yaxis.grid(alpha=0.3, linestyle='solid', linewidth=1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # ax.set_title('Precision over time for ' + name + ' models', fontsize=16)
        ax.set_title(name, fontsize=13)

        if save:
            ax.set_xlabel('years', fontsize=16)
            ax.set_ylabel('precision', fontsize=16)
            ax.legend(loc='upper right', ncol=2, fontsize=12)
            print('saving...')
            fig.savefig('./predictions/' + self.lang + '/precision_over_time_' + name + '.png', bbox_inches='tight')
            fig.clear()
            plt.clf()

    def find_100_most_recent(self):
        # remove context from next years
        for y in range(1940, 2004):
            for c, words in self.super_words[y].items():
                for appeared_word in words.keys():
                    # remove the word from the next years
                    for next_year in range(y + 1, 2010):
                        self.super_words[next_year][c].pop(appeared_word, None)
        # print('done removing!')

        # find recent words
        new_words = dict()
        for yr in range(1951, 2004):
            new_words[yr] = set()
            for cat, wrds in self.super_words[yr].items():
                new_words[yr] = new_words[yr] | wrds.keys()

        min_yr = 1950
        # finding 100 most recent nouns , their appearing year , actual classifier, predicted classifier
        recent_words = QuickDataFrame(columns=['word', 'year', 'actual classifier', 'predicted classifier'])
        for i in range(30):
            yr = 2003 - i
            min_yr = min(min_yr, yr)
            for w in new_words[yr]:
                cats = ''
                for cat, wrds in self.super_words[yr].items():
                    if w in wrds:
                        if cats != '':
                            cats += ' & ' + cat
                        else:
                            cats = cat
                recent_words.append([w, yr, cats, ''])
            if len(recent_words) > 100:
                break

        recent_words.set_index(recent_words['year'], unique=False)

        preds = self.predictions[self.all_methods[3]]

        # counting the number of correct predictions
        for year, cats in preds.items():
            if year < min_yr or str(year) not in recent_words.index:
                continue
            best_cat = dict()  # dictionary of word: [best category, probability]
            for cat, words in cats.items():
                for word, p in words.items():
                    if word not in best_cat:
                        best_cat[word] = [cat, p]
                    elif p > best_cat[word][1]:
                        best_cat[word] = [cat, p]

            indices = recent_words.index[str(year)]
            for word, cat_p in best_cat.items():
                for i in indices:
                    if recent_words['word'][i] == word:
                        recent_words['predicted classifier'][i] = cat_p[0]
                        break

        recent_words.to_csv('./predictions/' + self.lang + '/most_recent_context_words.csv')

    def bar_plot_category_precision(self, category_number, name):
        # find the frequency of each category
        first_year = 1951
        last_year = 2009
        cat_size = dict()
        for cat in self.super_words[last_year].keys():
            cat_size[cat] = 0
        for year in range(first_year, last_year):
            for cat in cat_size:
                cat_size[cat] += len(self.super_words[year][cat])
        sorted_cats = sorted(cat_size, key=lambda x: cat_size[x], reverse=True)

        # for each method create the bar plot
        for method, preds in self.predictions.items():
            # find the highest prediction probability for each word in each year
            best_cat = dict()
            for year in range(first_year, last_year + 1):
                best_cat[year] = dict()  # dictionary of word: [best category, probability]
                for cat, words in preds[year].items():
                    for word, p in words.items():
                        if word not in best_cat[year]:
                            best_cat[year][word] = [cat, p]
                        elif p > best_cat[year][word][1]:
                            best_cat[year][word] = [cat, p]
            # the number of true predictions
            tp = dict()
            for cat in cat_size:
                tp[cat] = 0
            # the number of all predictions for each category
            tp_fp = dict()
            for cat in cat_size:
                tp_fp[cat] = 0

            for year in range(first_year, last_year + 1):
                for word, cat_p in best_cat[year].items():
                    tp_fp[cat_p[0]] += 1
                    if word in self.super_words[year][cat_p[0]]:
                        tp[cat_p[0]] += 1

            print('actual context: ', sum(cat_size.values()))
            print('number of predictions:', sum(tp_fp.values()))

            # compute precision for each category
            precisions = []
            for cat in sorted_cats:
                if tp_fp[cat] == 0:
                    precisions.append(0)
                else:
                    precisions.append(tp[cat] / tp_fp[cat])

            # plot the precisions
            ChineseFont = FontProperties(fname='C:\Windows\Fonts\SimHei.ttf', size=11)
            fig, ax = plt.subplots(figsize=(50, 8))
            x_pos = list(range(len(precisions[:category_number])))

            ax.bar(x_pos, precisions[:category_number], color=c[3])

            plt.xlim([-1, len(precisions[:category_number]) + 1])
            ax.set_xticklabels(
                [cat + '\n' + str(cat_size[cat]) + '\n' + str(tp_fp[cat]) for cat in sorted_cats[:category_number]],
                fontproperties=ChineseFont)
            ax.set_xticks(x_pos)
            ax.set_yticks([i / 10 for i in range(0, 11)])

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            method_name = method
            if method in names:
                method_name = names[method]
            elif 'items' in method:
                method_name = 'Kernel (s)'
            elif 'uniform' in method:
                method_name = 'Kernel (u)'

            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.set_axisbelow(True)

            ax.set_ylabel('precision')
            ax.set_title('Precision of predictions for the classifiers \n' + name + ' : ' + method_name)
            fig.savefig('./predictions/' + self.lang + '/stats/bar-chart-precision-' + method_name + '.png',
                        bbox_inches='tight')

            fig.savefig('./predictions/' + self.lang + '/stats/bar-chart-precision-' + method_name + '.pdf',
                        format='pdf', transparent=True, bbox_inches='tight')

            # ax.set_ylabel('Predictive accuracy')
            # ax.set_title('Predictive accuracy for the classifiers \n' + name + ' : ' + method_name)
            # fig.savefig('./predictions/' + self.lang + '/stats/bar-chart-predictive_accuracy-' + method_name + '.png', bbox_inches='tight')

            fig.clear()
            plt.clf()

    def scatter_plot_category_precision_recall(self, name):
        # find the frequency of each category
        first_year = 1951
        last_year = 2009
        cat_size = dict()
        for cat in self.super_words[last_year].keys():
            cat_size[cat] = 0
        for year in range(first_year, last_year):
            for cat in cat_size:
                cat_size[cat] += len(self.super_words[year][cat])
        sorted_cats = sorted(cat_size, key=lambda x: cat_size[x], reverse=True)

        # for each method create the bar plot
        for method, preds in self.predictions.items():
            # find the highest prediction probability for each word in each year
            best_cat = dict()
            for year in range(first_year, last_year + 1):
                best_cat[year] = dict()  # dictionary of word: [best category, probability]
                for cat, words in preds[year].items():
                    for word, p in words.items():
                        if word not in best_cat[year]:
                            best_cat[year][word] = [cat, p]
                        elif p > best_cat[year][word][1]:
                            best_cat[year][word] = [cat, p]
            # the number of true predictions
            tp = dict()
            for cat in cat_size:
                tp[cat] = 0
            # the number of all predictions for each category
            tp_fp = dict()
            for cat in cat_size:
                tp_fp[cat] = 0

            for year in range(first_year, last_year + 1):
                for word, cat_p in best_cat[year].items():
                    tp_fp[cat_p[0]] += 1
                    if word in self.super_words[year][cat_p[0]]:
                        tp[cat_p[0]] += 1

            print('actual context: ', sum(cat_size.values()))
            print('number of predictions:', sum(tp_fp.values()))

            # compute precision for each category
            precisions = []
            recall = []
            sizes = []
            for cat in sorted_cats:
                if cat_size[cat] == 0:
                    continue
                else:
                    recall.append(tp[cat] / cat_size[cat])

                if tp_fp[cat] == 0:
                    precisions.append(0)
                else:
                    precisions.append(tp[cat] / tp_fp[cat])
                sizes.append(cat_size[cat])

            print('pearson correlation and p-value for size vs precision:', stats.pearsonr(sizes, precisions))
            print('pearson correlation and p-value for size vs recall:', stats.pearsonr(sizes, recall))
            # plot the precisions
            ChineseFont = FontProperties(fname='C:\Windows\Fonts\SimHei.ttf', size=11)
            fig, ax = plt.subplots(figsize=(6, 6))

            prev_ann = []
            for i in range(len(precisions)):
                s = 5000 * cat_size[sorted_cats[i]] / cat_size[sorted_cats[0]] + 10
                el = ax.scatter(recall[i], precisions[i], marker='.', s=s, alpha=0.7, facecolor=c[3])

                ann_flag = True
                for pa in prev_ann:
                    if abs(recall[i] - pa[0]) + abs(precisions[i] - pa[1]) < 0.2:
                        ann_flag = False
                if ann_flag:
                    ax.annotate(sorted_cats[i], xy=(recall[i], precisions[i]),
                                xytext=(recall[i], precisions[i] - 0.04), ha='center',
                                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.5),
                                fontproperties=ChineseFont)
                    prev_ann.append((recall[i], precisions[i]))

            ax.set_xticks([i / 10 for i in range(0, 11)])
            ax.set_yticks([i / 10 for i in range(0, 11)])

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            method_name = method
            if method in names:
                method_name = names[method]
            elif 'items' in method:
                method_name = 'Kernel (s)'
            elif 'uniform' in method:
                method_name = 'Kernel (u)'

            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            ax.set_axisbelow(True)

            ax.set_ylabel('Precision', fontsize=14)
            ax.set_xlabel('Recall', fontsize=14)
            # ax.set_title('Precision vs Recall of predictions for the classifiers \n' + name + ' : ' + method_name)
            fig.savefig('./predictions/' + self.lang + '/stats/scatter_plot-precision-recall-' + method_name + '.png',
                        bbox_inches='tight')

            fig.savefig('./predictions/' + self.lang + '/stats/scatter_plot-precision-recall-' + method_name + '.pdf',
                        format='pdf', transparent=True, bbox_inches='tight')
            fig.clear()
            plt.clf()

    def best_kernel_width(self, kws, end_year=2010, method='items-avg-exp_euc_sq_'):
        max_p = 0
        best_kw = 1.0
        for kw in kws:
            m = method + str(kw)
            p = self.__compute_precision(self.predictions[m], first_year=1940, last_year=end_year)
            if p > max_p:
                max_p = p
                best_kw = kw

        return best_kw

    @staticmethod
    def barplot3d(data, y_names, x_names, baseline):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca(projection='3d')

        x_len = len(x_names)
        y_len = len(y_names)
        x = np.arange(0, x_len, 1)
        y = np.arange(0, y_len, 1)
        x, y = np.meshgrid(x - 0.25, y - 0.5)
        x = x.flatten()
        y = y.flatten()
        z = np.zeros(x_len * y_len)

        rho = np.array(data).flatten()
        dx = 0.5 * np.ones_like(z)
        dy = dx.copy()
        dz = rho.flatten()

        # xx, yy = np.meshgrid(range(len(x_names)), range(len(y_names)))
        # zz = copy(yy)
        # zz.fill(baseline)

        # ax.plot_surface(xx, yy, zz,alpha=0.5)

        ax.w_xaxis.set_ticks([i for i in range(len(data[0]))])
        ax.w_xaxis.set_ticklabels(x_names)

        ax.w_yaxis.set_ticks([i for i in range(len(data))])
        ax.w_yaxis.set_ticklabels(y_names)

        # ax.set_title('models with the size based prior')
        ax.set_zlabel('Predictive accuracy (%)')
        ax.w_zaxis.set_tick_params(labelsize=12)

        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, len(data) / len(data[0]), 1, 1]))

        nrm = mpl.colors.Normalize(0, 30)
        c_range = (np.array(data) - 15).flatten()
        # colors = cm.viridis(nrm(c_range))
        # colors = cm.winter(nrm(c_range))
        colors = cm.RdYlGn(nrm(c_range))
        ax.bar3d(x, y, z, dx, dy, dz, colors)
        plt.tight_layout()
        # plt.show()
        fig.savefig('./predictions/barplot3D.png', bbox_inches='tight')
        fig.savefig('./predictions/barplot3D.pdf', format='pdf', transparent=True, bbox_inches='tight')
        fig.savefig('./predictions/barplot3D.eps', format='eps', transparent=True, bbox_inches='tight')
        fig.clear()
        plt.clf()

    @staticmethod
    def overlap(preds1, preds2, first_year=1951, last_year=2009):
        total_number = 0
        total_match_number = 0

        # counting the number of correct predictions
        for year in range(first_year, last_year + 1):

            best_cat1 = dict()  # dictionary of word: [best category, probability]
            for cat, words in preds1[year].items():
                for word, p in words.items():
                    if word not in best_cat1:
                        best_cat1[word] = [cat, p]
                    elif p > best_cat1[word][1]:
                        best_cat1[word] = [cat, p]

            best_cat2 = dict()  # dictionary of word: [best category, probability]
            for cat, words in preds2[year].items():
                for word, p in words.items():
                    if word not in best_cat2:
                        best_cat2[word] = [cat, p]
                    elif p > best_cat2[word][1]:
                        best_cat2[word] = [cat, p]

            if len(best_cat1) != len(best_cat2):
                raise Exception('# words in the two predictions don\'t match')
            if len(best_cat1) == 0:
                print('-- year is empty')
                continue
            match_number = 0
            for word in best_cat1.keys():
                if best_cat1[word][0] == best_cat2[word][0]:
                    match_number += 1

            print('\t', year, round(match_number / len(best_cat1), 2), '\t #words', len(best_cat1))

            total_match_number += match_number
            total_number += len(best_cat1)

        print('___________')
        print('-- overall', round(total_match_number / total_number, 2))
