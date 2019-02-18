import pickle
import numpy as np


def save_chi_w2vs():
    path = './data/similarity_models/zh-Chi-w2v/zh.tsv'
    w2vs = dict()
    w2v_size = 0
    context_in_w2v_size = 0
    with open(path, 'r', encoding='utf8') as infile:
        i = 1
        word = ''
        vector = []
        for line in infile:
            # print(line)
            if line[0] != ' ':
                w2v_size += 1
                context_in_w2v_size += 1
                w2vs[word] = np.array(vector)

                w, _, n = line.partition('[')
                word = w.split()[1]
                vector = []
            elif line[-1] == ']':
                n = line[:-1]
            elif line[-2] == ']':
                n = line[:-2]
            else:
                n = line
            for num in n.split():
                vector.append(float(num))
            i += 1

    with open('./data/w2v-chi.pkl', 'wb') as out_file:
        pickle.dump(w2vs, out_file)


def w2v_remove_non_superword():
    # read the w2v file and filter it with superwords
    with open('./data/w2v-chi.pkl', 'rb') as infile:
        w2v = pickle.load(infile)
    # Since the TbyY file is the final filtered list of context words, read it and make a list of all context words
    context_words = set()
    with open('./data/super_words-chi-Luis-YbyY(w2v).pkl', 'rb') as infile:
        super_words = pickle.load(infile)
        for year, cats in super_words.items():
            for cat, words in cats.items():
                context_words = context_words | words.keys()

    print(len(w2v))
    print(len(context_words))
    new_w2v = dict()

    for wrd in context_words:
        new_w2v[wrd] = w2v[wrd]

    with open('./data/w2v-chi-yby.pkl', 'wb') as outfile:
        pickle.dump(new_w2v, outfile)

