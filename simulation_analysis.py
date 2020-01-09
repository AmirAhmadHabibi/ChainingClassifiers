# randomly drop 100 points in [0,1] *[0,1] space
# arbitrarily choose two distant nouns as 2 categpries
# run a simulation separitly exemplar vs prototype
# grow each category using these two models
# order of the nouns and theposition and the seeds should be the same for both
# emerged ones black - other s grey
# one at each time is added
# a number of snapshots or a GIF
#
# simulate this multiple times
# compute FisherDiscriminant(means / determinant of the coveriances- check online) fro both categories> should be better for exemplar
# can plot this is over time and averages with a confidence band around them
# try a static prototype for first 10 years as well in
import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np
from palettable.cartocolors import qualitative

from utilitarian import Progresser

# colours = [qualitative.Vivid_10.mpl_colors[1], qualitative.Vivid_10.mpl_colors[0], qualitative.Vivid_10.mpl_colors[2]]
colours = qualitative.Vivid_10.mpl_colors
markers = ['^', 'o']


def fishers_linear_discriminant(a, b):
    a = np.array(a)
    b = np.array(b)
    # print(a.shape)
    if a.shape[1] == 1:
        return np.nan_to_num(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)) /
                             (np.var(a) + np.var(b)))
    else:
        return np.nan_to_num(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)) / (
                np.linalg.det(np.cov(a.T)) + np.linalg.det(np.cov(b.T))))


def vector_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return exp(- np.sum((a - b) ** 2))


def create_random_sample(n=100, d=2):
    points = []
    for i in range(n):
        points.append(tuple([random.random() for _ in range(d)]))
        # points.append((random.randint(0,1000), random.randint(0,1000)))
    a, b = 0, 0
    min_thr = vector_similarity(np.full(d, 0), np.full(d, 0.5))
    while vector_similarity(a, b) > min_thr:
        a = random.choice(points)
        b = random.choice(points)
    points.remove(a)
    points.remove(b)
    return points, a, b


def cat_similarity(point, cat_points, method='avg', pt_vec=None):
    if method == 'avg':
        # average similarity method
        similarity_sum = 0
        words_count = 0
        for word in cat_points:
            try:
                similarity_sum += vector_similarity(word, point)
                words_count += 1
            except Exception as e:
                print('2', e)
        similarity = similarity_sum / words_count if words_count != 0 else 0

    elif method == 'pt':

        similarity = vector_similarity(point, pt_vec)
    elif method == 'spt':
        similarity = vector_similarity(point, pt_vec)

    return similarity


def predict_next_point(point, a_points, b_points, method, pt_vec_a=None, pt_vec_b=None):
    if 'nn' in method:
        k = int(method[0:-2])
        if len(a_points) + len(b_points) <= k:
            k = 1
        # k = min(int(method[0:-2]),len(a_points)+len(b_points))
        max_sim = [(0, '') for _ in range(k)]
        for cat, x_points in {'a': a_points, 'b': b_points}.items():
            for word in x_points:
                try:
                    sim = vector_similarity(word, point)
                    if sim > max_sim[0][0]:
                        max_sim[0] = (sim, cat)
                    max_sim = sorted(max_sim, key=lambda t: t[0])
                except Exception as e:
                    print(10, e)
        nn_count = {cat: 0 for cat in ['a', 'b']}
        max_n = 0
        for x in max_sim:
            nn_count[x[1]] += 1
            max_n = max(max_n, nn_count[x[1]])
        best_cats = [cat for cat in ['a', 'b'] if nn_count[cat] == max_n]

        best_cat = random.choice(best_cats)
        # elif self.prior_dist == 'items' or self.prior_dist == 'frequency':
        #     best_cats = {bc: p_c[bc] for bc in best_cats}
        #     best_cat = sorted(best_cats.items(), key=lambda kv: kv[1])[-1][0]
        if best_cat == 'a':
            a_points.append(point)
        else:
            b_points.append(point)
    else:
        s_a = cat_similarity(point, a_points, method, pt_vec_a)
        s_b = cat_similarity(point, b_points, method, pt_vec_b)
        if s_a > s_b:
            a_points.append(point)
        else:
            b_points.append(point)
    return a_points, b_points


def plot_scatters(ax, a, b, rest, title, a_orig, b_orig):
    legend_scat = []
    legend_name = []
    for w in a:
        if w != a_orig:
            ax.scatter(w[0], w[1], marker=markers[1], label='A', alpha=0.9, facecolor=colours[0])
    dot = ax.scatter([], [], marker=markers[1], label='A', alpha=0.9, facecolor=colours[0])
    legend_scat.append(dot)
    legend_name.append('category A')

    for w in b:
        if w != b_orig:
            ax.scatter(w[0], w[1], marker=markers[1], label='B', alpha=0.9, facecolor=colours[1])
    dot = ax.scatter([], [], marker=markers[1], label='B', alpha=0.9, facecolor=colours[1])
    legend_scat.append(dot)
    legend_name.append('category B')

    for w in rest:
        ax.scatter(w[0], w[1], marker=markers[1], label='yet-to-emerge', alpha=0.3, facecolor='grey')
    dot = ax.scatter([], [], marker=markers[1], label='yet-to-emerge', alpha=0.3, facecolor='grey')
    legend_scat.append(dot)
    legend_name.append('yet-to-emerge')

    ax.scatter(a_orig[0], a_orig[1], marker=markers[0], alpha=0.6, s=110, facecolor=colours[0])
    dot = plt.scatter([], [], marker=markers[0], alpha=0.6, s=110, facecolor=colours[0])
    legend_scat.append(dot)
    legend_name.append('category A first point')

    ax.scatter(b_orig[0], b_orig[1], marker=markers[0], alpha=0.6, s=110, facecolor=colours[1])
    dot = plt.scatter([], [], marker=markers[0], alpha=0.6, s=110, facecolor=colours[1])
    legend_scat.append(dot)
    legend_name.append('category B first point')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title)
    return legend_scat, legend_name


def plot_all():
    points, a, b = create_random_sample()
    print('|', a, b, points)
    rest = set(points)
    points = [None] + points
    a_e, b_e = [a], [b]
    a_p, b_p = [a], [b]
    a_sp, b_sp = [a], [b]
    a_1n, b_1n = [a], [b]
    a_5n, b_5n = [a], [b]
    a_10n, b_10n = [a], [b]

    check_steps = {60, 80, 98}
    g2g = True
    prog = Progresser(len(points))
    for year, wrd in enumerate(points):
        fig, ax = plt.subplots(1, 6, figsize=(30, 5))
        if wrd is not None:
            rest.remove(wrd)
            # exemlplar model
            a_e, b_e = predict_next_point(wrd, a_e, b_e, method='avg')
            fda_e = fishers_linear_discriminant(a_e, b_e)

            # prototype model
            pt_vec_a = np.mean(a_p, axis=0)
            pt_vec_b = np.mean(b_p, axis=0)
            a_p, b_p = predict_next_point(wrd, a_p, b_p, 'pt', pt_vec_a, pt_vec_b)
            fda_p = fishers_linear_discriminant(a_p, b_p)
            # static prototype model
            if year <= 10:
                spt_vec_a = np.mean(a_sp, axis=0)
                spt_vec_b = np.mean(b_sp, axis=0)
            a_sp, b_sp = predict_next_point(wrd, a_sp, b_sp, 'spt', spt_vec_a, spt_vec_b)
            fda_sp = fishers_linear_discriminant(a_sp, b_sp)

            # 1NN model
            a_1n, b_1n = predict_next_point(wrd, a_1n, b_1n, method='1nn')
            fda_1n = fishers_linear_discriminant(a_1n, b_1n)

            # 5NN model
            a_5n, b_5n = predict_next_point(wrd, a_5n, b_5n, method='5nn')
            fda_5n = fishers_linear_discriminant(a_5n, b_5n)

            # 10NN model
            a_10n, b_10n = predict_next_point(wrd, a_10n, b_10n, method='10nn')
            fda_10n = fishers_linear_discriminant(a_10n, b_10n)

            legend_scat, legend_name = plot_scatters(ax[0], a_e, b_e, rest, 'Exemplar Model | Discrim. ' + str(
                round(fda_e, 1)) + ' | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[1], a_p, b_p, rest, 'Prototype Model | Discrim. ' + str(
                round(fda_p, 1)) + ' | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[2], a_sp, b_sp, rest,
                                                     'Static Prototype Model | Discrim. ' + str(
                                                         round(fda_sp, 1)) + ' | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[3], a_1n, b_1n, rest, '1NN Model | Discrim. ' + str(
                round(fda_1n, 1)) + ' | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[4], a_5n, b_5n, rest, '5NN Model | Discrim. ' + str(
                round(fda_5n, 1)) + ' | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[5], a_10n, b_10n, rest, '10NN Model | Discrim. ' + str(
                round(fda_10n, 1)) + ' | Iter. ' + str(year), a, b)

            if year in check_steps:
                scores = [fda_sp, fda_p, fda_1n, fda_5n, fda_10n]
                count = 0
                for s in scores:
                    if fda_e < s:
                        count += 1
                if count > 1:
                    g2g = False

        else:
            legend_scat, legend_name = plot_scatters(ax[0], a_e, b_e, rest, 'Exemplar Model | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[1], a_p, b_p, rest, 'Prototype Model | Iter. ' + str(year), a,
                                                     b)
            legend_scat, legend_name = plot_scatters(ax[2], a_sp, b_sp, rest,
                                                     'Static Prototype Model | Iter. ' + str(year), a, b)

            legend_scat, legend_name = plot_scatters(ax[3], a_1n, b_1n, rest, '1NN Model | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[4], a_5n, b_5n, rest, '5NN Model | Iter. ' + str(year), a, b)
            legend_scat, legend_name = plot_scatters(ax[5], a_10n, b_10n, rest, '10NN Model | Iter. ' + str(year), a, b)

        ax[0].legend(legend_scat, legend_name, scatterpoints=1, loc='lower left', ncol=2, fontsize=8)
        fig.savefig('./predictions/simulation/scatter_exemplar_vs_protot_vs_stprot-' + str(year) + '.png',
                    bbox_inches='tight')
        fig.savefig('./predictions/simulation/scatter_exemplar_vs_protot_vs_stprot-' + str(year) + '.pdf',
                    bbox_inches='tight')
        fig.clear()
        plt.clf()
        prog.count()
    return g2g


def plot_fda(iter=100, time=200, d=2, st=20):
    fda_e = [[] for _ in range(iter)]
    fda_p = [[] for _ in range(iter)]
    fda_sp = [[] for _ in range(iter)]
    fda_1n = [[] for _ in range(iter)]
    fda_5n = [[] for _ in range(iter)]
    fda_10n = [[] for _ in range(iter)]
    for i in range(iter):
        smpl_size = []
        points, a, b = create_random_sample(n=time, d=d)
        rest = set(points)
        a_e, b_e = [a], [b]
        a_p, b_p = [a], [b]
        a_sp, b_sp = [a], [b]
        a_1n, b_1n = [a], [b]
        a_5n, b_5n = [a], [b]
        a_10n, b_10n = [a], [b]
        for year, wrd in enumerate(points):
            rest.remove(wrd)
            smpl_size.append(len(a_e) + len(b_e))

            # exemlplar model
            a_e, b_e = predict_next_point(wrd, a_e, b_e, method='avg')
            fda_e[i].append(fishers_linear_discriminant(a_e, b_e))

            # prototype model
            pt_vec_a = np.mean(a_p, axis=0)
            pt_vec_b = np.mean(b_p, axis=0)
            a_p, b_p = predict_next_point(wrd, a_p, b_p, 'pt', pt_vec_a, pt_vec_b)
            fda_p[i].append(fishers_linear_discriminant(a_p, b_p))

            # static prototype model
            if year <= 10:
                spt_vec_a = np.mean(a_sp, axis=0)
                spt_vec_b = np.mean(b_sp, axis=0)
            a_sp, b_sp = predict_next_point(wrd, a_sp, b_sp, 'spt', spt_vec_a, spt_vec_b)
            fda_sp[i].append(fishers_linear_discriminant(a_sp, b_sp))

            # 1NN model
            a_1n, b_1n = predict_next_point(wrd, a_1n, b_1n, method='1nn')
            fda_1n[i].append(fishers_linear_discriminant(a_1n, b_1n))

            # 5NN model
            a_5n, b_5n = predict_next_point(wrd, a_5n, b_5n, method='5nn')
            fda_5n[i].append(fishers_linear_discriminant(a_5n, b_5n))

            # 10NN model
            a_10n, b_10n = predict_next_point(wrd, a_10n, b_10n, method='10nn')
            fda_10n[i].append(fishers_linear_discriminant(a_10n, b_10n))

    smpl_sz_sqrt = np.power(np.array(smpl_size), np.full(len(smpl_size), 0.5))
    fda_e_ci = np.divide(np.std(fda_e, axis=0), smpl_sz_sqrt)
    fda_e = np.mean(fda_e, axis=0)
    fda_p_ci = np.divide(np.std(fda_p, axis=0), smpl_sz_sqrt)
    fda_p = np.mean(fda_p, axis=0)
    fda_sp_ci = np.divide(np.std(fda_sp, axis=0), smpl_sz_sqrt)
    fda_sp = np.mean(fda_sp, axis=0)
    fda_1n_ci = np.divide(np.std(fda_1n, axis=0), smpl_sz_sqrt)
    fda_1n = np.mean(fda_1n, axis=0)
    fda_5n_ci = np.divide(np.std(fda_5n, axis=0), smpl_sz_sqrt)
    fda_5n = np.mean(fda_5n, axis=0)
    fda_10n_ci = np.divide(np.std(fda_10n, axis=0), smpl_sz_sqrt)
    fda_10n = np.mean(fda_10n, axis=0)
    # print(fda_e)
    # print(fda_p)
    # print(fda_sp)

    fda_e, fda_p, fda_sp, fda_1n, fda_5n, fda_10n = fda_e[st:], fda_p[st:], fda_sp[st:], fda_1n[st:], fda_5n[
                                                                                                      st:], fda_10n[st:]
    fda_e_ci, fda_p_ci, fda_sp_ci, fda_1n_ci, fda_5n_ci, fda_10n_ci = fda_e_ci[st:], fda_p_ci[st:], fda_sp_ci[
                                                                                                    st:], fda_1n_ci[
                                                                                                          st:], fda_5n_ci[
                                                                                                                st:], fda_10n_ci[
                                                                                                                      st:]
    min_fda = min(min(fda_p - fda_p_ci), min(fda_e - fda_e_ci), min(fda_sp - fda_sp_ci), min(fda_1n - fda_1n_ci),
                  min(fda_5n - fda_5n_ci), min(fda_10n - fda_10n_ci))
    max_fda = max(max(fda_p + fda_p_ci), max(fda_e + fda_e_ci), max(fda_sp + fda_sp_ci), max(fda_1n + fda_1n_ci),
                  max(fda_5n + fda_5n_ci), max(fda_10n + fda_10n_ci))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    x = range(st, len(fda_e) + st)

    ax.plot(x, fda_e, label='Exemplar Model', color=colours[0])
    ax.fill_between(x, fda_e + fda_e_ci, fda_e - fda_e_ci, color=colours[0], alpha=.3)

    ax.plot(x, fda_p, label='Prototype Model', color=colours[1])
    ax.fill_between(x, fda_p + fda_p_ci, fda_p - fda_p_ci, color=colours[1], alpha=.3)

    ax.plot(x, fda_sp, label='Static Prototype Model', color=colours[2])
    ax.fill_between(x, fda_sp + fda_sp_ci, fda_sp - fda_sp_ci, color=colours[2], alpha=.3)

    ax.plot(x, fda_1n, label='1NN Model', color=colours[3])
    ax.fill_between(x, fda_1n + fda_1n_ci, fda_1n - fda_1n_ci, color=colours[3], alpha=.3)

    ax.plot(x, fda_5n, label='5NN Model', color=colours[4])
    ax.fill_between(x, fda_5n + fda_5n_ci, fda_5n - fda_5n_ci, color=colours[4], alpha=.3)

    ax.plot(x, fda_10n, label='10NN Model', color=colours[5])
    ax.fill_between(x, fda_10n + fda_10n_ci, fda_10n - fda_10n_ci, color=colours[5], alpha=.3)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Fisher\'s Linear Disciminant over time - ' + str(iter) + ' iterations - ' + str(d) + 'D')
    ax.set_ylim(min_fda, max_fda)

    ax.legend(loc='lower left', ncol=1, fontsize=8)
    fig.savefig(
        './predictions/simulation/fda_exemplar_vs_prototype_vs_stprot-t' + str(time) + '-i' + str(iter) + '-d' + str(
            d) + '.png',
        bbox_inches='tight')
    fig.savefig(
        './predictions/simulation/fda_exemplar_vs_prototype_vs_stprot-t' + str(time) + '-i' + str(iter) + '-d' + str(
            d) + '.pdf',
        bbox_inches='tight')
    fig.clear()
    plt.clf()


if __name__ == "__main__":
    # i = 0
    # while not plot_all():
    #     i += 1
    #     print(i, 'run.')

    # plot_fda(iter=1000, time=200, d=1)
    ds = [1, 2, 3, 4, 5, 10]
    # for d in ds:
    #     plot_fda(iter=2000, time=300, d=d, st=50)
    plot_fda(iter=200, time=250, d=2, st=30)

# | (0.9235080256482227, 0.3730594356039615) (0.10095599101037045, 0.8966567966884782) [(0.4656925080561475, 0.793280766163747), (0.18613964685037554, 0.6556723858644705), (0.6391438576058391, 0.9580290079442022), (0.9260149088490528, 0.973267465364533), (0.3614713300962078, 0.4431013868540177), (0.11633021957321754, 0.5441216740709701), (0.15280714524003747, 0.1740009135732964), (0.1844044127118929, 0.6724679859686815), (0.2397420849165024, 0.060286302285712745), (0.8147625082136195, 0.4131591473120573), (0.053440899097773054, 0.7416168249315344), (0.06456724853732954, 0.9257159559085006), (0.39948246789607256, 0.338857485331576), (0.7425421805400814, 0.21527838679207412), (0.8820167759170637, 0.01031336073334288), (0.4133464722231903, 0.0038604229422691816), (0.25622792960932195, 0.3084816780850975), (0.1617857890651253, 0.4071856001672136), (0.22930297947397782, 0.3527812860010625), (0.7392257129360158, 0.5154299253729652), (0.9233955913297499, 0.5109749233177631), (0.10563914514612616, 0.15273329734988805), (0.778275302989325, 0.9650511876672754), (0.31924314195716375, 0.48103363938949717), (0.310383819440767, 0.7374738111756481), (0.34639018393504284, 0.9450194922315), (0.25134073011518754, 0.3267807646660321), (0.8294872079469048, 0.9073165798365709), (0.6343294966244615, 0.22443849352770406), (0.9517249527945246, 0.34611481230553176), (0.9305636815779054, 0.04338054503969302), (0.5411948037743048, 0.5994335749523427), (0.725680533595271, 0.8120082557370968), (0.8581443545575754, 0.2865567152393508), (0.7814861853794433, 0.40407491806768214), (0.7255582599288769, 0.6114228421925509), (0.9972036975910568, 0.2538116216948971), (0.40607962246374085, 0.8064089376204544), (0.8720899842539175, 0.8982586219515918), (0.06577903604245872, 0.6559474785154276), (0.2294022416946807, 0.45091095910077017), (0.8702917758003543, 0.8568921924196063), (0.5931936483097348, 0.011712487004782779), (0.007656832504073452, 0.9953699055981401), (0.05806363610707277, 0.812907863698026), (0.7507363523403097, 0.17167413799446518), (0.19861003884381712, 0.9585383667210126), (0.732272859569644, 0.8723383282330914), (0.767922668273956, 0.4127969768433253), (0.5721202830567642, 0.4585558766226171), (0.4180840978097995, 0.3438797464742891), (0.041720303877728626, 0.6092859417435875), (0.4069715587787317, 0.694534311781199), (0.41219707352925017, 0.7957137059028128), (0.5541326398755594, 0.9690733273175306), (0.49503770148483894, 0.35777760555162375), (0.17638165037100506, 0.8206536426535324), (0.8473629185986296, 0.25969260659398774), (0.5329132681672749, 0.15835462239590958), (0.9355941042508492, 0.45588095850368926), (0.8138359539853119, 0.9517649028486022), (0.09268480955533598, 0.11618119703801999), (0.15663185262130264, 0.7744051100839695), (0.5297733448106271, 0.5648051353196947), (0.6434344042081142, 0.7212014404847759), (0.07623084218430831, 0.7584120450730125), (0.5344724158730868, 0.759123162586681), (0.7753328111088033, 0.6727951785509038), (0.7303440079174355, 0.803398042962119), (0.24665205302777737, 0.5698913428196507), (0.8263300059265509, 0.938682072736202), (0.5433601924547923, 0.41312460004786655), (0.7298740291307598, 0.07270961667524023), (0.4086114066775264, 0.6756835330352423), (0.5868030321982018, 0.46783674330070846), (0.43373674466523293, 0.37622660461437996), (0.11424643750842334, 0.5792631625393865), (0.7106445297712263, 0.254433026682778), (0.44155623647295217, 0.6028235049321282), (0.9324076373230721, 0.5845155466149856), (0.8421621255679071, 0.09685939559040169), (0.7880655896993697, 0.22081700856350894), (0.3026613942948797, 0.7418925854919696), (0.5855562453747107, 0.664288702197936), (0.5532278187634299, 0.07589031186230899), (0.0989046021488339, 0.7168122286432431), (0.5440826056533389, 0.8893673421602751), (0.36417739647151937, 0.8256503989367291), (0.8216411529720733, 0.9261805235556758), (0.8559378994809744, 0.6102954303716138), (0.4128276810522803, 0.621804467311643), (0.5832541325618843, 0.8184821732015651), (0.81540858686587, 0.2997322180797738), (0.3768850641147381, 0.6188000935015204), (0.48409047617806955, 0.6714819529087022), (0.03789613656769575, 0.6553380939751823), (0.613960543743613, 0.041720998103005225), (0.9301732558528006, 0.3243229902072361)]
