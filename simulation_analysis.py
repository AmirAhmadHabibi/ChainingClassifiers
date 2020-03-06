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

from scipy.spatial import ConvexHull
from utilitarian import Progresser

# colours = [qualitative.Vivid_10.mpl_colors[1], qualitative.Vivid_10.mpl_colors[0], qualitative.Vivid_10.mpl_colors[2]]
colours = qualitative.Vivid_10.mpl_colors
markers = ['^', 'o']


def fishers_linear_discriminant(ps):
    ps = [np.array(p) for p in ps]
    # print(a.shape)
    if ps[0].shape[1] == 1:
        sum_of_dist = 0
        for i in range(len(ps) - 1):
            for j in range(i, len(ps)):
                a, b = ps[i], ps[j]
                sum_of_dist += np.nan_to_num(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)))
        sum_of_var = sum(np.nan_to_num(np.var(x)) for x in ps)
        return sum_of_dist / sum_of_var
    else:
        sum_of_dist = 0
        for i in range(len(ps) - 1):
            for j in range(i, len(ps)):
                a, b = ps[i], ps[j]
                sum_of_dist += np.nan_to_num(np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0)))
        sum_of_var = sum(np.nan_to_num(np.linalg.det(np.cov(x.T))) for x in ps)
        return sum_of_dist / sum_of_var


def convex_hall_area(points):
    areas = []
    # print(points)
    for ps in points:
        if len(ps) < 3:
            areas.append(0)
            continue
        hull = ConvexHull(np.array(ps))
        # areas.append(PolyArea2D(hull.simplices))
        areas.append(hull.volume)
    # print('+++++')
    return sum(areas) / float(len(areas))


def vector_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return exp(- np.sum((a - b) ** 2))


def is_separate(centers, thr):
    sep = True
    for i in range(len(centers) - 1):
        for j in range(i, len(centers)):
            if vector_similarity(centers[i], centers[j]) < thr:
                sep = False
    return sep


def create_random_sample(n=100, d=2, c=2):
    while True:
        try:
            points = []
            for i in range(n + c):
                points.append(tuple([random.random() for _ in range(d)]))
                # points.append((random.randint(0,1000), random.randint(0,1000)))
            centers = [random.choice(points) for _ in range(c)]
            min_thr = vector_similarity(np.full(d, 0), np.full(d, 2 / c))
            while not is_separate(centers, min_thr):
                centers = [random.choice(points) for _ in range(c)]

            for point in centers:
                points.remove(point)
            return points, centers
        except:
            pass


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


def predict_next_point(point, cnt_points, method, cnt_prt=None):
    if 'nn' in method:
        k = int(method[0:-2])
        if sum(len(x) for x in cnt_points) <= k:
            k = 1
        # k = min(int(method[0:-2]),len(a_points)+len(b_points))
        max_sim = [(0, '') for _ in range(k)]
        for cat, x_points in enumerate(cnt_points):
            for word in x_points:
                try:
                    sim = vector_similarity(word, point)
                    if sim > max_sim[0][0]:
                        max_sim[0] = (sim, cat)
                    max_sim = sorted(max_sim, key=lambda t: t[0])
                except Exception as e:
                    print(10, e)
        nn_count = {cat: 0 for cat in range(len(cnt_points))}
        max_n = 0
        for x in max_sim:
            nn_count[x[1]] += 1
            max_n = max(max_n, nn_count[x[1]])
        best_cats = [cat for cat in range(len(cnt_points)) if nn_count[cat] == max_n]

        best_cat = random.choice(best_cats)
        # elif self.prior_dist == 'items' or self.prior_dist == 'frequency':
        #     best_cats = {bc: p_c[bc] for bc in best_cats}
        #     best_cat = sorted(best_cats.items(), key=lambda kv: kv[1])[-1][0]
        cnt_points[best_cat].append(point)
    else:
        if cnt_prt is None:
            cnt_prt = [None for i in range(len(cnt_points))]
        max_sim = 0
        max_cnt = 0
        for i, x in enumerate(cnt_points):
            s = cat_similarity(point, x, method, cnt_prt[i])
            if s > max_sim:
                max_sim = s
                max_cnt = i

        cnt_points[max_cnt].append(point)
    return cnt_points


def plot_scatters(ax, cnt_points, rest, title, cnt_orig):
    labels = 'ABCDEF'
    legend_scat = []
    legend_name = []
    for i, x in enumerate(cnt_points):
        for w in x:
            if w != cnt_orig[i]:
                ax.scatter(w[0], w[1], marker=markers[1], label=labels[i], alpha=0.9, facecolor=colours[i])
        dot = ax.scatter([], [], marker=markers[1], label=labels[i], alpha=0.9, facecolor=colours[i])
        legend_scat.append(dot)
        legend_name.append('category ' + labels[i])

    for w in rest:
        ax.scatter(w[0], w[1], marker=markers[1], label='yet-to-emerge', alpha=0.3, facecolor='grey')
    dot = ax.scatter([], [], marker=markers[1], label='yet-to-emerge', alpha=0.3, facecolor='grey')
    legend_scat.append(dot)
    legend_name.append('yet-to-emerge')

    for i, x in enumerate(cnt_points):
        ax.scatter(cnt_orig[i][0], cnt_orig[i][1], marker=markers[0], alpha=0.6, s=110, facecolor=colours[i])
        dot = plt.scatter([], [], marker=markers[0], alpha=0.6, s=110, facecolor=colours[i])
        legend_scat.append(dot)
        legend_name.append('category ' + labels[i] + ' first point')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title)
    return legend_scat, legend_name


def plot_all(d=2, c=2, points=None, cnt=None):
    if points is None:
        points, cnt = create_random_sample(n=100, d=d, c=c)
    else:
        points, cnt = points, cnt
    print('|', cnt, points)
    rest = set(points)
    points = [None] + points
    cnt_e = [[x] for x in cnt]
    cnt_p = [[x] for x in cnt]
    cnt_sp = [[x] for x in cnt]
    # a_p, b_p = [a], [b]
    # a_sp, b_sp = [a], [b]

    check_steps = {60, 80, 100}
    g2g = True
    prog = Progresser(len(points))
    for year, wrd in enumerate(points):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        if wrd is not None:
            rest.remove(wrd)
            # exemlplar model
            cnt_e = predict_next_point(wrd, cnt_e, method='avg')
            fda_e = fishers_linear_discriminant(cnt_e)
            areas_e = convex_hall_area(cnt_e)

            # prototype model
            cnt_prt_vec = [np.mean(x, axis=0) for x in cnt_p]
            cnt_p = predict_next_point(wrd, cnt_p, 'pt', cnt_prt_vec)
            fda_p = fishers_linear_discriminant(cnt_p)
            areas_p = convex_hall_area(cnt_p)
            # static prototype model
            if year <= 10:
                cnt_spt_vec = [np.mean(x, axis=0) for x in cnt_sp]
            cnt_sp = predict_next_point(wrd, cnt_sp, 'spt', cnt_spt_vec)
            fda_sp = fishers_linear_discriminant(cnt_sp)
            areas_sp = convex_hall_area(cnt_sp)

            legend_scat, legend_name = plot_scatters(ax[0], cnt_e, rest, 'Exemplar Model | iter. ' + str(year) +
                                                     '\n cat size = ' + str(round(areas_e, 2)) +
                                                     ' | discrim. = ' + str(round(fda_e, 1))
                                                     , cnt)
            legend_scat, legend_name = plot_scatters(ax[1], cnt_p, rest, 'Prototype Model | iter. ' + str(year) +
                                                     '\n cat size = ' + str(round(areas_p, 2)) +
                                                     ' | discrim. = ' + str(round(fda_p, 1)), cnt)
            legend_scat, legend_name = plot_scatters(ax[2], cnt_sp, rest,
                                                     'Static Prototype Model | iter. ' + str(year) +
                                                     '\n cat size = ' + str(round(areas_sp, 2)) +
                                                     ' | discrim. = ' + str(round(fda_sp, 1)), cnt)

            if year in check_steps:
                scores = [fda_sp, fda_p]
                count = 0
                for s in scores:
                    if fda_e < s:
                        count += 1
                if count > 0:
                    g2g = False

        else:
            legend_scat, legend_name = plot_scatters(ax[0], cnt_e, rest, 'Exemplar Model | Iter. ' + str(year), cnt)
            legend_scat, legend_name = plot_scatters(ax[1], cnt_p, rest, 'Prototype Model | Iter. ' + str(year), cnt)
            legend_scat, legend_name = plot_scatters(ax[2], cnt_sp, rest,
                                                     'Static Prototype Model | Iter. ' + str(year), cnt)

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
    prog = Progresser(iter)
    for i in range(iter):
        smpl_size = []
        points, cnt = create_random_sample(n=time, d=d, c=3)
        rest = set(points)
        cnt_e = [[x] for x in cnt]
        cnt_p = [[x] for x in cnt]
        cnt_sp = [[x] for x in cnt]
        # a_e, b_e = [a], [b]
        # a_p, b_p = [a], [b]
        # a_sp, b_sp = [a], [b]
        for year, wrd in enumerate(points):
            rest.remove(wrd)
            smpl_size.append(sum(len(x) for x in cnt_e))

            # exemlplar model
            cnt_e = predict_next_point(wrd, cnt_e, method='avg')
            fda_e[i].append(fishers_linear_discriminant(cnt_e))

            # prototype model
            cnt_prt_vec = [np.mean(x, axis=0) for x in cnt_p]
            cnt_p = predict_next_point(wrd, cnt_p, 'pt', cnt_prt_vec)
            fda_p[i].append(fishers_linear_discriminant(cnt_p))
            # static prototype model
            if year <= 10:
                cnt_spt_vec = [np.mean(x, axis=0) for x in cnt_sp]
            cnt_sp = predict_next_point(wrd, cnt_sp, 'spt', cnt_spt_vec)
            fda_sp[i].append(fishers_linear_discriminant(cnt_sp))
        prog.count()

    smpl_sz_sqrt = np.power(np.array(smpl_size), np.full(len(smpl_size), 0.5))
    fda_e_ci = np.divide(np.std(fda_e, axis=0), smpl_sz_sqrt)
    fda_e = np.mean(fda_e, axis=0)
    fda_p_ci = np.divide(np.std(fda_p, axis=0), smpl_sz_sqrt)
    fda_p = np.mean(fda_p, axis=0)
    fda_sp_ci = np.divide(np.std(fda_sp, axis=0), smpl_sz_sqrt)
    fda_sp = np.mean(fda_sp, axis=0)
    # print(fda_e)
    # print(fda_p)
    # print(fda_sp)

    fda_e, fda_p, fda_sp = fda_e[st:], fda_p[st:], fda_sp[st:]
    fda_e_ci, fda_p_ci, fda_sp_ci = fda_e_ci[st:], fda_p_ci[st:], fda_sp_ci[st:]
    min_fda = min(min(fda_p - fda_p_ci), min(fda_e - fda_e_ci), min(fda_sp - fda_sp_ci))
    max_fda = max(max(fda_p + fda_p_ci), max(fda_e + fda_e_ci), max(fda_sp + fda_sp_ci))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x = range(st, len(fda_e) + st)

    ax.plot(x, fda_e, label='Exemplar Model', color=colours[0])
    ax.fill_between(x, fda_e + fda_e_ci, fda_e - fda_e_ci, color=colours[0], alpha=.3)

    ax.plot(x, fda_p, label='Prototype Model', color=colours[1])
    ax.fill_between(x, fda_p + fda_p_ci, fda_p - fda_p_ci, color=colours[1], alpha=.3)

    ax.plot(x, fda_sp, label='Static Prototype Model', color=colours[2])
    ax.fill_between(x, fda_sp + fda_sp_ci, fda_sp - fda_sp_ci, color=colours[2], alpha=.3)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_title('Fisher\'s Linear Discriminant over time - ' + str(iter) + ' iterations - ' + str(d) + 'D')
    ax.set_ylim(min_fda, max_fda)
    ax.set_ylabel('Fisher\'s Linear Discriminant')

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


def plot_convex_hull_area(iter=100, time=200, d=2, st=20):
    areas_e = [[] for _ in range(iter)]
    areas_p = [[] for _ in range(iter)]
    areas_sp = [[] for _ in range(iter)]
    prog = Progresser(iter)
    for i in range(iter):

        smpl_size = []
        points, cnt = create_random_sample(n=time, d=d, c=3)
        rest = set(points)
        cnt_e = [[x] for x in cnt]
        cnt_p = [[x] for x in cnt]
        cnt_sp = [[x] for x in cnt]
        # a_e, b_e = [a], [b]
        # a_p, b_p = [a], [b]
        # a_sp, b_sp = [a], [b]
        for year, wrd in enumerate(points):
            rest.remove(wrd)
            smpl_size.append(sum(len(x) for x in cnt_e))

            # exemlplar model
            cnt_e = predict_next_point(wrd, cnt_e, method='avg')
            areas_e[i].append(convex_hall_area(cnt_e))

            # prototype model
            cnt_prt_vec = [np.mean(x, axis=0) for x in cnt_p]
            cnt_p = predict_next_point(wrd, cnt_p, 'pt', cnt_prt_vec)
            areas_p[i].append(convex_hall_area(cnt_p))
            # static prototype model
            if year <= 10:
                cnt_spt_vec = [np.mean(x, axis=0) for x in cnt_sp]
            cnt_sp = predict_next_point(wrd, cnt_sp, 'spt', cnt_spt_vec)
            areas_sp[i].append(convex_hall_area(cnt_sp))
        prog.count()

    # areas_e=
    smpl_sz_sqrt = np.power(np.array(smpl_size), np.full(len(smpl_size), 0.5))
    areas_e_ci = np.divide(np.std(areas_e, axis=0), smpl_sz_sqrt)
    areas_e = np.mean(areas_e, axis=0)
    areas_p_ci = np.divide(np.std(areas_p, axis=0), smpl_sz_sqrt)
    areas_p = np.mean(areas_p, axis=0)
    areas_sp_ci = np.divide(np.std(areas_sp, axis=0), smpl_sz_sqrt)
    areas_sp = np.mean(areas_sp, axis=0)
    # print(fda_e)
    # print(fda_p)
    # print(fda_sp)

    areas_e, areas_p, areas_sp = areas_e[st:], areas_p[st:], areas_sp[st:]
    areas_e_ci, areas_p_ci, areas_sp_ci = areas_e_ci[st:], areas_p_ci[st:], areas_sp_ci[st:]
    min_area = min(min(areas_p - areas_p_ci), min(areas_e - areas_e_ci), min(areas_sp - areas_sp_ci))
    max_area = max(max(areas_p + areas_p_ci), max(areas_e + areas_e_ci), max(areas_sp + areas_sp_ci))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    x = range(st, len(areas_e) + st)

    ax.plot(x, areas_e, label='Exemplar Model', color=colours[0])
    ax.fill_between(x, areas_e + areas_e_ci, areas_e - areas_e_ci, color=colours[0], alpha=.3)

    ax.plot(x, areas_p, label='Prototype Model', color=colours[1])
    ax.fill_between(x, areas_p + areas_p_ci, areas_p - areas_p_ci, color=colours[1], alpha=.3)

    ax.plot(x, areas_sp, label='Static Prototype Model', color=colours[2])
    ax.fill_between(x, areas_sp + areas_sp_ci, areas_sp - areas_sp_ci, color=colours[2], alpha=.3)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_title('Average area of the convex hull of the categories over time - ' + str(iter) + ' iterations - ' + str(d) + 'D')
    ax.set_ylim(min_area, max_area)
    ax.set_ylabel('Mean category size')

    ax.legend(loc='lower left', ncol=1, fontsize=8)
    fig.savefig(
        './predictions/simulation/convex_area_exemplar_vs_prototype_vs_stprot-t' + str(time) + '-i' + str(
            iter) + '-d' + str(
            d) + '.png',
        bbox_inches='tight')
    fig.savefig(
        './predictions/simulation/convex_area_exemplar_vs_prototype_vs_stprot-t' + str(time) + '-i' + str(
            iter) + '-d' + str(
            d) + '.pdf',
        bbox_inches='tight')
    fig.clear()
    plt.clf()


if __name__ == "__main__":
    # i = 0
    # while not plot_all(c=3):
    #     i += 1
    #     print(i, 'run.')
    cnt = [(0.1455432018041226, 0.9134842707220673), (0.22463687880229266, 0.19500885307951743),
           (0.4011271971048177, 0.7336546095748743)]
    points = [(0.5818045384331219, 0.5762358052179433), (0.028035162291188187, 0.7061287119502047),
              (0.21375665714608838, 0.9128429102358602), (0.798150828685189, 0.11408850152680827),
              (0.2821457652238676, 0.5825437879807353), (0.22558857801283239, 0.4172394772091109),
              (0.4338611580900049, 0.4846142005442945), (0.4008927530732457, 0.850943414565061),
              (0.3905726904605953, 0.6669301741337843), (0.9596292892913569, 0.3231486909935637),
              (0.2116835472310724, 0.12543378682227802), (0.5138820297702353, 0.9028903931498934),
              (0.058248580280712114, 0.8671941127942852), (0.45553975971048344, 0.8752850703523601),
              (0.12453422816978965, 0.6247195991283194), (0.4803428789549109, 0.066103524144707),
              (0.2765549814547513, 0.28761505823862765), (0.5604461024913889, 0.761424389632871),
              (0.04072852603830668, 0.25988170999066273), (0.3385266600824549, 0.9462956501544023),
              (0.7813870349181707, 0.7083115049280515), (0.2963579289177536, 0.6321680431554291),
              (0.8737021800272069, 0.5420768234904341), (0.6148154032042411, 0.33945559499791733),
              (0.7828289490649326, 0.2976753467527984), (0.37608040632918727, 0.8184214474669564),
              (0.3095016854809721, 0.5712800915856474), (0.005804051207783711, 0.3204580104419855),
              (0.43014781331531615, 0.09556886669465803), (0.36582729597727726, 0.4005413634636512),
              (0.017698084721369023, 0.9073597121493749), (0.15065902966382583, 0.6265503598982497),
              (0.8668514762144213, 0.07859407004855345), (0.4795915548713491, 0.6772366417948011),
              (0.5319990008271651, 0.17071634437608718), (0.5580419894901678, 0.3411658515369196),
              (0.5786224955083621, 0.01952074490828537), (0.7175801993669316, 0.4590439503732886),
              (0.3933012848282238, 0.5969904644843461), (0.06881725800293848, 0.4198893487412948),
              (0.8581116080428534, 0.4413938008202517), (0.5555470549373404, 0.5392079687634297),
              (0.10214899488091345, 0.423422414942514), (0.5755362094589911, 0.3343582683909603),
              (0.2876850372603157, 0.30700779758777963), (0.771458005234639, 0.3606347952980511),
              (0.2621831487023648, 0.8806886362029063), (0.4519275202709272, 0.09275203726224668),
              (0.2195648367496803, 0.623386595044182), (0.7278442327971089, 0.025188298345723226),
              (0.5953274084371022, 0.09706215609589652), (0.8336083270879262, 0.2559451108872227),
              (0.4273027108468115, 0.5910734064840442), (0.7126775275536252, 0.35503987388810376),
              (0.725505943744959, 0.41083889900385073), (0.9434891101020579, 0.9160064793243932),
              (0.2632783528888626, 0.5433793923900541), (0.5025574753006001, 0.2617333619300264),
              (0.9439404333553563, 0.7000409570860252), (0.5118245407102264, 0.8801076464833872),
              (0.7925219293996917, 0.13763854000851627), (0.25337856752302934, 0.6947421833932887),
              (0.2948163098399307, 0.23933283021590546), (0.3561327921210673, 0.3374211354965574),
              (0.8954231054300427, 0.3873831442413146), (0.9863833254978025, 0.6427294806196349),
              (0.022907878430291517, 0.18479181743283057), (0.9724954445834187, 0.17754774238086923),
              (0.733109114309614, 0.04164686183743871), (0.08532113244581041, 0.5331216868826446),
              (0.9967921994851785, 0.13444158453993893), (0.7446855243893052, 0.4601415326948991),
              (0.5169491669692706, 0.43471953169230626), (0.25912995984145193, 0.9653574049049953),
              (0.7364228586092129, 0.07842521854853035), (0.46481510619558086, 0.12274487818068092),
              (0.34033323535337345, 0.823575450024714), (0.5158167935580117, 0.09720469943029564),
              (0.10992886223330023, 0.5188427892430926), (0.28887084296638665, 0.7863266606375819),
              (0.5523299969724729, 0.7717068549970892), (0.653551857568527, 0.6535269538348255),
              (0.0017001268577012674, 0.5389358503129809), (0.9255746540551079, 0.41887730455531147),
              (0.016309329361204772, 0.754638019330504), (0.8563703027311936, 0.9552889559841518),
              (0.06062465144530382, 0.2031202844490635), (0.12392941695119764, 0.2761406427570521),
              (0.5586419831233649, 0.6680843798809987), (0.2871956347581025, 0.6736911609110398),
              (0.469988519722347, 0.7732688736351121), (0.8837403073127701, 0.06540736407688308),
              (0.38790501364325114, 0.781893600562396), (0.5863469430581794, 0.3205421055710378),
              (0.022438659988085696, 0.9463880540643553), (0.5995058153448833, 0.576731821920926),
              (0.9872137749714829, 0.7948973733356186), (0.3520721242686091, 0.7021856625560716),
              (0.594270312836369, 0.700610346078838), (0.44512309139067263, 0.7600055589712579)]

    # plot_all(c=3,points=points,cnt=cnt)

    # plot_fda(iter=1000, time=200, d=1)
    # ds = [1, 2, 3, 4, 5, 10]
    # for d in ds:
    #     plot_fda(iter=2000, time=300, d=d, st=50)
    plot_fda(iter=500, time=300, d=2, st=30)
    # plot_convex_hull_area(iter=500, time=300, d=2, st=30)

# | (0.9235080256482227, 0.3730594356039615) (0.10095599101037045, 0.8966567966884782) [(0.4656925080561475, 0.793280766163747), (0.18613964685037554, 0.6556723858644705), (0.6391438576058391, 0.9580290079442022), (0.9260149088490528, 0.973267465364533), (0.3614713300962078, 0.4431013868540177), (0.11633021957321754, 0.5441216740709701), (0.15280714524003747, 0.1740009135732964), (0.1844044127118929, 0.6724679859686815), (0.2397420849165024, 0.060286302285712745), (0.8147625082136195, 0.4131591473120573), (0.053440899097773054, 0.7416168249315344), (0.06456724853732954, 0.9257159559085006), (0.39948246789607256, 0.338857485331576), (0.7425421805400814, 0.21527838679207412), (0.8820167759170637, 0.01031336073334288), (0.4133464722231903, 0.0038604229422691816), (0.25622792960932195, 0.3084816780850975), (0.1617857890651253, 0.4071856001672136), (0.22930297947397782, 0.3527812860010625), (0.7392257129360158, 0.5154299253729652), (0.9233955913297499, 0.5109749233177631), (0.10563914514612616, 0.15273329734988805), (0.778275302989325, 0.9650511876672754), (0.31924314195716375, 0.48103363938949717), (0.310383819440767, 0.7374738111756481), (0.34639018393504284, 0.9450194922315), (0.25134073011518754, 0.3267807646660321), (0.8294872079469048, 0.9073165798365709), (0.6343294966244615, 0.22443849352770406), (0.9517249527945246, 0.34611481230553176), (0.9305636815779054, 0.04338054503969302), (0.5411948037743048, 0.5994335749523427), (0.725680533595271, 0.8120082557370968), (0.8581443545575754, 0.2865567152393508), (0.7814861853794433, 0.40407491806768214), (0.7255582599288769, 0.6114228421925509), (0.9972036975910568, 0.2538116216948971), (0.40607962246374085, 0.8064089376204544), (0.8720899842539175, 0.8982586219515918), (0.06577903604245872, 0.6559474785154276), (0.2294022416946807, 0.45091095910077017), (0.8702917758003543, 0.8568921924196063), (0.5931936483097348, 0.011712487004782779), (0.007656832504073452, 0.9953699055981401), (0.05806363610707277, 0.812907863698026), (0.7507363523403097, 0.17167413799446518), (0.19861003884381712, 0.9585383667210126), (0.732272859569644, 0.8723383282330914), (0.767922668273956, 0.4127969768433253), (0.5721202830567642, 0.4585558766226171), (0.4180840978097995, 0.3438797464742891), (0.041720303877728626, 0.6092859417435875), (0.4069715587787317, 0.694534311781199), (0.41219707352925017, 0.7957137059028128), (0.5541326398755594, 0.9690733273175306), (0.49503770148483894, 0.35777760555162375), (0.17638165037100506, 0.8206536426535324), (0.8473629185986296, 0.25969260659398774), (0.5329132681672749, 0.15835462239590958), (0.9355941042508492, 0.45588095850368926), (0.8138359539853119, 0.9517649028486022), (0.09268480955533598, 0.11618119703801999), (0.15663185262130264, 0.7744051100839695), (0.5297733448106271, 0.5648051353196947), (0.6434344042081142, 0.7212014404847759), (0.07623084218430831, 0.7584120450730125), (0.5344724158730868, 0.759123162586681), (0.7753328111088033, 0.6727951785509038), (0.7303440079174355, 0.803398042962119), (0.24665205302777737, 0.5698913428196507), (0.8263300059265509, 0.938682072736202), (0.5433601924547923, 0.41312460004786655), (0.7298740291307598, 0.07270961667524023), (0.4086114066775264, 0.6756835330352423), (0.5868030321982018, 0.46783674330070846), (0.43373674466523293, 0.37622660461437996), (0.11424643750842334, 0.5792631625393865), (0.7106445297712263, 0.254433026682778), (0.44155623647295217, 0.6028235049321282), (0.9324076373230721, 0.5845155466149856), (0.8421621255679071, 0.09685939559040169), (0.7880655896993697, 0.22081700856350894), (0.3026613942948797, 0.7418925854919696), (0.5855562453747107, 0.664288702197936), (0.5532278187634299, 0.07589031186230899), (0.0989046021488339, 0.7168122286432431), (0.5440826056533389, 0.8893673421602751), (0.36417739647151937, 0.8256503989367291), (0.8216411529720733, 0.9261805235556758), (0.8559378994809744, 0.6102954303716138), (0.4128276810522803, 0.621804467311643), (0.5832541325618843, 0.8184821732015651), (0.81540858686587, 0.2997322180797738), (0.3768850641147381, 0.6188000935015204), (0.48409047617806955, 0.6714819529087022), (0.03789613656769575, 0.6553380939751823), (0.613960543743613, 0.041720998103005225), (0.9301732558528006, 0.3243229902072361)]

# number 8:
# | [(0.1455432018041226, 0.9134842707220673), (0.22463687880229266, 0.19500885307951743), (0.4011271971048177, 0.7336546095748743)] [(0.5818045384331219, 0.5762358052179433), (0.028035162291188187, 0.7061287119502047), (0.21375665714608838, 0.9128429102358602), (0.798150828685189, 0.11408850152680827), (0.2821457652238676, 0.5825437879807353), (0.22558857801283239, 0.4172394772091109), (0.4338611580900049, 0.4846142005442945), (0.4008927530732457, 0.850943414565061), (0.3905726904605953, 0.6669301741337843), (0.9596292892913569, 0.3231486909935637), (0.2116835472310724, 0.12543378682227802), (0.5138820297702353, 0.9028903931498934), (0.058248580280712114, 0.8671941127942852), (0.45553975971048344, 0.8752850703523601), (0.12453422816978965, 0.6247195991283194), (0.4803428789549109, 0.066103524144707), (0.2765549814547513, 0.28761505823862765), (0.5604461024913889, 0.761424389632871), (0.04072852603830668, 0.25988170999066273), (0.3385266600824549, 0.9462956501544023), (0.7813870349181707, 0.7083115049280515), (0.2963579289177536, 0.6321680431554291), (0.8737021800272069, 0.5420768234904341), (0.6148154032042411, 0.33945559499791733), (0.7828289490649326, 0.2976753467527984), (0.37608040632918727, 0.8184214474669564), (0.3095016854809721, 0.5712800915856474), (0.005804051207783711, 0.3204580104419855), (0.43014781331531615, 0.09556886669465803), (0.36582729597727726, 0.4005413634636512), (0.017698084721369023, 0.9073597121493749), (0.15065902966382583, 0.6265503598982497), (0.8668514762144213, 0.07859407004855345), (0.4795915548713491, 0.6772366417948011), (0.5319990008271651, 0.17071634437608718), (0.5580419894901678, 0.3411658515369196), (0.5786224955083621, 0.01952074490828537), (0.7175801993669316, 0.4590439503732886), (0.3933012848282238, 0.5969904644843461), (0.06881725800293848, 0.4198893487412948), (0.8581116080428534, 0.4413938008202517), (0.5555470549373404, 0.5392079687634297), (0.10214899488091345, 0.423422414942514), (0.5755362094589911, 0.3343582683909603), (0.2876850372603157, 0.30700779758777963), (0.771458005234639, 0.3606347952980511), (0.2621831487023648, 0.8806886362029063), (0.4519275202709272, 0.09275203726224668), (0.2195648367496803, 0.623386595044182), (0.7278442327971089, 0.025188298345723226), (0.5953274084371022, 0.09706215609589652), (0.8336083270879262, 0.2559451108872227), (0.4273027108468115, 0.5910734064840442), (0.7126775275536252, 0.35503987388810376), (0.725505943744959, 0.41083889900385073), (0.9434891101020579, 0.9160064793243932), (0.2632783528888626, 0.5433793923900541), (0.5025574753006001, 0.2617333619300264), (0.9439404333553563, 0.7000409570860252), (0.5118245407102264, 0.8801076464833872), (0.7925219293996917, 0.13763854000851627), (0.25337856752302934, 0.6947421833932887), (0.2948163098399307, 0.23933283021590546), (0.3561327921210673, 0.3374211354965574), (0.8954231054300427, 0.3873831442413146), (0.9863833254978025, 0.6427294806196349), (0.022907878430291517, 0.18479181743283057), (0.9724954445834187, 0.17754774238086923), (0.733109114309614, 0.04164686183743871), (0.08532113244581041, 0.5331216868826446), (0.9967921994851785, 0.13444158453993893), (0.7446855243893052, 0.4601415326948991), (0.5169491669692706, 0.43471953169230626), (0.25912995984145193, 0.9653574049049953), (0.7364228586092129, 0.07842521854853035), (0.46481510619558086, 0.12274487818068092), (0.34033323535337345, 0.823575450024714), (0.5158167935580117, 0.09720469943029564), (0.10992886223330023, 0.5188427892430926), (0.28887084296638665, 0.7863266606375819), (0.5523299969724729, 0.7717068549970892), (0.653551857568527, 0.6535269538348255), (0.0017001268577012674, 0.5389358503129809), (0.9255746540551079, 0.41887730455531147), (0.016309329361204772, 0.754638019330504), (0.8563703027311936, 0.9552889559841518), (0.06062465144530382, 0.2031202844490635), (0.12392941695119764, 0.2761406427570521), (0.5586419831233649, 0.6680843798809987), (0.2871956347581025, 0.6736911609110398), (0.469988519722347, 0.7732688736351121), (0.8837403073127701, 0.06540736407688308), (0.38790501364325114, 0.781893600562396), (0.5863469430581794, 0.3205421055710378), (0.022438659988085696, 0.9463880540643553), (0.5995058153448833, 0.576731821920926), (0.9872137749714829, 0.7948973733356186), (0.3520721242686091, 0.7021856625560716), (0.594270312836369, 0.700610346078838), (0.44512309139067263, 0.7600055589712579)]
