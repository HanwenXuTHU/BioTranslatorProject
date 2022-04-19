import pickle
import collections
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt


MEDIUM_SIZE = 8
SMALLER_SIZE = 6
BIGGER_SIZE = 25
plt.rc('font', family='Helvetica', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	# fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	# fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 2
FIG_WIDTH = 2


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def main():
    textomics_path = "/disk1/hwxu/protein2def/proteinTextomics/results/metrics/goa_human_eval_cl_fold_{}.pkl"
    nearest_path = "/disk1/hwxu/protein2def/proteinTextomics/results/metrics/goa_human_nst_fold_{}.pkl"
    n_fold = 5
    textomics_flat = collections.OrderedDict()
    nearest_flat = collections.OrderedDict()
    metrics = ['bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL',
               'meteor']
    for m in metrics:
        textomics_flat[m] = []
    for m in metrics:
        nearest_flat[m] = []
    for i in range(n_fold):
        textomics = load_obj(textomics_path.format(i))
        nearest = load_obj(nearest_path.format(i))
        for j in range(0, 5):
            textomics_flat[metrics[j]].append(textomics['bleu']['{}_avg'.format(metrics[j])])
        for j in range(5, 8):
            textomics_flat[metrics[j]].append(textomics['rouge']['{}_avg'.format(metrics[j])])
        textomics_flat['meteor'].append(textomics['meteor'])

        for j in range(0, 5):
            nearest_flat[metrics[j]].append(nearest['bleu']['{}_avg'.format(metrics[j])])
        for j in range(5, 8):
            nearest_flat[metrics[j]].append(nearest['rouge']['{}_avg'.format(metrics[j])])
        nearest_flat['meteor'].append(nearest['meteor'])

    mean = np.zeros([len(metrics), 2])
    yerr = np.zeros([len(metrics), 2])
    i = 0
    for m in metrics:
        mean[i, 0], mean[i, 1] = np.mean(textomics_flat[m]), np.mean(nearest_flat[m])
        yerr[i, 0], yerr[i, 1] = np.std(textomics_flat[m]), np.std(nearest_flat[m])
        i += 1

    fig, ax = plt.subplots(figsize=(1.6*FIG_WIDTH, FIG_HEIGHT))

    n_groups = np.size(mean, 0)
    nmethod = np.size(mean, 1)
    index = np.arange(n_groups)
    bar_width = 1. / nmethod * 0.8
    opacity = 1

    ax.bar(index + (nmethod - 1 - 1) * bar_width, mean[:, 0], yerr=yerr[:, 0], width=bar_width, alpha=opacity,
           color='#66c2a5',  # ,color_l[i],
           label='ProTranslator')
    ax.bar(index + (nmethod - 1 - 0) * bar_width, mean[:, 1], yerr=yerr[:, 1], width=bar_width, alpha=opacity,
           color='#fc8d62',  # ,color_l[i],
           label='Nearest term')

    csfont = {'family': 'Helvetica'}
    ax.set_xticklabels(metrics)

    if nmethod == 1:
        ax.set_xticks(index)
    else:
        ax.set_xticks(index + bar_width * (nmethod - 0.5) * 1. / 2 - 0.1)
    plt.legend(loc='upper right', frameon=False, ncol=1, fontsize=4)
    # plt.legend(loc='upper left',bbox_to_anchor=(0.1, 1.1), frameon=False, ncol=1, fontsize=4)

    # plt.setp(ax.get_xticklabels(), rotation=30, ha="right", va="center",
    #         rotation_mode="anchor")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    max_y = 0.6  # min(np.ceil(np.max(mean*10))/10,1.0)
    min_y = 0
    if min_y > 0.80:
        min_y = 0.80
    ax.set_ylim([min_y, max_y])
    if min_y < 0.80:
        step_size = 0.
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        step_size = 0.05
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    print(min_y, max_y)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    # ax.legend(method_l)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(0.5*abs(x1 - x0) / abs(y1 - y0))

    fig.tight_layout()
    plt.title('Protein Text Generation')
    #plt.show()
    plt.savefig('pdfs/protein_textomics_0.5.pdf')

    debug = 0


if __name__ == '__main__':
    main()